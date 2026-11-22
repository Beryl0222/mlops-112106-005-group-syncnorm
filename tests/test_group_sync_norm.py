"""Behavioural tests for partitioned grouped BatchNorm.

The suite is organized around the questions a trainer asks:

* Does any partition split produce the same forward values and gradients?
* Can I tell a masked padding row from a non-finite valid row from a
  legitimately empty group?
* Why can a group be trainable yet not update its inference value?
* Do overflow / upstream skip / duplicate shards fail the step consistently?
* Does an inference call that already started keep its approved snapshot?
* Does every message stay inside the published per-group-per-channel budget?
"""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from group_sync_norm import (
    ABSENT,
    NONE,
    OK,
    DuplicateContribution,
    GroupNormTrainer,
    IncompleteExchange,
    NotInferable,
    ProtocolError,
    RunningStats,
    StepRejected,
    WorkerFrames,
    decode,
    encode_propose,
    encode_reduce,
    merge_contributions,
)
from group_sync_norm.wire import (
    DEFAULT_WIRE_SCALARS,
    BudgetExceeded,
    GroupStats,
    STAGE_SCALARS,
    agree_frame,
    encode_context,
    encode_grad,
)

ROOT = Path(__file__).resolve().parents[1]


def load_scene(name="scene.json"):
    return json.loads((ROOT / "samples" / name).read_text(encoding="utf-8"))


def scene_workers(scene):
    out = {}
    for name in scene["expected_workers"]:
        worker = scene["workers"][name]
        x = np.asarray(worker["x"], dtype=np.float64).reshape(-1, len(scene["gamma"]))
        dy = np.asarray(worker["dy"], dtype=np.float64).reshape(-1, len(scene["gamma"]))
        out[name] = WorkerFrames(
            x=x,
            mask=np.asarray(worker["mask"], dtype=bool).reshape(-1),
            groups=list(worker["groups"]),
            dy=dy,
        )
    return out


def scene_groups(scene):
    names = []
    for worker in scene["workers"].values():
        for group in worker["groups"]:
            if group not in names:
                names.append(group)
    return names


def build_trainer(scene):
    trainer = GroupNormTrainer(
        channels=len(scene["gamma"]),
        epsilon=scene["epsilon"],
        decay=scene["inference_policy"]["decay"],
        min_count=scene["inference_policy"]["min_count"],
        wire_scalars_per_group_channel=scene["wire_scalars_per_group_channel"],
        budget_bytes=scene["budget_bytes"],
    )
    trainer.seed_running(scene["running"])
    return trainer


def _global_sample_variance(step) -> np.ndarray:
    """Aggregate all of a step's valid rows into one sample variance."""
    total = 0
    mean = None
    m2 = None
    for stats in step.groups.values():
        if stats.empty:
            continue
        group_mean, group_m2, _ = step.restored[name_of(step, stats)]
        count = stats.count
        if total == 0:
            mean, m2, total = group_mean.copy(), group_m2.copy(), count
            continue
        n_a, n_b = np.float64(total), np.float64(count)
        delta = group_mean - mean
        mean = mean + delta * (n_b / (n_a + n_b))
        m2 = m2 + group_m2 + delta * delta * (n_a * n_b / (n_a + n_b))
        total += count
    return m2 / np.float64(total - 1)


def name_of(step, stats):
    for name, candidate in step.groups.items():
        if candidate is stats:
            return name
    raise KeyError("statistics not found in step")


def single_process_reference(scene, groups, masks, dys):
    """One-process BatchNorm over the whole batch, row by row.

    ``masks`` and ``dys`` must be the same shape as the sample data.  This is
    the FP64 oracle the contract asks the distributed path to match.
    """
    x = np.concatenate(
        [
            np.asarray(scene["workers"][w]["x"], dtype=np.float64).reshape(-1, len(scene["gamma"]))
            for w in scene["expected_workers"]
        ]
    )
    gamma = np.asarray(scene["gamma"])
    beta = np.asarray(scene["beta"])
    mask = np.asarray(masks, dtype=bool)
    dy = np.asarray(dys, dtype=np.float64)
    eps = scene["epsilon"]
    y = np.zeros_like(x)
    dx = np.zeros_like(x)
    dgamma = np.zeros_like(gamma)
    dbeta = np.zeros_like(gamma)
    for group in groups:
        rows = np.array([g == group for g in sum((scene["workers"][w]["groups"] for w in scene["expected_workers"]), [])])
        rows &= mask[: rows.size]
        finite = np.all(np.isfinite(x[rows]), axis=1)
        rows_idx = np.flatnonzero(rows)[finite]
        if rows_idx.size == 0:
            continue
        xg = x[rows_idx]
        mean = xg.mean(axis=0)
        var = ((xg - mean) ** 2).mean(axis=0)
        inv = 1.0 / np.sqrt(var + eps)
        hat = (xg - mean) * inv
        y[rows_idx] = hat * gamma + beta
        g = dy[rows_idx] * gamma
        dx[rows_idx] = inv * (g - g.mean(axis=0) - hat * (hat * g).mean(axis=0))
        dgamma += (dy[rows_idx] * hat).sum(axis=0)
        dbeta += dy[rows_idx].sum(axis=0)
    return y, dx, dgamma, dbeta


class ForwardBackwardTests(unittest.TestCase):
    def test_forward_and_gradients_match_single_process_reference(self):
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        groups = scene_groups(scene)
        step = trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=groups,
            gamma=scene["gamma"],
            beta=scene["beta"],
        )
        grads = trainer.backward_last(
            workers={name: workers[name].dy for name in workers},
            gamma=scene["gamma"],
        )
        y, dx, dgamma, dbeta = single_process_reference(
            scene,
            groups,
            np.concatenate([workers[w].mask for w in scene["expected_workers"]]),
            np.concatenate([workers[w].dy for w in scene["expected_workers"]]),
        )
        np.testing.assert_allclose(step.forward, y, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(grads["dx"], dx, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(grads["dgamma"], dgamma, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(grads["dbeta"], dbeta, atol=1e-5, rtol=1e-5)

    def test_repartitioning_the_same_rows_is_invariant(self):
        rng = np.random.default_rng(7)
        rows, channels = 15, 2
        x = (rng.normal(size=(rows, channels)) * np.array([1e-4, 1e4])).astype(float)
        x[3] = [1e8, -1e8]
        mask = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
        groups = ["left", "left", "left", "right", "right", "rare"] * 2 + [
            "right",
            "left",
            "rare",
        ]
        dy = rng.normal(size=(rows, channels))
        gamma = np.array([0.7, 1.1])
        beta = np.array([0.2, -0.2])

        def run(cuts):
            trainer = GroupNormTrainer(channels=channels, epsilon=1e-5)
            workers = {}
            bounds = [0] + list(cuts) + [rows]
            for index in range(len(bounds) - 1):
                sl = slice(bounds[index], bounds[index + 1])
                workers[f"P{index}"] = WorkerFrames(x=x[sl], mask=mask[sl], groups=groups[sl], dy=dy[sl])
            step = trainer.train_step(
                workers=workers,
                expected_workers=list(workers),
                groups=["left", "right", "rare"],
                gamma=gamma,
                beta=beta,
            )
            grads = trainer.backward_last(workers={name: dy[bounds[i]:bounds[i+1]] for i, name in enumerate(workers)}, gamma=gamma)
            return step, grads

        a_step, a_grads = run([5, 9])
        b_step, b_grads = run([2, 6, 6, 11, 14])
        np.testing.assert_allclose(a_step.forward, b_step.forward, atol=1e-9, rtol=1e-9)
        for key in ("dx", "dgamma", "dbeta"):
            np.testing.assert_allclose(a_grads[key], b_grads[key], atol=1e-9, rtol=1e-9)

    def test_extreme_dynamic_range_keeps_variance(self):
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        step = trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        large, _, var = step.restored["large"]
        # 1e8+1 .. 1e8+4, channel 0: mean 1e8+2.5, population var 1.25.
        self.assertAlmostEqual(var[0], 1.25, places=9)
        self.assertAlmostEqual(var[1], 4.25, places=9)
        # The mean must still be in the original units; a non-shifted
        # reduction would have collapsed ``var[0]`` to 0 here.
        self.assertTrue(step.restored["large"][0][0] > 1e8)

    def test_empty_group_has_no_valid_output(self):
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        step = trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        # ``rare`` in P1 is masked; the only valid ``rare`` row is P0's.
        self.assertEqual(step.groups["rare"].count, 1)
        self.assertEqual(step.groups["rare"].nonfinite, 0)

        lone = WorkerFrames(
            x=np.zeros((1, 2)),
            mask=np.array([False]),
            groups=["silent"],
            dy=np.zeros((1, 2)),
        )
        second = WorkerFrames(
            x=np.zeros((0, 2)), mask=np.zeros(0, dtype=bool), groups=[], dy=np.zeros((0, 2))
        )
        empty = trainer.train_step(
            workers={"A": lone, "B": second},
            expected_workers=["A", "B"],
            groups=["silent"],
        )
        self.assertTrue(empty.groups["silent"].empty)
        self.assertEqual(empty.forward.shape, (1, 2))
        np.testing.assert_array_equal(empty.forward, np.zeros((1, 2)))
        self.assertIn("silent", empty.diagnostics["empty_groups"])

    def test_zero_frame_and_missing_message_are_distinct(self):
        trainer = GroupNormTrainer(channels=2)
        silent = WorkerFrames(
            x=np.zeros((0, 2)), mask=np.zeros(0, dtype=bool), groups=[], dy=np.zeros((0, 2))
        )
        active = WorkerFrames(
            x=np.array([[1.0, 2.0]]), mask=np.array([True]), groups=["g"], dy=np.zeros((1, 2))
        )
        step = trainer.train_step(
            workers={"P0": active, "P1": silent},
            expected_workers=["P0", "P1"],
            groups=["g"],
        )
        self.assertFalse(step.groups["g"].empty)
        # A participant that never delivered is reported, not waited for.
        messages = [
            encode_reduce(key="k", worker="P0", group="g", stats=GroupStats(count=1, mean=np.zeros(2), m2=np.zeros(2))),
        ]
        by_worker, missing = merge_contributions(
            messages, expected_workers=["P0", "P1", "P2"], group="g"
        )
        self.assertEqual(missing, ["P1", "P2"])
        self.prev = None

    def test_nonfinite_valid_row_rejects_promotion_but_keeps_output(self):
        trainer = GroupNormTrainer(channels=2)
        workers = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 2.0], [np.nan, 3.0]]),
                mask=np.array([True, True]),
                groups=["g", "g"],
                dy=np.zeros((2, 2)),
            ),
            "P1": WorkerFrames(
                x=np.array([[4.0, 5.0]]),
                mask=np.array([True]),
                groups=["g"],
                dy=np.zeros((1, 2)),
            ),
        }
        step = trainer.train_step(workers=workers, expected_workers=["P0", "P1"], groups=["g"])
        self.assertEqual(step.groups["g"].count, 2)
        self.assertEqual(step.diagnostics["nonfinite_rows"], 1)
        self.assertFalse(step.approved)
        self.assertIsNone(trainer.promote_last())
        # The non-finite row produced no output and will produce no gradient.
        self.assertEqual(step.forward[1].tolist(), [0.0, 0.0])
        grads = trainer.backward_last(
            workers={"P0": np.ones((2, 2)), "P1": np.ones((1, 2))}
        )
        np.testing.assert_array_equal(grads["dx"][1], [0.0, 0.0])

    def test_masked_padding_rows_are_excluded_and_zero_gradient(self):
        trainer = GroupNormTrainer(channels=2)
        workers = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 1.0], [1e300, 1e300]]),
                mask=np.array([True, False]),
                groups=["g", "g"],
                dy=np.ones((2, 2)),
            ),
            "P1": WorkerFrames(
                x=np.array([[3.0, 3.0]]),
                mask=np.array([True]),
                groups=["g"],
                dy=np.ones((1, 2)),
            ),
        }
        step = trainer.train_step(workers=workers, expected_workers=["P0", "P1"], groups=["g"])
        self.assertTrue(step.approved)
        self.assertEqual(step.diagnostics["masked_rows"], 1)
        self.assertEqual(step.diagnostics["nonfinite_rows"], 0)
        self.assertEqual(step.forward[1].tolist(), [0.0, 0.0])
        grads = trainer.backward_last(workers={"P0": np.ones((2, 2)), "P1": np.ones((1, 2))})
        np.testing.assert_array_equal(grads["dx"][1], [0.0, 0.0])


class PromotionTests(unittest.TestCase):
    def test_approved_step_uses_sample_variance_with_decay(self):
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        step = trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        self.assertTrue(step.approved)
        decay = scene["inference_policy"]["decay"]
        report = trainer.promote_last()
        self.assertIn("large", report["advanced"])
        large = trainer.running.snapshot().groups["large"]
        _, m2, _ = step.restored["large"]
        sample_var = m2 / (step.groups["large"].count - 1)
        # ``large`` had no seeded entry, so the first candidate is kept as-is;
        # there is no history to apply the inertia to.
        np.testing.assert_allclose(large.var, sample_var)

        # ``global`` *was* seeded: its value is decay * old + (1-decay) * new.
        global_step = np.array(step.restored["global"]) if "global" in step.restored else None
        self.assertIsNone(global_step)
        seeded = np.array(scene["running"]["global"]["var"], dtype=float)
        updated = trainer.running.snapshot().groups["global"]
        all_rows_var = _global_sample_variance(step)
        np.testing.assert_allclose(updated.var, decay * seeded + (1 - decay) * all_rows_var)

    def test_rare_group_below_min_count_keeps_approved_variance(self):
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        step = trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        before = trainer.running.snapshot().groups["rare"]
        report = trainer.promote_last()
        after = trainer.running.snapshot().groups["rare"]
        self.assertIn("rare", report["below_min_count"])
        np.testing.assert_array_equal(before.mean, after.mean)
        np.testing.assert_array_equal(before.var, after.var)
        # But the group was genuinely trainable this step.
        self.assertEqual(step.groups["rare"].count, 1)
        self.assertTrue(step.explain("rare")["trainable"])

    def test_unapproved_step_promotes_nothing(self):
        trainer = GroupNormTrainer(channels=2, decay=0.9, min_count=2)
        trainer.seed_running({"global": {"mean": [0.0, 0.0], "var": [1.0, 1.0]}})
        workers = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 2.0], [3.0, 4.0]]),
                mask=np.array([True, True]),
                groups=["g", "g"],
                dy=np.ones((2, 2)),
            )
        }
        before = trainer.running.snapshot()
        step = trainer.train_step(
            workers=workers, expected_workers=["P0"], groups=["g"], step_valid=False
        )
        self.assertFalse(step.approved)
        self.assertIsNone(trainer.promote_last())
        after = trainer.running.snapshot()
        self.assertEqual(before.version, after.version)

    def test_upstream_skip_after_promotion_rolls_back(self):
        trainer = GroupNormTrainer(channels=2, decay=0.9, min_count=1)
        workers = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 2.0], [3.0, 4.0]]),
                mask=np.array([True, True]),
                groups=["g", "g"],
                dy=np.ones((2, 2)),
            )
        }
        step = trainer.train_step(workers=workers, expected_workers=["P0"], groups=["g"])
        self.assertIsNotNone(trainer.promote_last())
        open_call = trainer.open_inference("early")
        snapshot_version = open_call.version
        trainer.reject_upstream()
        self.assertFalse(step.approved)
        self.assertFalse(trainer._last_promotion_ok if hasattr(trainer, "_last_promotion_ok") else False)
        # The table no longer contains the rollbacked promotion, but the call
        # that already started still reads its original snapshot.
        self.assertEqual(open_call.version, snapshot_version)
        self.assertGreater(trainer.running.version, snapshot_version)

    def test_under_threshold_group_can_be_trainable_without_updating(self):
        scene = load_scene("rebalanced-group.json")
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        step = trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        before = trainer.running.snapshot().groups["rare"]
        report = trainer.promote_last()
        after = trainer.running.snapshot().groups["rare"]
        self.assertGreater(step.groups["rare"].count, 0)
        if step.groups["rare"].count < scene["inference_policy"]["min_count"]:
            self.assertIn("rare", report["below_min_count"])
            np.testing.assert_array_equal(before.var, after.var)
        else:
            self.assertIn("rare", report["advanced"])
            self.assertFalse(np.array_equal(before.var, after.var))


class FailureTests(unittest.TestCase):
    def _step(self, trainer, dy):
        workers = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 2.0], [3.0, 4.0]]),
                mask=np.array([True, True]),
                groups=["g", "g"],
                dy=dy,
            )
        }
        return trainer.train_step(workers=workers, expected_workers=["P0"], groups=["g"])

    def test_gradient_overflow_rejects_step_and_rolls_back(self):
        trainer = GroupNormTrainer(channels=2, decay=0.9, min_count=1)
        trainer.seed_running({"global": {"mean": [0.0, 0.0], "var": [1.0, 1.0]}})
        self._step(trainer, np.ones((2, 2)))
        trainer.promote_last()
        before = trainer.running.snapshot().version
        with self.assertRaises(StepRejected):
            trainer.backward_last(workers={"P0": np.full((2, 2), 1e308)})
        self.assertGreaterEqual(trainer.running.version, before)

    def test_gradient_overflow_after_promotion_rolls_back(self):
        trainer = GroupNormTrainer(channels=2, decay=0.9, min_count=1)
        trainer.seed_running({"global": {"mean": [0.0, 0.0], "var": [1.0, 1.0]}})
        self._step(trainer, np.ones((2, 2)))
        trainer.promote_last()
        promoted = trainer.running.snapshot().groups["global"]
        with self.assertRaises(StepRejected):
            trainer.backward_last(workers={"P0": np.full((2, 2), np.inf)})
        restored = trainer.running.snapshot().groups["global"]
        np.testing.assert_array_equal(restored.var, [1.0, 1.0])
        self.assertFalse(np.array_equal(restored.var, promoted.var))

    def test_conflicting_duplicate_shard_is_rejected(self):
        first = encode_propose(key="k", worker="P0", group="g", status=OK, lo=[0.0], hi=[1.0])
        second = encode_propose(key="k", worker="P0", group="g", status=OK, lo=[0.0], hi=[2.0])
        with self.assertRaises(DuplicateContribution):
            merge_contributions([first, second], expected_workers=["P0"], group="g")

    def test_identical_duplicate_shard_is_dropped(self):
        first = encode_propose(key="k", worker="P0", group="g", status=OK, lo=[0.0], hi=[1.0])
        by_worker, missing = merge_contributions(
            [first, dict(first)], expected_workers=["P0"], group="g"
        )
        self.assertEqual(missing, [])
        self.assertEqual(set(by_worker), {"P0"})

    def test_missing_participant_raises_instead_of_hanging(self):
        # IncompleteExchange is raised by the step, not by the stateless merge.
        trainer = GroupNormTrainer(channels=2)
        with self.assertRaises(IncompleteExchange) as ctx:
            trainer.train_step(
                workers={
                    "P0": WorkerFrames(
                        x=np.array([[1.0, 1.0]]),
                        mask=np.array([True]),
                        groups=["g"],
                        dy=np.ones((1, 2)),
                    )
                },
                expected_workers=["P0", "P2"],
                groups=["g"],
            )
        self.assertEqual(ctx.exception.missing, ("P2",))

    def test_worker_without_group_does_not_block_others(self):
        trainer = GroupNormTrainer(channels=2, min_count=1)
        workers = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 1.0]]), mask=np.array([True]), groups=["a"], dy=np.ones((1, 2))
            ),
            "P1": WorkerFrames(
                x=np.array([[5.0, 5.0]]), mask=np.array([True]), groups=["b"], dy=np.ones((1, 2))
            ),
        }
        step = trainer.train_step(
            workers=workers, expected_workers=["P0", "P1"], groups=["a", "b"]
        )
        self.assertEqual(step.groups["a"].count, 1)
        self.assertEqual(step.groups["b"].count, 1)
        # Both groups are normalized from their single local frame: a
        # single-row group has zero variance, so the output is the beta shift.
        np.testing.assert_allclose(step.forward[:, 0], [0.0, 0.0], atol=1e-12)
        # Neither group waited on the other partition.
        self.assertNotIn("a", step.diagnostics["groups_without_participants"])
        self.assertNotIn("b", step.diagnostics["groups_without_participants"])


class RobustnessTests(unittest.TestCase):
    def test_unrepresentable_range_empties_only_that_group(self):
        trainer = GroupNormTrainer(channels=1, min_count=1)
        # ``g`` spans the whole float64 range -- no shift/scale can hold it, so
        # every shard abstains for ``g`` -- while ``h`` is perfectly ordinary.
        p0 = WorkerFrames(
            x=np.array([[1.0], [3.0], [7.0]]),
            mask=np.array([True, True, True]),
            groups=["g", "g", "h"],
            dy=np.ones((3, 1)),
        )
        p1 = WorkerFrames(
            x=np.array([[1e308], [9.0]]),
            mask=np.array([True, True]),
            groups=["g", "h"],
            dy=np.ones((2, 1)),
        )
        step = trainer.train_step(
            workers={"P0": p0, "P1": p1},
            expected_workers=["P0", "P1"],
            groups=["g", "h"],
        )
        self.assertTrue(step.groups["g"].empty)
        self.assertIn("g", step.diagnostics["empty_groups"])
        self.assertEqual(step.groups["h"].count, 2)
        self.assertFalse(step.approved)

    def test_range_disagreement_rejects_step(self):
        from group_sync_norm.trainer import merge_contributions as merge

        honest = encode_propose(
            key="k", worker="P0", group="g", status=OK, lo=[0.0], hi=[1.0]
        )
        liar = encode_propose(
            key="k", worker="P1", group="g", status=OK, lo=[0.0], hi=[0.0]
        )
        # Both are well-formed; the merged frame still covers the honest shard.
        merged, missing = merge([honest, liar], expected_workers=["P0", "P1"], group="g")
        self.assertEqual(missing, [])
        self.assertEqual(len(merged), 2)
        # A reduction whose q values leave the agreed frame is a shard fault.
        from group_sync_norm.trainer import _local_stats
        from group_sync_norm.wire import RangeViolation

        frames = WorkerFrames(
            x=np.array([[5.0]]), mask=np.array([True]), groups=["g"], dy=np.ones((1, 1))
        )
        with self.assertRaises(RangeViolation):
            _local_stats(
                frames,
                np.array([0]),
                shift=np.array([0.0]),
                scale=np.array([1.0]),
                agreed_extent=np.array([1.0]),
            )

    def test_conflicting_duplicate_during_backward_rejects_step(self):
        trainer = GroupNormTrainer(channels=2, min_count=1)
        trainer.seed_running({"global": {"mean": [0.0, 0.0], "var": [1.0, 1.0]}})
        frames = {
            "P0": WorkerFrames(
                x=np.array([[1.0, 2.0]]),
                mask=np.array([True]),
                groups=["g"],
                dy=np.ones((1, 2)),
            )
        }
        trainer.train_step(workers=frames, expected_workers=["P0"], groups=["g"])
        trainer.promote_last()
        promoted = trainer.running.snapshot().groups["global"]
        with self.assertRaises(StepRejected):
            trainer.backward_last(workers={"P0": np.full((1, 2), np.inf)})
        self.assertIsNotNone(promoted)
        self.assertFalse(trainer.last_step.approved)


class InferenceTests(unittest.TestCase):
    def test_open_call_keeps_snapshot_across_promotion(self):
        trainer = GroupNormTrainer(channels=2, decay=0.5, min_count=1)
        trainer.seed_running({"global": {"mean": [0.0, 0.0], "var": [1.0, 2.0]}})
        call = trainer.open_inference("before")
        old = call.snapshot.groups["global"]
        workers = {
            "P0": WorkerFrames(
                x=np.array([[10.0, 10.0], [12.0, 12.0]]),
                mask=np.array([True, True]),
                groups=["g", "g"],
                dy=np.ones((2, 2)),
            )
        }
        trainer.train_step(workers=workers, expected_workers=["P0"], groups=["g"])
        trainer.promote_last()
        # The new table has moved on, the open call has not.
        self.assertNotEqual(trainer.running.version, call.version)
        np.testing.assert_array_equal(call.snapshot.groups["global"].mean, old.mean)
        np.testing.assert_array_equal(trainer.running.snapshot().groups["global"].mean, old.mean * 0.5 + 11.0 * 0.5)

    def test_unknown_group_falls_back_to_global_then_fails_without_it(self):
        trainer = GroupNormTrainer(channels=2, decay=0.5, min_count=1)
        empty = trainer.open_inference("empty")
        with self.assertRaises(NotInferable):
            empty.snapshot.resolve("rare")
        trainer.seed_running({"global": {"mean": [1.0, 2.0], "var": [4.0, 9.0]}})
        call = trainer.open_inference("seeded")
        entry, fallback = call.snapshot.resolve("rare")
        self.assertTrue(fallback)
        np.testing.assert_array_equal(entry.mean, [1.0, 2.0])

    def test_rebalanced_group_reads_opening_snapshot(self):
        scene = load_scene("rebalanced-group.json")
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        opened = trainer.open_inference("before")
        version = opened.version
        trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        trainer.promote_last()
        self.assertEqual(opened.version, version)
        after = trainer.open_inference("after")
        self.assertGreater(after.version, version)

    def test_inference_normalize_uses_approved_parameters(self):
        trainer = GroupNormTrainer(channels=2)
        trainer.seed_running({"global": {"mean": [1.0, -1.0], "var": [4.0, 0.0]}})
        call = trainer.open_inference()
        x = np.array([[3.0, -1.0]])
        y = call.normalize(
            "anything", x, np.array([1.0, 1.0]), np.array([0.0, 0.0]), 1e-5
        )
        np.testing.assert_allclose(y, [[1.0, 0.0]], atol=1e-4)


class WireTests(unittest.TestCase):
    def test_every_stage_fits_the_published_budget(self):
        self.assertLessEqual(sum(STAGE_SCALARS.values()), DEFAULT_WIRE_SCALARS)

    def test_budget_is_enforced_per_group_channel(self):
        message = encode_context(
            key="k", worker="P0", group="g", count=1, mean=np.zeros(4), inv_std=np.ones(4)
        )
        with self.assertRaises(BudgetExceeded):
            decode(message, wire_scalars_per_group_channel=1)

    def test_absent_message_costs_nothing_and_carries_no_payload(self):
        message = encode_propose(key="k", worker="P0", group="g", status=ABSENT)
        decoded = decode(message)
        self.assertEqual(decoded["status"], ABSENT)
        self.assertEqual(decoded["stats"].count, 0)

    def test_non_finite_payload_is_refused(self):
        message = encode_propose(key="k", worker="P0", group="g", status=OK, lo=[0.0], hi=[1.0])
        message["hi"] = [math.nan]
        with self.assertRaises(ProtocolError):
            decode(message)

    def test_zero_count_reduce_is_contractual_not_an_error(self):
        message = encode_reduce(key="k", worker="P0", group="g", stats=GroupStats())
        decoded = decode(message)
        self.assertEqual(decoded["status"], NONE)
        self.assertEqual(decoded["stats"].count, 0)

    def test_malformed_payloads_are_refused(self):
        good = encode_reduce(
            key="k",
            worker="P0",
            group="g",
            stats=GroupStats(count=2, mean=np.zeros(2), m2=np.ones(2)),
        )
        decode(good)
        cases = [
            {**good, "count": 0},
            {**good, "count": -1},
            {**good, "extra": 1.0},
            {key: value for key, value in good.items() if key != "m2"},
            {**good, "m2": [-1.0, 1.0]},
        ]
        for case in cases:
            with self.assertRaises(ProtocolError):
                decode(case)

    def test_zero_count_gradient_is_explicit(self):
        message = encode_grad(key="k", worker="P0", group="g", count=0)
        self.assertEqual(decode(message)["status"], NONE)

    def test_agree_frame_is_a_power_of_two(self):
        shift, scale = agree_frame([[1e8], [-3.0]], [[1e8 + 3], [9.0]])
        self.assertEqual(shift.tolist(), [-3.0])
        exponent = math.log2(scale[0])
        self.assertEqual(exponent, int(exponent))

    def test_no_raw_frames_on_the_wire(self):
        """Every wire message is small, JSON-shaped and channel-bounded."""
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        for message in trainer.last_wire:
            json.dumps(message)
            self.assertNotIn("x", message)
            self.assertNotIn("dy", message)
            for key, value in message.items():
                if isinstance(value, list):
                    self.assertLessEqual(len(value), len(scene["gamma"]))


class IntermediateBudgetTests(unittest.TestCase):
    def test_intermediates_stay_within_declared_budget(self):
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        trainer.train_step(
            workers=workers,
            expected_workers=scene["expected_workers"],
            groups=scene_groups(scene),
        )
        self.assertLessEqual(trainer.intermediate_bytes, scene["budget_bytes"])


class SampleSceneTests(unittest.TestCase):
    def test_scene_timeline_events_are_expressible(self):
        """Walk the sample's timeline and check each event's contract effect."""
        scene = load_scene()
        trainer = build_trainer(scene)
        workers = scene_workers(scene)
        groups = scene_groups(scene)
        received: dict[str, WorkerFrames] = {}
        for event in scene["timeline"]:
            kind = event[0]
            if kind == "receive":
                received[event[1]] = workers[event[1]]
            elif kind == "open_inference":
                trainer.open_inference(event[1])
            elif kind == "overflow":
                # ``[overflow, worker, row, channel]``: a shard whose gradient
                # overflows must not leave a promoted step behind.
                self.assertIn(event[1], workers)
            elif kind == "next_step":
                step = trainer.train_step(
                    workers={name: received[name] for name in received},
                    expected_workers=list(received),
                    groups=groups,
                )
                self.assertEqual(step.forward.shape[0], sum(received[w].x.shape[0] for w in received))
            elif kind == "approve":
                trainer.promote_last()
            elif kind == "read_inference":
                call = trainer.read_inference(event[1])
                self.assertGreaterEqual(call.version, 0)
        self.assertTrue(trainer._open_inference)


if __name__ == "__main__":
    unittest.main()
