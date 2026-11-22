"""Partitioned training/inference coordination for a grouped BatchNorm layer.

This module is the only place that knows how per-worker reductions are
combined.  It never sees another worker's raw frames: arrays stay local and
only :mod:`~group_sync_norm.wire` messages cross the partition boundary.

A step is a four-phase exchange per group
=========================================

1. ``propose`` -- each participating worker publishes the extremes it holds for
   each group.  :func:`~group_sync_norm.wire.agree_frame` turns them into one
   shared ``shift``/``scale`` (``shift`` = smallest value, ``scale`` = a power
   of two covering the extent).  Division by a power of two is exact, so values
   near ``1e8`` do not wipe out the variance of samples that differ by one.
2. ``reduce``  -- each worker reduces its *own* valid rows in that frame and
   publishes ``(count, mean, m2)``.  The coordinator merges with Chan's
   parallel formula.  The result is mathematically identical to a single-pass
   reduction of all rows, so it cannot depend on how rows were split.
3. ``context`` -- the coordinator broadcasts the merged ``(mean_q, inv_std_q)``
   so every worker computes with the *same* statistics.  A row's forward value
   depends only on that row and those statistics.
4. ``grad``    -- each worker publishes ``(sum_g, sum_gq)``; the coordinator
   merges them into ``dgamma``, ``dbeta`` and the BN chain-rule term.  The same
   merged statistics feed the broadcast, so the gradients correspond to the
   forward values that were actually produced.

Absence and emptiness
=====================
A worker with no frames for a group sends ``propose(none)`` / ``reduce(none)``;
a worker that is not a participant sends ``propose(absent)``.  Neither blocks
the others.  A participant that simply has not delivered is reported by
:class:`~group_sync_norm.wire.IncompleteExchange` (carrying the names and
groups) instead of being waited on forever.

Failure consistency
===================
Gradient overflow, an upstream skip, a conflicting duplicate, or non-finite
valid rows mark the step rejected.  A rejected step promotes nothing; if the
step had already been promoted, :meth:`GroupNormTrainer.reject_pending` rolls
the table back.  Already-open inference calls are unaffected either way
because they hold immutable snapshots.

Inference
=========
Each group resolves to its own approved entry when it has one, otherwise to
the approved ``global`` entry.  The table only advances on approved steps:
eligible groups use their sample-variance candidate with inertia ``decay``;
ineligible groups keep their previous approved value untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .state import GLOBAL_GROUP, InferenceCall, RunningStats
from .wire import (
    ABSENT,
    BudgetExceeded,
    CONTEXT,
    DEFAULT_WIRE_SCALARS,
    GRAD,
    NONE,
    OK,
    PROPOSE,
    REDUCE,
    SKIPPED,
    GroupStats,
    IncompleteExchange,
    ProtocolError,
    RANGE_TOLERANCE_ULPS,
    RangeViolation,
    agree_frame,
    decode,
    encode_context,
    encode_grad,
    encode_propose,
    encode_reduce,
    merge_many,
    message_canonical,
    restore_moments,
)


class DuplicateContribution(ProtocolError):
    """The same participant sent two different messages for the same slot."""


class StepRejected(RuntimeError):
    """A step was rejected after its statistics may have been promoted."""


@dataclass
class WorkerFrames:
    """Rows one partition holds for one step.

    ``x`` is ``[rows, channels]``, ``mask`` is one boolean per row, ``groups``
    names each row's calibration group, and ``dy`` is the upstream gradient
    with respect to the normalized output (same shape as ``x``).
    """

    x: np.ndarray
    mask: np.ndarray
    groups: Sequence[str]
    dy: np.ndarray

    def validate(self, channels: int | None = None) -> None:
        x = np.asarray(self.x)
        dy = np.asarray(self.dy)
        mask = np.asarray(self.mask, dtype=bool)
        if x.size == 0 and dy.size == 0 and channels is not None:
            # A partition with no rows may pass ``[]``; normalise it to
            # ``(0, channels)`` so downstream shapes are uniform.
            self.x = x.reshape(0, channels)
            self.dy = dy.reshape(0, channels)
            self.mask = mask.reshape(0)
            x, dy, mask = self.x, self.dy, self.mask
        if x.ndim != 2 or dy.shape != x.shape:
            raise ValueError("x and dy must share a 2-D shape")
        if channels is not None and x.shape[1] != channels:
            raise ValueError(f"expected {channels} channels, got {x.shape[1]}")
        if mask.shape != (x.shape[0],) or len(self.groups) != x.shape[0]:
            raise ValueError("mask and groups must have one entry per row")
        for group in self.groups:
            if not isinstance(group, str) or not group:
                raise ValueError("group names must be non-empty strings")


@dataclass
class LocalReduce:
    """One worker's local view of one group (no copies of the caller's rows)."""

    name: str
    status: str
    count: int
    nonfinite: int
    rows: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    reason: str | None = None


def reduce_local(frames: WorkerFrames, *, channels: int) -> dict[str, LocalReduce]:
    """Reduce one partition into per-group extremes and finite-row indices.

    Only unmasked rows count.  Unmasked rows holding NaN/Inf are recorded in
    ``nonfinite`` and excluded from the moments; the step then becomes
    ineligible for promotion, but every group still merges normally.
    """
    frames.validate(channels)
    x = np.asarray(frames.x, dtype=np.float64)
    mask = np.asarray(frames.mask, dtype=bool)
    groups = list(frames.groups)
    out: dict[str, LocalReduce] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for name in sorted(set(groups)):
            selected = np.array([g == name for g in groups], dtype=bool) & mask
            raw = x[selected]
            finite = np.all(np.isfinite(raw), axis=1)
            nonfinite = int((~finite).sum())
            index = np.flatnonzero(selected)[finite]
            if index.size == 0:
                out[name] = LocalReduce(
                    name=name,
                    status=NONE,
                    count=0,
                    nonfinite=nonfinite,
                    rows=index,
                    reason=None if nonfinite == 0 else "all valid rows are non-finite",
                )
                continue
            good = raw[finite]
            out[name] = LocalReduce(
                name=name,
                status=OK,
                count=int(index.size),
                nonfinite=nonfinite,
                rows=index,
                lo=good.min(axis=0),
                hi=good.max(axis=0),
            )
    return out


def merge_contributions(
    messages,
    *,
    expected_workers: Sequence[str],
    group: str | None = None,
    wire_scalars_per_group_channel: int = DEFAULT_WIRE_SCALARS,
) -> tuple[dict[str, Mapping], list[str]]:
    """Validate, deduplicate, and completeness-check a set of contributions.

    Returns ``(by_worker, missing_workers)``.  An identical duplicate is dropped
    (re-delivery); a conflicting duplicate raises
    :class:`DuplicateContribution`.  Missing workers are reported, never waited
    on here.
    """
    by_worker: dict[str, Mapping] = {}
    canonical: dict[str, tuple] = {}
    expected = set(expected_workers)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_workers must be unique")
    for raw in messages:
        decoded = decode(raw, wire_scalars_per_group_channel=wire_scalars_per_group_channel)
        if group is not None and decoded["group"] != group:
            raise ProtocolError("message routed to the wrong group")
        worker = decoded["worker"]
        if worker not in expected:
            raise ProtocolError(f"message from unknown worker {worker!r}")
        if worker in by_worker:
            if canonical[worker] != message_canonical(raw):
                raise DuplicateContribution(
                    f"conflicting duplicate from {worker!r} for group {decoded['group']!r}"
                )
            continue
        by_worker[worker] = decoded
        canonical[worker] = message_canonical(raw)
    return by_worker, sorted(expected - set(by_worker))


def merge_group_stats(
    contributions: Mapping[str, Mapping],
    *,
    shift: np.ndarray,
    scale: np.ndarray,
) -> GroupStats:
    """Chan-merge the ``reduce`` stage of one group in the agreed frame."""
    parts: list[GroupStats] = []
    for worker in sorted(contributions):
        message = contributions[worker]
        if message["status"] != OK:
            continue
        stats = message.get("stats")
        if stats is None or stats.empty:
            continue
        parts.append(
            GroupStats(
                count=stats.count,
                nonfinite=stats.nonfinite,
                shift=np.asarray(shift, dtype=np.float64),
                scale=np.asarray(scale, dtype=np.float64),
                mean=np.asarray(stats.mean, dtype=np.float64),
                m2=np.asarray(stats.m2, dtype=np.float64),
            )
        )
    return merge_many(parts) if parts else GroupStats()


@dataclass
class StepResult:
    """Result of one synchronized training forward."""

    forward: np.ndarray
    groups: dict[str, GroupStats]
    restored: dict[str, tuple]
    diagnostics: dict[str, object]
    version_before: int
    approved: bool = False

    def explain(self, group: str) -> dict[str, object]:
        stats = self.groups.get(group)
        if stats is None:
            return {"group": group, "observed": False}
        mean, m2, var = self.restored[group]
        return {
            "group": group,
            "observed": True,
            "count": stats.count,
            "nonfinite": stats.nonfinite,
            "trainable": stats.count > 0,
            "empty": stats.empty,
            "forward_mean": None if mean is None else mean.tolist(),
            "forward_var": None if var is None else var.tolist(),
            "approved_step": self.approved,
        }


class GroupNormTrainer:
    """Coordinator for partitioned grouped BatchNorm.

    Rows are concatenated in ``expected_workers`` order; each worker block keeps
    its local row order.
    """

    def __init__(
        self,
        *,
        channels: int,
        epsilon: float = 1e-5,
        decay: float = 0.85,
        min_count: int = 3,
        wire_scalars_per_group_channel: int = DEFAULT_WIRE_SCALARS,
        budget_bytes: int | None = None,
    ) -> None:
        self.channels = int(channels)
        self.epsilon = float(epsilon)
        self.wire_budget = int(wire_scalars_per_group_channel)
        self.running = RunningStats(decay=decay, min_count=min_count)
        self.budget_bytes = budget_bytes
        self.last_step: StepResult | None = None
        self.last_grads: dict[str, np.ndarray] | None = None
        self.last_wire: list[dict] = []
        self.last_promotion: dict | None = None
        self.inference_log: list[dict] = []
        self.intermediate_bytes = 0
        self._step_index = 0
        self._open_inference: dict[str, InferenceCall] = {}
        self._input: dict[str, WorkerFrames] | None = None
        self._shapes: dict[str, tuple] | None = None
        self._masks: dict[str, np.ndarray] | None = None
        self._local: dict[str, dict[str, LocalReduce]] | None = None
        self._frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._context: dict[str, tuple] | None = None
        self._stats: dict[str, GroupStats] | None = None
        self._expected: list[str] | None = None
        self._index = 0
        self._approved = True
        self._pending: dict | None = None

    # -- approved table ---------------------------------------------------

    def seed_running(self, table: Mapping[str, Mapping]) -> None:
        """Seed approved entries from ``{"mean": [...], "var": [...]}`` records."""
        for name, entry in table.items():
            self.running.prime(name, entry["mean"], entry["var"])

    def open_inference(self, name: str = "default") -> InferenceCall:
        """Start a read against the currently approved table."""
        call = InferenceCall(snapshot=self.running.snapshot())
        self._open_inference[name] = call
        return call

    def read_inference(self, name: str = "default") -> InferenceCall:
        try:
            return self._open_inference[name]
        except KeyError as exc:
            raise KeyError(f"no inference call named {name!r}") from exc

    def release_inference(self, name: str = "default") -> None:
        self._open_inference.pop(name, None)

    def explain_inference(self, group: str) -> dict[str, object]:
        """Why ``group`` does or does not have its own inference parameters."""
        snapshot = self.running.snapshot()
        entry = snapshot.groups.get(group)
        fallback = snapshot.fallback is not None
        trainable = None
        if self.last_step is not None and group in self.last_step.groups:
            trainable = self.last_step.groups[group].count > 0
        if entry is not None:
            reason = "approved"
        elif fallback:
            reason = "fallback_global"
        else:
            reason = "not_inferable"
        latest = next(
            (item for item in reversed(self.inference_log) if group in item.get("seen", ())),
            None,
        )
        return {
            "group": group,
            "reason": reason,
            "has_own_entry": entry is not None,
            "global_available": fallback,
            "version": snapshot.version,
            "last_trainable": trainable,
            "last_step": latest,
        }

    # -- forward ----------------------------------------------------------

    def train_step(
        self,
        *,
        workers: Mapping[str, WorkerFrames],
        expected_workers: Sequence[str],
        groups: Sequence[str],
        step_index: int | None = None,
        step_valid: bool = True,
        upstream_skipped: Sequence[str] = (),
        gamma=None,
        beta=None,
        promote: bool = False,
    ) -> StepResult:
        """Run the synchronized forward for one training step.

        ``step_index`` defaults to an internally increasing counter.  Reusing an
        index is allowed (a replay), but the wire keys are then identical, which
        is exactly what makes duplicate detection observable.
        """
        index = self._step_index if step_index is None else int(step_index)
        _validate_workers(workers, expected_workers, self.channels)
        group_list = list(dict.fromkeys(groups))
        if not group_list:
            raise ValueError("at least one group is required")
        if GLOBAL_GROUP in group_list:
            raise ValueError(f"{GLOBAL_GROUP!r} is reserved for the global fallback")
        wire: list[dict] = []

        local = _MissingAsNone(
            {
                name: reduce_local(workers[name], channels=self.channels)
                for name in expected_workers
                if name in workers
            }
        )

        # -- propose -------------------------------------------------------
        range_violations: list[str] = []
        frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for group in group_list:
            messages = []
            for worker in expected_workers:
                if worker not in workers:
                    # No message at all: the merge reports it as incomplete,
                    # never as a zero-frame declaration.
                    continue
                entry = local[worker].get(group)
                if entry is None:
                    messages.append(
                        encode_propose(
                            key=_key(index, PROPOSE),
                            worker=worker,
                            group=group,
                            status=ABSENT,
                            reason="not a participant",
                        )
                    )
                elif entry.status != OK:
                    messages.append(
                        encode_propose(
                            key=_key(index, PROPOSE),
                            worker=worker,
                            group=group,
                            status=NONE,
                            reason=entry.reason,
                        )
                    )
                else:
                    messages.append(
                        encode_propose(
                            key=_key(index, PROPOSE),
                            worker=worker,
                            group=group,
                            status=OK,
                            lo=entry.lo,
                            hi=entry.hi,
                        )
                    )
            by_worker, missing = merge_contributions(
                messages,
                expected_workers=expected_workers,
                group=group,
                wire_scalars_per_group_channel=self.wire_budget,
            )
            if missing:
                raise IncompleteExchange(missing, groups=[group])
            wire.extend(messages)
            los = [m["lo"] for m in by_worker.values() if m["status"] == OK]
            his = [m["hi"] for m in by_worker.values() if m["status"] == OK]
            try:
                frames[group] = (
                    (np.zeros(self.channels), np.ones(self.channels))
                    if not los
                    else agree_frame(los, his)
                )
            except OverflowError:
                # The participants' joint extent does not fit in float64.  No
                # frame can represent it, so every shard abstains and the group
                # stays empty for this step; the step is not promoted.
                range_violations.append(f"{group}: agreed extent not representable")
                frames[group] = (np.zeros(self.channels), np.ones(self.channels))

        # -- reduce --------------------------------------------------------
        nonfinite_by_group: dict[str, int] = {}
        overflowed: set[tuple[str, str]] = set()
        merged: dict[str, GroupStats] = {}
        restored: dict[str, tuple] = {}
        for group in group_list:
            shift, scale = frames[group]
            messages = []
            total_nonfinite = 0
            for worker in expected_workers:
                if worker not in workers:
                    continue
                entry = local[worker].get(group)
                if entry is None:
                    messages.append(
                        encode_reduce(
                            key=_key(index, REDUCE),
                            worker=worker,
                            group=group,
                            stats=GroupStats(),
                            status=ABSENT,
                            reason="not a participant",
                        )
                    )
                    continue
                total_nonfinite += entry.nonfinite
                if entry.status != OK:
                    messages.append(
                        encode_reduce(
                            key=_key(index, REDUCE),
                            worker=worker,
                            group=group,
                            stats=GroupStats(),
                            reason=entry.reason,
                        )
                    )
                    continue
                try:
                    stats = _local_stats(
                        workers[worker],
                        entry.rows,
                        shift,
                        scale,
                        declared=(entry.lo, entry.hi),
                        agreed_extent=scale,
                    )
                except RangeViolation as exc:
                    # The shard's reduction disagrees with the frame every
                    # participant announced.  It is a shard fault: this worker
                    # abstains (skipped), and the step cannot be promoted even
                    # if the other shards merge cleanly.
                    range_violations.append(f"{group}:{worker}: {exc.detail}")
                    messages.append(
                        encode_reduce(
                            key=_key(index, REDUCE),
                            worker=worker,
                            group=group,
                            stats=GroupStats(),
                            status=SKIPPED,
                            reason=f"range violation: {exc.detail}",
                        )
                    )
                    continue
                if stats is None:
                    overflowed.add((group, worker))
                    messages.append(
                        encode_reduce(
                            key=_key(index, REDUCE),
                            worker=worker,
                            group=group,
                            stats=GroupStats(),
                            status=SKIPPED,
                            reason="local reduction overflow",
                        )
                    )
                    continue
                messages.append(
                    encode_reduce(
                        key=_key(index, REDUCE),
                        worker=worker,
                        group=group,
                        stats=stats,
                    )
                )
            by_worker, missing = merge_contributions(
                messages,
                expected_workers=expected_workers,
                group=group,
                wire_scalars_per_group_channel=self.wire_budget,
            )
            if missing:
                raise IncompleteExchange(missing, groups=[group])
            wire.extend(messages)
            nonfinite_by_group[group] = total_nonfinite
            stats = merge_group_stats(by_worker, shift=shift, scale=scale)
            merged[group] = stats
            restored[group] = (
                (None, None, None)
                if stats.empty
                else restore_moments(stats, shift=shift, scale=scale)
            )

        # -- context broadcast + per-worker forward ------------------------
        gamma_arr = np.ones(self.channels) if gamma is None else np.asarray(gamma, dtype=np.float64)
        beta_arr = np.zeros(self.channels) if beta is None else np.asarray(beta, dtype=np.float64)
        context: dict[str, tuple] = {}
        for group in group_list:
            stats = merged[group]
            if stats.empty:
                context[group] = (None, None, 0)
                continue
            shift, scale = frames[group]
            # The context lives in the q frame: ``(q - mean_q) * inv_std_q``
            # equals ``(x - mean) / sqrt(var + eps)`` because scale cancels.
            inv_std_q = scale / np.sqrt(restored[group][2] + self.epsilon)
            message = encode_context(
                key=_key(index, CONTEXT),
                worker=expected_workers[0],
                group=group,
                count=stats.count,
                mean=np.asarray(stats.mean),
                inv_std=inv_std_q,
            )
            decoded = decode(message, wire_scalars_per_group_channel=self.wire_budget)
            wire.append(message)
            context[group] = (
                np.asarray(decoded["mean"], dtype=np.float64),
                np.asarray(decoded["inv_std"], dtype=np.float64),
                int(decoded["count"]),
            )

        blocks: list[np.ndarray] = []
        for worker in expected_workers:
            if worker not in workers:
                continue
            frames_w = workers[worker]
            x = np.asarray(frames_w.x, dtype=np.float64)
            mask = np.asarray(frames_w.mask, dtype=bool)
            out = np.zeros_like(x)
            for group in group_list:
                mean_q, inv_std_q, _ = context[group]
                if mean_q is None:
                    continue
                selected = np.array([g == group for g in frames_w.groups], dtype=bool) & mask
                if not np.any(selected):
                    continue
                raw = x[selected]
                with np.errstate(over="ignore", invalid="ignore"):
                    finite = np.all(np.isfinite(raw), axis=1)
                    shift, scale = frames[group]
                    q = (np.where(finite[:, None], raw, 0.0) - shift) / scale
                    qn = (q - mean_q) * inv_std_q
                    out[selected] = np.where(finite[:, None], qn * gamma_arr + beta_arr, 0.0)
            blocks.append(out)
        forward = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, self.channels))

        # -- budget check for kept intermediates ---------------------------
        held = _bytes_of(context) + _bytes_of(frames) + _bytes_of(
            {name: entry.rows for name, entries in local.items() for entry in entries.values()}
        )
        if self.budget_bytes is not None and held > self.budget_bytes:
            raise BudgetExceeded(held, self.budget_bytes, "forward intermediates")

        # -- commit per-step state ----------------------------------------
        self._input = dict(workers)
        self._shapes = {
            name: np.asarray(workers[name].x).shape
            for name in expected_workers
            if name in workers
        }
        self._masks = {
            name: np.asarray(workers[name].mask, dtype=bool).copy()
            for name in expected_workers
            if name in workers
        }
        self._local = local
        self._frames = frames
        self._context = context
        self._stats = merged
        self._expected = list(expected_workers)
        self._index = index
        self._approved = (
            bool(step_valid)
            and not tuple(upstream_skipped)
            and sum(nonfinite_by_group.values()) == 0
            and not overflowed
            and not range_violations
        )
        self._pending = None
        self.intermediate_bytes = int(held)
        self.last_wire = wire

        participant_counts = {
            group: sum(
                1
                for w in expected_workers
                if w in workers and local[w].get(group) is not None
            )
            for group in group_list
        }
        self.last_step = StepResult(
            forward=forward,
            groups=merged,
            restored=restored,
            diagnostics={
                # Rows excluded by the mask (padding), never counted.
                "masked_rows": int(
                    sum(
                        int((~np.asarray(workers[w].mask, dtype=bool)).sum())
                        for w in expected_workers
                        if w in workers
                    )
                ),
                # Unmasked rows dropped because they hold NaN/Inf: a data fault.
                "nonfinite_rows": int(sum(nonfinite_by_group.values())),
                "nonfinite_by_group": dict(nonfinite_by_group),
                # Participants exist but no valid frame: contractually empty.
                "empty_groups": [name for name, stats in merged.items() if stats.empty],
                "groups_without_participants": [
                    name for name, count in participant_counts.items() if count == 0
                ],
                "reduce_overflow": sorted(overflowed),
                "range_violations": list(range_violations),
                "upstream_skipped": sorted(upstream_skipped),
                "step_valid": bool(step_valid),
                "frames": {name: (f[0].copy(), f[1].copy()) for name, f in frames.items()},
            },
            version_before=self.running.version,
            approved=self._approved,
        )
        self.last_grads = None
        self.last_promotion = None
        self._step_index = index + 1
        if promote:
            self.promote_last()
        return self.last_step

    # -- backward ---------------------------------------------------------

    def backward_last(
        self,
        *,
        workers: Mapping[str, np.ndarray],
        expected_workers: Sequence[str] | None = None,
        gamma=None,
        allow_partial: bool = False,
    ) -> dict[str, np.ndarray]:
        """Synchronized backward for the most recent forward.

        ``workers`` maps worker names to their ``dy``.  Workers with rows for
        the step but no valid rows in a group send an explicit zero-count
        contribution; that is normal.  Omitting a worker entirely is a missing
        message and raises :class:`IncompleteExchange` unless
        ``allow_partial=True``.

        Raises :class:`StepRejected` if any contribution is non-finite; in that
        case no gradients are returned and the step's promotion (if any) is
        rolled back so updates and statistics agree.
        """
        if self.last_step is None or self._stats is None:
            raise RuntimeError("no training step has run")
        expected = list(self._expected if expected_workers is None else expected_workers)
        if set(expected) != set(self._expected):
            raise ValueError("backward must cover the forward's participants")
        for name in workers:
            if name not in self._shapes:
                raise ValueError(f"unknown worker {name!r}")
            dy = np.asarray(workers[name], dtype=np.float64)
            if dy.shape != self._shapes[name]:
                raise ValueError(f"dy for {name!r} does not match the forward")
        missing = sorted(set(expected) - set(workers))
        if missing and not allow_partial:
            raise IncompleteExchange(missing)

        gamma_arr = np.ones(self.channels) if gamma is None else np.asarray(gamma, dtype=np.float64)
        dgamma = np.zeros(self.channels, dtype=np.float64)
        dbeta = np.zeros(self.channels, dtype=np.float64)
        dx_blocks = {
            name: np.zeros(self._shapes[name], dtype=np.float64) for name in expected
        }
        bad: list[str] = []
        # Rows that are unmasked but non-finite have no forward value and
        # therefore no gradient.  Masked rows are excluded by ``mask``.
        finite_rows = {
            name: (
                np.all(np.isfinite(np.asarray(self._input[name].x, dtype=np.float64)), axis=1)
                if name in self._input
                else np.zeros(self._shapes[name][0], dtype=bool)
            )
            for name in expected
        }

        for group in self._stats:
            stats = self._stats[group]
            mean_q, inv_std_q, count = self._context[group]
            if stats.empty or mean_q is None:
                continue
            shift, scale = self._frames[group]
            messages = []
            usable: dict[str, np.ndarray] = {}
            for worker in expected:
                entry = self._local[worker].get(group)
                if entry is None or entry.status != OK or worker not in workers:
                    messages.append(
                        encode_grad(
                            key=_key(self._index, GRAD),
                            worker=worker,
                            group=group,
                            count=0,
                        )
                    )
                    continue
                x_rows = np.asarray(self._input[worker].x, dtype=np.float64)[entry.rows]
                dy_rows = np.asarray(workers[worker], dtype=np.float64)[entry.rows]
                with np.errstate(over="ignore", invalid="ignore"):
                    q = (x_rows - shift) / scale
                    qn = (q - mean_q) * inv_std_q
                    sum_g = dy_rows.sum(axis=0)
                    sum_gq = (dy_rows * qn).sum(axis=0)
                if not (np.all(np.isfinite(sum_g)) and np.all(np.isfinite(sum_gq))):
                    bad.append(f"{group}:{worker}")
                    messages.append(
                        encode_grad(
                            key=_key(self._index, GRAD),
                            worker=worker,
                            group=group,
                            count=0,
                        )
                    )
                    continue
                messages.append(
                    encode_grad(
                        key=_key(self._index, GRAD),
                        worker=worker,
                        group=group,
                        count=int(entry.count),
                        sum_g=sum_g,
                        sum_gq=sum_gq,
                    )
                )
                usable[worker] = entry.rows
            try:
                by_worker, missing_msgs = merge_contributions(
                    messages,
                    expected_workers=expected,
                    group=group,
                    wire_scalars_per_group_channel=self.wire_budget,
                )
            except DuplicateContribution:
                # A conflicting duplicate shard makes this step's update and
                # statistics equally untrustworthy: drop both.
                self._reject_current_step()
                raise
            if missing_msgs:
                raise IncompleteExchange(missing_msgs, groups=[group])
            self.last_wire.extend(messages)

            total_g = np.zeros(self.channels, dtype=np.float64)
            total_gq = np.zeros(self.channels, dtype=np.float64)
            for message in by_worker.values():
                if message["status"] != OK:
                    continue
                total_g += message["sum_g"]
                total_gq += message["sum_gq"]
            dgamma += total_gq
            dbeta += total_g

            n = np.float64(stats.count)
            for worker, rows in usable.items():
                x_rows = np.asarray(self._input[worker].x, dtype=np.float64)[rows]
                q = (x_rows - shift) / scale
                qn = (q - mean_q) * inv_std_q
                g = np.asarray(workers[worker], dtype=np.float64)[rows] * gamma_arr
                # d/dx in q frame: inv_std_q / scale = inv_std in x units.
                dx_blocks[worker][rows] = (inv_std_q / scale) * (
                    g - gamma_arr * total_g / n - gamma_arr * qn * total_gq / n
                )

        for worker in expected:
            keep = self._masks[worker] & finite_rows[worker]
            dx_blocks[worker] = np.where(keep[:, None], dx_blocks[worker], 0.0)

        if bad:
            self._reject_current_step()
            raise StepRejected(
                f"gradient overflow in {bad}; the step's update and statistics are discarded"
            )

        dx = np.concatenate([dx_blocks[name] for name in expected], axis=0)
        result = {"dx": dx, "dgamma": dgamma, "dbeta": dbeta}
        self.last_grads = result
        return result

    # -- promotion --------------------------------------------------------

    def promote_last(self) -> dict | None:
        """Commit the most recent step's statistics if it is approvable.

        Ineligible groups (``count < max(min_count, 2)``) keep their previously
        approved entry, byte for byte.  A named group's first candidate is
        stored as-is; later ones use the policy inertia.  ``global`` aggregates
        every valid frame of the step under the same rule, so a group without
        its own approved entry still has a defined fallback.

        Returns ``None`` when the step is rejected.  When no entry can advance
        (for example every group was a single frame) a report is still returned
        with an empty ``advanced`` list, so the caller can distinguish
        "rejected" from "accepted but nothing eligible".
        """
        if self.last_step is None or self._stats is None:
            raise RuntimeError("no training step has run")
        if not self._approved:
            return None
        payload: dict[str, tuple] = {}
        for name, stats in self._stats.items():
            if stats.empty:
                continue
            mean, m2, _ = self.last_step.restored[name]
            sample_var = np.zeros_like(mean) if stats.count < 2 else m2 / np.float64(stats.count - 1)
            payload[name] = (mean, sample_var, stats.count)
        # ``global`` is the same reduction over every valid frame.  It uses the
        # same threshold and inertia, so a step that cannot meet ``min_count``
        # overall leaves the approved fallback untouched.
        global_entry = _aggregate_global(self.last_step.restored, self._stats, self.channels)
        if global_entry is not None:
            payload[GLOBAL_GROUP] = global_entry

        before = self.running.export()
        report = self.running.apply(approved=True, groups=payload)
        seen = sorted(self._stats)
        if report is not None:
            report["seen"] = seen
            report["step_index"] = self._index
            self.inference_log.append(report)
            if report["advanced"]:
                self._pending = {
                    "step_index": self._index,
                    "before": before,
                    "version_after": self.running.version,
                }
        self.last_promotion = report
        return report

    def _reject_current_step(self) -> None:
        """Mark the current step rejected and roll back its promotion."""
        self._approved = False
        if self.last_step is not None:
            self.last_step.approved = False
        if self._pending is not None and self._pending["step_index"] == self._index:
            self.reject_pending()

    def reject_upstream(self) -> None:
        """Record an upstream decision to skip the step after the fact.

        Used when the optimizer announces a skip *after* the forward (for
        example a gradient-norm check).  The step's statistics are dropped and
        any promotion made for it is rolled back.  This is the only way an
        otherwise valid forward ends up not promoting: the caller decided.
        """
        self._reject_current_step()

    def reject_pending(self) -> None:
        """Roll back the current step's promotion, if one happened.

        Idempotent: once a promotion has been rolled back its snapshot cannot be
        restored twice, so repeated rejection reports cannot churn the table.
        """
        if self._pending is None:
            return
        self.running.restore(self._pending["before"])
        self._pending = None
        self._approved = False
        if self.last_step is not None:
            self.last_step.approved = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _MissingAsNone(dict):
    """Lookup map whose absent key yields ``None`` instead of raising."""

    def __missing__(self, name):
        return None


def _key(index: int, stage: str) -> str:
    return f"step{index}:{stage}"


def _validate_workers(workers: Mapping[str, WorkerFrames], expected: Sequence[str], channels: int):
    """Validate the frames that *are* present.

    A participant absent from ``workers`` has not delivered yet.  That is an
    incomplete exchange (reported with its name), not a caller error and never
    an unbounded wait.  Unknown names are a genuine caller error.
    """
    if len(set(expected)) != len(expected):
        raise ValueError("expected_workers must be unique")
    extra = sorted(set(workers) - set(expected))
    if extra:
        raise ValueError(f"workers not listed in expected_workers: {extra}")
    for name in expected:
        if name in workers:
            workers[name].validate(channels)


def _local_stats(
    frames: WorkerFrames,
    rows: np.ndarray,
    shift: np.ndarray,
    scale: np.ndarray,
    declared: tuple[np.ndarray, np.ndarray] | None = None,
    agreed_extent: np.ndarray | None = None,
) -> GroupStats | None:
    """Reduce one worker's finite rows in the agreed frame.

    Returns ``None`` when the local reduction would leave float64; the worker
    then reports ``skipped`` instead of putting a non-finite number on the wire.
    Raises :class:`RangeViolation` if the rows would not have produced the
    extremes this worker announced (a shard whose contents changed under the
    step, or a peer that lied about its range).
    """
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(frames.x, dtype=np.float64)[rows]
        if declared is not None:
            lo, hi = declared
            observed_lo = values.min(axis=0)
            observed_hi = values.max(axis=0)
            tol_lo = RANGE_TOLERANCE_ULPS * np.abs(np.spacing(lo))
            tol_hi = RANGE_TOLERANCE_ULPS * np.abs(np.spacing(hi))
            if np.any(observed_lo < lo - tol_lo) or np.any(observed_hi > hi + tol_hi):
                raise RangeViolation(
                    "shard",
                    "reduced values fall outside the declared frame",
                )
        q = (values - shift) / scale
        if not np.all(np.isfinite(q)):
            return None
        if agreed_extent is not None:
            # Every honest proposal lies inside the agreed extent, so every
            # reduced value must land in ``[0, 1]`` up to the range tolerance.
            tol = RANGE_TOLERANCE_ULPS * np.abs(np.spacing(agreed_extent))
            if np.any(q < -(tol / agreed_extent)) or np.any(q > 1.0 + tol / agreed_extent):
                raise RangeViolation(
                    "shard", "reduced values fall outside the agreed frame"
                )
        mean = q.mean(axis=0)
        centered = q - mean
        m2 = np.einsum("ij,ij->j", centered, centered)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(m2))):
        return None
    try:
        return GroupStats(
            count=int(q.shape[0]),
            nonfinite=0,
            shift=shift,
            scale=scale,
            mean=mean,
            m2=m2,
        )
    except ProtocolError:
        return None


def _aggregate_global(restored: Mapping[str, tuple], stats: Mapping[str, GroupStats], channels: int):
    """Chan-merge all groups' restored moments into the ``global`` candidate."""
    total = 0
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    for name, (group_mean, group_m2, _) in restored.items():
        count = stats[name].count
        if count == 0 or group_mean is None:
            continue
        if total == 0:
            mean = np.asarray(group_mean, dtype=np.float64).copy()
            m2 = np.asarray(group_m2, dtype=np.float64).copy()
            total = count
            continue
        n_a = np.float64(total)
        n_b = np.float64(count)
        delta = np.asarray(group_mean, dtype=np.float64) - mean
        mean = mean + delta * (n_b / (n_a + n_b))
        m2 = m2 + np.asarray(group_m2, dtype=np.float64) + delta * delta * (n_a * n_b / (n_a + n_b))
        total += count
    if total == 0:
        return None
    sample_var = m2 / np.float64(total - 1) if total >= 2 else np.zeros_like(mean)
    return (mean, sample_var, total)


def _bytes_of(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_bytes_of(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_bytes_of(v) for v in value)
    return 0
