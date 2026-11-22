"""Replay one training step under different partitionings and inspect it.

The tool answers the two questions a trainer asks about this operator:

1. Do different shardings of the same step produce the same forward values and
   the same gradients?
2. Why is a group trainable yet its inference value did not update?

Usage::

    python3 replay_step.py samples/scene.json
    python3 replay_step.py samples/rebalanced-group.json --splits 3,4
    python3 replay_step.py samples/scene.json --group rare --json

The input file is treated as data, not code: it is parsed with ``json`` and
validated before any array is built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from group_sync_norm import GroupNormTrainer, WorkerFrames


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "expected_workers",
        "workers",
        "gamma",
        "beta",
        "epsilon",
        "running",
        "inference_policy",
        "wire_scalars_per_group_channel",
        "budget_bytes",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"scene is missing {sorted(missing)}")
    if set(data["expected_workers"]) != set(data["workers"]):
        raise ValueError("expected_workers must match the worker keys")
    return data


def _rows(data: dict):
    """Flatten the scene into (worker, row) order: x, mask, groups, dy."""
    channels = len(data["gamma"])
    chunks = []
    for name in data["expected_workers"]:
        worker = data["workers"][name]
        x = np.asarray(worker["x"], dtype=np.float64).reshape(-1, channels)
        dy = np.asarray(worker["dy"], dtype=np.float64).reshape(-1, channels)
        rows = x.shape[0]
        chunks.append(
            (
                name,
                x,
                np.asarray(worker["mask"], dtype=bool).reshape(rows),
                list(worker["groups"]),
                dy,
            )
        )
    return chunks


def _frames(chunks, owner):
    """Split rows into one ``WorkerFrames`` per owner label."""
    channels = chunks[0][1].shape[1]
    buckets: dict[str, list] = {}
    offset = 0
    for _, x, mask, groups, dy in chunks:
        for row in range(x.shape[0]):
            bucket = buckets.setdefault(owner[offset + row], [[], [], [], []])
            bucket[0].append(x[row])
            bucket[1].append(mask[row])
            bucket[2].append(groups[row])
            bucket[3].append(dy[row])
        offset += x.shape[0]
    frames = {}
    for label, (xs, masks, groups, dys) in buckets.items():
        frames[label] = WorkerFrames(
            x=np.asarray(xs, dtype=np.float64).reshape(-1, channels),
            mask=np.asarray(masks, dtype=bool),
            groups=groups,
            dy=np.asarray(dys, dtype=np.float64).reshape(-1, channels),
        )
    return frames


def _groups(chunks):
    names = []
    for _, _, _, groups, _ in chunks:
        for group in groups:
            if group not in names:
                names.append(group)
    return names


def _trainer(data):
    trainer = GroupNormTrainer(
        channels=len(data["gamma"]),
        epsilon=data["epsilon"],
        decay=data["inference_policy"]["decay"],
        min_count=data["inference_policy"]["min_count"],
        wire_scalars_per_group_channel=data["wire_scalars_per_group_channel"],
        budget_bytes=data["budget_bytes"],
    )
    trainer.seed_running(data["running"])
    return trainer


def replay(data: dict, split_sizes: list[int]) -> dict:
    chunks = _rows(data)
    groups = _groups(chunks)
    total = sum(x.shape[0] for _, x, _, _, _ in chunks)
    if sum(split_sizes) != total:
        raise ValueError(f"split sizes {split_sizes} do not cover {total} rows")
    owner = []
    for index, size in enumerate(split_sizes):
        owner.extend([f"S{index}"] * size)

    trainer = _trainer(data)
    frames = _frames(chunks, owner)
    step = trainer.train_step(
        workers=frames,
        expected_workers=list(frames),
        groups=groups,
        gamma=data["gamma"],
        beta=data["beta"],
    )
    grads = trainer.backward_last(
        workers={name: frames[name].dy for name in frames},
        gamma=data["gamma"],
    )
    report = trainer.promote_last()
    return {
        "split": split_sizes,
        "forward": step.forward,
        "grads": grads,
        "approved": step.approved,
        "diagnostics": step.diagnostics,
        "promotion": report,
        "trainer": trainer,
        "groups": groups,
    }


def explain(data: dict, group: str) -> dict:
    result = replay(data, [sum(x.shape[0] for _, x, _, _, _ in _rows(data))])
    trainer = result["trainer"]
    step = trainer.last_step
    info = step.explain(group) if step is not None else {"group": group, "observed": False}
    info["inference"] = trainer.explain_inference(group)
    report = result["promotion"] or {}
    info["promotion"] = {
        "advanced": group in report.get("advanced", []),
        "below_min_count": group in report.get("below_min_count", []),
        "fallback_only": group in report.get("fallback_only", []),
    }
    info["min_count"] = data["inference_policy"]["min_count"]
    threshold = max(int(data["inference_policy"]["min_count"]), 2)
    if info.get("count", 0) and not info["promotion"]["advanced"]:
        if info["count"] < threshold:
            info["why_not_updated"] = (
                f"group had {info['count']} valid frame(s) this step, but an approved "
                f"variance needs at least {threshold} (max(min_count, 2)); its "
                "previously approved value is kept unchanged."
            )
        else:
            info["why_not_updated"] = (
                "the step itself was not approved (overflow / skip / duplicate / "
                "non-finite valid row), so no group's statistics advanced."
            )
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay one step under different shardings")
    parser.add_argument("scene", type=Path)
    parser.add_argument(
        "--splits",
        default=None,
        help="comma separated row counts to compare, e.g. 3,4",
    )
    parser.add_argument("--group", default=None, help="explain one group's promotion")
    parser.add_argument("--json", action="store_true", help="machine readable output")
    args = parser.parse_args(argv)

    data = _load(args.scene)
    chunks = _rows(data)
    total = sum(x.shape[0] for _, x, _, _, _ in chunks)
    if args.group is not None:
        info = explain(data, args.group)
        payload = {
            "scene": args.scene.name,
            "group": args.group,
            "info": _jsonable(info),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_default))
        return 0

    if args.splits:
        sizes = [int(part) for part in args.splits.split(",") if part.strip()]
        alternatives = [sizes, [1] * total, [total]]
    else:
        alternatives = [[total], [1] * total]
    runs = [replay(data, sizes) for sizes in alternatives]
    base = runs[0]
    checks = []
    for run in runs[1:]:
        checks.append(
            {
                "split": run["split"],
                "forward_max_abs_diff": float(np.max(np.abs(run["forward"] - base["forward"]))),
                "dx_max_abs_diff": float(np.max(np.abs(run["grads"]["dx"] - base["grads"]["dx"]))),
                "dgamma_max_abs_diff": float(
                    np.max(np.abs(run["grads"]["dgamma"] - base["grads"]["dgamma"]))
                ),
                "dbeta_max_abs_diff": float(
                    np.max(np.abs(run["grads"]["dbeta"] - base["grads"]["dbeta"]))
                ),
            }
        )
    payload = {
        "scene": args.scene.name,
        "rows": total,
        "groups": base["groups"],
        "approved": base["approved"],
        "diagnostics": _jsonable(base["diagnostics"]),
        "promotion": _jsonable(base["promotion"]),
        "comparisons": checks,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_default))
    else:
        print(f"scene {payload['scene']}: {total} rows, groups {payload['groups']}")
        print(f"approved step: {payload['approved']}")
        print(
            "diagnostics: "
            f"masked={payload['diagnostics']['masked_rows']} "
            f"nonfinite={payload['diagnostics']['nonfinite_rows']} "
            f"empty={payload['diagnostics']['empty_groups']}"
        )
        for check in checks:
            print(
                f"split {check['split']}: forward {check['forward_max_abs_diff']:.3e}, "
                f"dx {check['dx_max_abs_diff']:.3e}, "
                f"dgamma {check['dgamma_max_abs_diff']:.3e}"
            )
    return 0


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


if __name__ == "__main__":
    sys.exit(main())
