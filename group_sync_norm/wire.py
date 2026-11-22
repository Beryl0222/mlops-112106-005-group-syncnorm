"""Wire-level messages and their validation.

Only per-group / per-channel sufficient statistics travel between peers.
Raw frames never do, so a merger can never be replaced by "gather everybody's
rows and reduce locally".  The encodings below are the *entire* cross-process
vocabulary of this package.

Framing
=======
Every message is a JSON-shaped ``dict``.  ``key`` is unique per exchange, so a
duplicate with identical content is dropped (the sender merely re-delivered)
while a duplicate with different content is rejected.  ``status`` is one of

``ok``        the sender holds data for the group (possibly zero valid frames)
``none``      the sender declares the group but has no frames for it at all
``absent``    the sender is not a participant for that group in this step
``skipped``   the sender did not execute (or could not reduce) this group

``none``/``absent``/``skipped`` never carry statistics.  Learning why nothing
arrived therefore needs no extra round trip: a peer that waits for a group it
never announced receives ``absent`` (it is not a member) or ``ok`` with
``count == 0`` (it is a member but has no frames).

Stages
======
``propose``        one ``lo``/``hi`` pair per channel, in the group's own units
``reduce``         ``count``, ``nonfinite`` and per-channel ``mean``/``m2``

Only these sufficient statistics ever cross a partition boundary.  The
coordinator reconstructs a group's mean/variance from them; it never needs --
and cannot obtain -- the rows they came from.
``context``        merged per-channel ``mean``/``inv_std`` broadcast back
``grad``           per-channel ``sum_g``/``sum_gq`` for the affine/BN chain rule

``propose`` is what lets peers agree on a shift/scale frame before any moment
is reduced.  Without it, merging means around ``1e8`` would cancel the variance
of nearby samples; with it, every reduced value is an exactly representable
``q = (x - shift) / scale`` and the reduction is stable.

Budget
======
``wire_scalars_per_group_channel`` bounds each stage per worker per group per
channel.  The built-in stages use 2 + 2 + 2 + 2 = 8 scalars, comfortably
inside the sample's 20.  Empty/absent/skipped messages cost nothing: they are
the reports that keep a peer from waiting forever.

Each stage is a *whole-group* barrier in the step: the coordinator merges one
stage for one group across all participants before any worker proceeds to the
next stage.  Within a stage nothing waits -- a ``none``/``absent`` report is a
complete contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

OK = "ok"
NONE = "none"
ABSENT = "absent"
SKIPPED = "skipped"
STATUSES = (OK, NONE, ABSENT, SKIPPED)

PROPOSE = "propose"
REDUCE = "reduce"
CONTEXT = "context"
GRAD = "grad"
STAGES = (PROPOSE, REDUCE, CONTEXT, GRAD)

PHASE_TRAIN = "train"
PHASE_INFERENCE = "inference"
PHASES = (PHASE_TRAIN, PHASE_INFERENCE)

#: Scalars published per group and channel in each stage.
STAGE_SCALARS: dict[str, int] = {
    PROPOSE: 2,   # lo, hi
    REDUCE: 2,    # mean, m2
    CONTEXT: 2,   # mean, inv_std
    GRAD: 2,      # sum_g, sum_gq, merged by the coordinator
}

#: Default ceiling from the sample exchange contract.
DEFAULT_WIRE_SCALARS = 20

#: Two-sided ULP allowance when checking that a published value lies inside a
#: declared range.  Messages carry exact IEEE doubles; this is the rounding
#: slack at the range edge, not a display tolerance.
RANGE_TOLERANCE_ULPS = 2


class ProtocolError(ValueError):
    """A peer sent something the exchange format does not allow."""


class RangeViolation(ProtocolError):
    """A participant's reduction disagrees with its declared value range.

    Per-channel range is a distributed invariant: publishing ``[lo, hi]``
    promises that every value reduced lies inside it.  A participant that
    announces no range (or an empty group) and then publishes moments has
    broken the agreement, so the group cannot be trusted at the published
    precision (1e-5 against the FP64 reference).
    """

    def __init__(self, group: str, detail: str) -> None:
        super().__init__(f"range violation for group {group!r}: {detail}")
        self.group = group
        self.detail = detail


class IncompleteExchange(RuntimeError):
    """One or more expected participants have not delivered this step.

    The operator never blocks on a missing peer: the exception names the
    missing participants and the affected groups, so the caller can decide to
    wait or abandon the step.  A partition that has no frames for a group still
    delivers a ``none`` message; that is *not* an incomplete exchange, and it
    never affects any other group's progress.
    """

    def __init__(self, missing: Sequence[str], *, groups: Sequence[str] = ()) -> None:
        super().__init__(f"missing participant messages: {list(missing)}")
        self.missing = tuple(missing)
        self.groups = tuple(groups)


class BudgetExceeded(RuntimeError):
    """The step would store or exchange more than the declared budget."""

    def __init__(self, used: int, budget: int, stage: str) -> None:
        super().__init__(f"{stage} needs {used} bytes, budget is {budget}")
        self.used = used
        self.budget = budget
        self.stage = stage


class NotInferable(RuntimeError):
    """No approved statistics are available for a requested group."""

    def __init__(self, group: str) -> None:
        super().__init__(
            f"no approved inference statistics for group {group!r} "
            "(neither the group nor the global fallback has been approved)"
        )
        self.group = group


@dataclass(frozen=True)
class GroupStats:
    """Sufficient statistics of one group on one side of the exchange.

    ``mean``/``m2`` are expressed in the agreed shifted/scaled coordinate
    ``q = (x - shift) / scale``, where ``m2`` is the sum of squared deviations
    from ``mean`` (Chan's parallel form).  Keeping ``m2`` rather than
    ``sumsq`` avoids the catastrophic cancellation that ``E[x^2] - E[x]^2``
    suffers on data like the sample's ``1e8`` offsets.  ``count`` is the number
    of valid rows; ``nonfinite`` counts unmasked rows that were rejected.
    """

    count: int = 0
    nonfinite: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    shift: np.ndarray | None = None
    scale: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.count < 0 or self.nonfinite < 0:
            raise ProtocolError("counts must be non-negative")
        if self.count == 0:
            return
        if self.mean is None or self.m2 is None:
            raise ProtocolError("non-empty statistics missing mean/m2")
        shape = np.asarray(self.mean).shape
        if np.asarray(self.m2).shape != shape:
            raise ProtocolError("statistics channel shapes disagree")
        for name in ("mean", "m2"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ProtocolError(f"{name} must be finite")
        if np.any(np.asarray(self.m2) < 0):
            raise ProtocolError("m2 must be non-negative")

    @property
    def empty(self) -> bool:
        return self.count == 0


def merge_stats(left: GroupStats, right: GroupStats) -> GroupStats:
    """Chan-merge two statistics that share a shift/scale frame."""
    if left.empty and right.empty:
        return GroupStats(count=0, nonfinite=left.nonfinite + right.nonfinite)
    if left.empty:
        return right
    if right.empty:
        return left
    for name in ("shift", "scale"):
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None or b is None:
            continue
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ProtocolError(f"cannot merge statistics with different {name}")
    n_a = np.float64(left.count)
    n_b = np.float64(right.count)
    n = n_a + n_b
    delta = np.asarray(right.mean, dtype=np.float64) - np.asarray(left.mean, dtype=np.float64)
    mean = np.asarray(left.mean, dtype=np.float64) + delta * (n_b / n)
    m2 = (
        np.asarray(left.m2, dtype=np.float64)
        + np.asarray(right.m2, dtype=np.float64)
        + delta * delta * (n_a * n_b / n)
    )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(m2)):
        raise OverflowError("merged moments are not finite")
    return GroupStats(
        count=left.count + right.count,
        nonfinite=left.nonfinite + right.nonfinite,
        shift=left.shift,
        scale=left.scale,
        mean=mean,
        m2=m2,
        lo=left.lo,
        hi=right.hi,
    )


def merge_many(stats: Iterable[GroupStats]) -> GroupStats:
    merged = GroupStats()
    for item in stats:
        merged = merge_stats(merged, item)
    return merged


def agree_frame(lo_candidates: Iterable[Sequence[float]], hi_candidates: Iterable[Sequence[float]]):
    """Turn per-worker extremes into one shift/scale frame.

    ``shift`` is the smallest observed value and ``scale`` is the smallest
    power of two that covers the merged extent.  Division by a power of two is
    exact, so ``q = (x - shift) / scale`` does not add rounding to the
    reduction, and ``q`` is guaranteed to lie in ``[0, 1]`` (up to the usual
    representable rounding) for every participating worker.
    """
    los = [np.asarray(v, dtype=np.float64) for v in lo_candidates]
    his = [np.asarray(v, dtype=np.float64) for v in hi_candidates]
    if not los:
        return None, None
    shift = los[0].copy()
    hi = his[0].copy()
    for item in los[1:]:
        shift = np.minimum(shift, item)
    for item in his[1:]:
        hi = np.maximum(hi, item)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = hi - shift
    if not np.all(np.isfinite(extent)):
        raise OverflowError("group extent is not representable in float64")
    scale = np.empty_like(extent)
    for index in np.ndindex(extent.shape):
        value = float(extent[index])
        if value == 0.0:
            scale[index] = 1.0
            continue
        exponent = math.ceil(math.log2(value))
        try:
            candidate = math.ldexp(1.0, exponent)
        except OverflowError as exc:
            raise OverflowError("group scale is not representable in float64") from exc
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise OverflowError("group scale is not representable in float64")
        if candidate < value:  # log2 rounding guard
            candidate = math.ldexp(1.0, exponent + 1)
        if not math.isfinite(candidate):
            raise OverflowError("group scale is not representable in float64")
        scale[index] = candidate
    return shift, scale


def restore_moments(merged: GroupStats, shift: np.ndarray, scale: np.ndarray):
    """Convert merged ``q``-frame moments to the group's own units.

    Returns ``(mean, m2, var)`` with ``var`` the *population* variance
    (``m2 / count``).  Raises :class:`OverflowError` when the restoration would
    leave float64.
    """
    if merged.empty:
        return None, None, None
    scale = np.asarray(scale, dtype=np.float64)
    mean = np.asarray(shift, dtype=np.float64) + scale * np.asarray(merged.mean)
    m2 = scale * scale * np.asarray(merged.m2)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(m2))):
        raise OverflowError("group moments are not representable in float64")
    var = m2 / np.float64(merged.count)
    if not np.all(np.isfinite(var)) or np.any(var < 0):
        raise OverflowError("group variance is not representable in float64")
    return mean, m2, var


# --------------------------------------------------------------------------
# Encoding / decoding
# --------------------------------------------------------------------------


def _as_float_array(value: Any, what: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - numpy detail
        raise ProtocolError(f"{what} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ProtocolError(f"{what} must be finite")
    return array


def _base(stage: str, phase: str, worker: str, group: str, status: str, key: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ProtocolError(f"unknown stage {stage!r}")
    if phase not in PHASES:
        raise ProtocolError(f"unknown phase {phase!r}")
    if status not in STATUSES:
        raise ProtocolError(f"unknown status {status!r}")
    if not isinstance(worker, str) or not worker:
        raise ProtocolError("worker must be a non-empty string")
    if not isinstance(group, str) or not group:
        raise ProtocolError("group must be a non-empty string")
    if not isinstance(key, str) or not key:
        raise ProtocolError("message key must be a non-empty string")
    return {
        "key": key,
        "stage": stage,
        "phase": phase,
        "worker": worker,
        "group": group,
        "status": status,
    }


def encode_propose(
    *, key: str, worker: str, group: str, status: str, lo=None, hi=None,
    phase: str = PHASE_TRAIN, reason: str | None = None,
) -> dict[str, Any]:
    message = _base(PROPOSE, phase, worker, group, status, key)
    if status == OK and lo is not None:
        lo_arr = _as_float_array(lo, "lo")
        hi_arr = _as_float_array(hi, "hi")
        if lo_arr.shape != hi_arr.shape:
            raise ProtocolError("lo/hi shapes disagree")
        if np.any(lo_arr > hi_arr):
            raise ProtocolError("lo must not exceed hi")
        message["lo"] = lo_arr.tolist()
        message["hi"] = hi_arr.tolist()
    elif status == OK:
        raise ProtocolError("ok proposals must carry lo/hi")
    if reason is not None:
        message["reason"] = str(reason)
    return message


def encode_reduce(
    *, key: str, worker: str, group: str, stats: GroupStats,
    phase: str = PHASE_TRAIN, reason: str | None = None, status: str | None = None,
) -> dict[str, Any]:
    if status is None:
        status = OK if not stats.empty else NONE
    if status != OK and not stats.empty:
        raise ProtocolError("non-ok reduce messages must not carry statistics")
    message = _base(REDUCE, phase, worker, group, status, key)
    if not stats.empty:
        message.update(
            {
                "count": int(stats.count),
                "nonfinite": int(stats.nonfinite),
                "mean": np.asarray(stats.mean, dtype=np.float64).tolist(),
                "m2": np.asarray(stats.m2, dtype=np.float64).tolist(),
            }
        )
    if reason is not None:
        message["reason"] = str(reason)
    return message


def encode_context(
    *, key: str, worker: str, group: str, count: int, mean, inv_std,
    phase: str = PHASE_TRAIN,
) -> dict[str, Any]:
    message = _base(CONTEXT, phase, worker, group, OK, key)
    message.update(
        {
            "count": int(count),
            "mean": _as_float_array(mean, "mean").tolist(),
            "inv_std": _as_float_array(inv_std, "inv_std").tolist(),
        }
    )
    return message


def encode_grad(
    *, key: str, worker: str, group: str, count: int, sum_g=None, sum_gq=None,
    phase: str = PHASE_TRAIN,
) -> dict[str, Any]:
    """One worker's per-group gradient reduction.

    ``sum_g`` is ``sum_i dy_i`` and ``sum_gq`` is ``sum_i dy_i * q_i`` where
    ``q`` is the *normalized* value produced by the forward pass.  Therefore
    ``dbeta = sum_g`` and ``dgamma = sum_gq`` for the layer, and the BatchNorm
    chain-rule term is ``gamma * sum_gq``.  Both are merged in the same
    coordinate as the moments, so the reduction is grouped exactly like the
    forward statistics.

    A worker with no finite rows for the group sends ``count == 0``, which is
    encoded as ``status = "none"``: it is a report, not a missing message, and
    it cannot stall the merge.
    """
    count = int(count)
    if count < 0:
        raise ProtocolError("count must be non-negative")
    status = OK if count > 0 else NONE
    message = _base(GRAD, phase, worker, group, status, key)
    if status != OK:
        return message
    arrays = {
        "sum_g": _as_float_array(sum_g, "sum_g"),
        "sum_gq": _as_float_array(sum_gq, "sum_gq"),
    }
    shape = arrays["sum_g"].shape
    if any(item.shape != shape for item in arrays.values()):
        raise ProtocolError("gradient channel shapes disagree")
    message.update({"count": count, **{k: v.tolist() for k, v in arrays.items()}})
    return message


def decode(
    message: Mapping[str, Any],
    *,
    stage: str | None = None,
    wire_scalars_per_group_channel: int = DEFAULT_WIRE_SCALARS,
) -> dict[str, Any]:
    """Validate one message and return a normalised copy.

    ``message`` is treated as untrusted input: unknown stages, wrong field
    sets, non-finite numbers and negative counts are all rejected.  The returned
    dict contains only JSON primitives, never references to caller arrays.
    """
    if not isinstance(message, Mapping):
        raise ProtocolError("message must be a mapping")
    message_stage = message.get("stage")
    if message_stage not in STAGES:
        raise ProtocolError(f"unknown stage {message_stage!r}")
    if stage is not None and message_stage != stage:
        raise ProtocolError(f"expected stage {stage!r}, got {message_stage!r}")
    if message.get("phase") not in PHASES:
        raise ProtocolError(f"unknown phase {message.get('phase')!r}")
    if message.get("status") not in STATUSES:
        raise ProtocolError(f"unknown status {message.get('status')!r}")
    for field in ("key", "worker", "group"):
        if not isinstance(message.get(field), str) or not message[field]:
            raise ProtocolError(f"{field} must be a non-empty string")
    status = message["status"]
    out = {field: message[field] for field in ("key", "stage", "phase", "worker", "group", "status")}
    if "reason" in message:
        out["reason"] = str(message["reason"])

    if status != OK:
        if set(message) - set(out) - {"reason"}:
            raise ProtocolError(f"{status} messages must not carry payload")
        out["stats"] = GroupStats()
        out["wire_scalars"] = 0
        return out

    if message_stage == PROPOSE:
        allowed = {"key", "stage", "phase", "worker", "group", "status", "lo", "hi"}
        if set(message) - allowed:
            raise ProtocolError("propose message has unexpected fields")
        if "lo" not in message or "hi" not in message:
            raise ProtocolError("ok proposal must carry lo/hi")
        lo = _as_float_array(message["lo"], "lo")
        hi = _as_float_array(message["hi"], "hi")
        if lo.shape != hi.shape or lo.size == 0:
            raise ProtocolError("lo/hi must be non-empty and the same shape")
        if np.any(lo > hi):
            raise ProtocolError("lo must not exceed hi")
        out["lo"], out["hi"] = lo, hi
        channels = lo.size
        cost = channels * STAGE_SCALARS[PROPOSE]
    elif message_stage == REDUCE:
        allowed = {
            "key", "stage", "phase", "worker", "group", "status", "reason",
            "count", "nonfinite", "mean", "m2",
        }
        if set(message) - allowed:
            raise ProtocolError("reduce message has unexpected fields")
        if not {"count", "nonfinite", "mean", "m2"} <= set(message):
            raise ProtocolError("reduce message is missing a required field")
        count = message.get("count")
        nonfinite = message.get("nonfinite")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ProtocolError("ok reduce messages must have a positive count")
        if not isinstance(nonfinite, int) or isinstance(nonfinite, bool) or nonfinite < 0:
            raise ProtocolError("nonfinite must be a non-negative integer")
        mean = _as_float_array(message.get("mean"), "mean")
        m2 = _as_float_array(message.get("m2"), "m2")
        if mean.shape != m2.shape or mean.size == 0:
            raise ProtocolError("mean/m2 must be non-empty and the same shape")
        if np.any(m2 < 0):
            raise ProtocolError("m2 must be non-negative")
        out["stats"] = GroupStats(count=count, nonfinite=nonfinite, mean=mean, m2=m2)
        channels = mean.size
        cost = channels * STAGE_SCALARS[REDUCE]
    elif message_stage == CONTEXT:
        allowed = {"key", "stage", "phase", "worker", "group", "status", "count", "mean", "inv_std"}
        if set(message) - allowed:
            raise ProtocolError("context message has unexpected fields")
        count = message.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ProtocolError("context count must be positive")
        mean = _as_float_array(message.get("mean"), "mean")
        inv_std = _as_float_array(message.get("inv_std"), "inv_std")
        if mean.shape != inv_std.shape or mean.size == 0:
            raise ProtocolError("mean/inv_std must be non-empty and the same shape")
        if np.any(inv_std <= 0):
            raise ProtocolError("inv_std must be positive")
        out.update({"count": count, "mean": mean, "inv_std": inv_std})
        channels = mean.size
        cost = channels * STAGE_SCALARS[CONTEXT]
    else:  # GRAD
        allowed = {"key", "stage", "phase", "worker", "group", "status", "count", "sum_g", "sum_gq"}
        if set(message) - allowed:
            raise ProtocolError("gradient message has unexpected fields")
        count = message.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ProtocolError("gradient count must be positive")
        fields = {}
        for name in ("sum_g", "sum_gq"):
            fields[name] = _as_float_array(message.get(name), name)
        shape = fields["sum_g"].shape
        if shape == () or shape[0] == 0 or any(v.shape != shape for v in fields.values()):
            raise ProtocolError("gradient arrays must be non-empty and share a shape")
        out.update({"count": count, **fields})
        channels = shape[0]
        cost = channels * STAGE_SCALARS[GRAD]

    if cost > wire_scalars_per_group_channel:
        raise BudgetExceeded(cost, wire_scalars_per_group_channel, message_stage)
    out["wire_scalars"] = int(cost)
    return out


def message_canonical(message: Mapping[str, Any]) -> tuple:
    """A hashable, type-strict projection used to detect conflicting duplicates."""
    items = []
    for key in sorted(message):
        value = message[key]
        if isinstance(value, list):
            items.append((key, tuple(float(v) for v in value)))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            items.append((key, float(value)))
        else:
            items.append((key, value))
    return tuple(items)


def duplicate_key(message: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Identity of a contribution: a second message for the same slot conflicts."""
    return (message["stage"], message["phase"], message["worker"], message["group"])
