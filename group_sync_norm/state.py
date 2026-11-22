"""Running statistics, approval, and immutable inference snapshots.

Semantics (mirroring the exchange contract)
===========================================

* A training step is *approved* only when it is an effective optimization step
  **and** its running statistics are accepted together.  Otherwise the whole
  step's statistics are discarded -- there is no partial promotion.
* For an approved step, a group with at least ``max(min_count, 2)`` valid
  frames publishes the *sample* variance (``m2 / (n - 1)``) as its candidate.
  The extra factor of two is not a policy knob: a sample variance needs two
  observations, so a single frame can be trained on but must never overwrite
  an approved variance with a fabricated zero.  The stored variance becomes
  ``decay * old + (1 - decay) * candidate``; with no history the candidate is
  stored unscaled.  The mean uses the same inertia.
* A group below the threshold is **not** updated.  Its previously approved
  value is retained untouched -- it is neither decayed toward an ineligible
  candidate nor cleared.  This is what makes "trainable but inference value
  unchanged" observable rather than mysterious.
* ``global`` aggregates every valid frame and uses the same threshold and
  inertia as any named group.  A sub-threshold round leaves the approved
  ``global`` entry untouched, exactly like a named group.
* Any group without an approved value falls back to the approved ``global``
  entry.  If ``global`` itself has never met the threshold, a request for an
  unapproved group raises :class:`~group_sync_norm.wire.NotInferable`.
* Inference resolves each requested group from an immutable snapshot taken when
  the read *started*.  A call that began before a promotion keeps reading the
  old values even if the promotion completes mid-call.

Dtypes
======
The table is kept in float64 so a long run does not accumulate float32 drift;
parameters are transcoded to the requested working dtype on the way out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .wire import NotInferable, ProtocolError

GLOBAL_GROUP = "global"


@dataclass(frozen=True)
class GroupEntry:
    """One group's approved inference parameters."""

    mean: np.ndarray
    var: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        var = np.asarray(self.var, dtype=np.float64)
        if mean.shape != var.shape or mean.ndim != 1 or mean.size == 0:
            raise ProtocolError("approved statistics must be a non-empty 1-D vector pair")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var))):
            raise ProtocolError("approved statistics must be finite")
        if np.any(var < 0):
            raise ProtocolError("approved variance must be non-negative")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "var", var.copy())


@dataclass(frozen=True)
class InferenceSnapshot:
    """Immutable view of the approved table at a point in time."""

    version: int
    groups: Mapping[str, GroupEntry]
    fallback: GroupEntry | None

    def resolve(self, group: str) -> tuple[GroupEntry, bool]:
        """Return ``(entry, was_fallback)`` for ``group``."""
        if group in self.groups:
            return self.groups[group], False
        if self.fallback is not None:
            return self.fallback, True
        raise NotInferable(group)

    def params(self, group: str, epsilon: float, *, dtype=np.float64):
        """Effective ``(mean, inv_std)`` for inference.

        ``epsilon`` is the public numerical floor, not a correction for an
        under-sampled group: a group that never met ``min_count`` is simply not
        in the table and resolves through the fallback rule instead.
        """
        entry, _ = self.resolve(group)
        mean = entry.mean.astype(dtype, copy=True)
        inv_std = (1.0 / np.sqrt(entry.var + float(epsilon))).astype(dtype)
        return mean, inv_std


class RunningStats:
    """Mutable approved-statistics table owned by the coordinator."""

    def __init__(self, *, decay: float, min_count: int) -> None:
        self._decay = float(decay)
        self._min_count = int(min_count)
        if not (0.0 <= self._decay < 1.0):
            raise ValueError("decay must be in [0, 1)")
        if self._min_count < 1:
            raise ValueError("min_count must be at least 1")
        self._entries: dict[str, GroupEntry] = {}
        self._version = 0

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def min_count(self) -> int:
        return self._min_count

    @property
    def version(self) -> int:
        return self._version

    def snapshot(self) -> InferenceSnapshot:
        """Freeze the table.

        The copy happens against one value of ``version``; a later promotion
        cannot mutate arrays already held by the snapshot.
        """
        groups = {
            name: GroupEntry(mean=entry.mean.copy(), var=entry.var.copy())
            for name, entry in self._entries.items()
        }
        return InferenceSnapshot(
            version=self._version,
            groups=groups,
            fallback=groups.get(GLOBAL_GROUP),
        )

    def export(self) -> dict[str, GroupEntry]:
        """Copy of the current entries, used for rollback."""
        return {
            name: GroupEntry(mean=entry.mean.copy(), var=entry.var.copy())
            for name, entry in self._entries.items()
        }

    def restore(self, entries: Mapping[str, GroupEntry]) -> None:
        """Replace the table with ``entries`` and advance the version.

        Rollback always advances the version even if the restored bytes equal
        the current ones, because the *decision history* changed.
        """
        self._entries = {
            name: GroupEntry(mean=entry.mean.copy(), var=entry.var.copy())
            for name, entry in entries.items()
        }
        self._version += 1

    def prime(self, group: str, mean, var) -> None:
        """Preload an approved entry (seed a service that already has history)."""
        self._entries[group] = GroupEntry(mean=mean, var=var)
        self._version += 1

    def apply(self, *, approved: bool, groups: Mapping[str, tuple]) -> dict | None:
        """Advance the table by one training step.

        ``groups`` maps each observed group name to ``(mean, sample_var,
        count)`` where ``mean`` and ``sample_var`` are already restored to the
        group's own units.  ``count`` is the *valid* frame count; groups with
        ``count == 0`` are ignored.

        ``approved=False`` (gradient overflow, upstream skip, conflicting
        duplicate, ...) leaves the table exactly as it was and returns ``None``.

        Returns a report with

        ``advanced``        groups whose stored value changed
        ``below_min_count`` groups that were trainable this step but had too
        ``fallback_only``   groups whose only inference path is ``global``
        """
        if not approved:
            return None

        valid = {
            name: item
            for name, item in groups.items()
            if int(item[2]) > 0
        }
        if not valid:
            return None

        report: dict[str, object] = {
            "advanced": [],
            "below_min_count": [],
            "fallback_only": [],
            "observed": sorted(valid),
            "version_before": self._version,
            "version_after": None,
        }

        # Named groups and ``global`` share one rule.  ``global`` is a peer of
        # the named entries, not their average of averages: it is computed from
        # every valid frame before it ever reaches this method.
        for name in sorted(valid):
            mean, sample_var, count = valid[name]
            if int(count) < max(self._min_count, 2):
                report["below_min_count"].append(name)
                continue
            mean = np.asarray(mean, dtype=np.float64)
            sample_var = np.asarray(sample_var, dtype=np.float64)
            old = self._entries.get(name)
            if old is None:
                entry = GroupEntry(mean=mean, var=sample_var)
            else:
                entry = GroupEntry(
                    mean=self._decay * old.mean + (1.0 - self._decay) * mean,
                    var=self._decay * old.var + (1.0 - self._decay) * sample_var,
                )
            self._entries[name] = entry
            report["advanced"].append(name)

        for name in valid:
            if name != GLOBAL_GROUP and name not in self._entries:
                report["fallback_only"].append(name)

        # Only a round that actually changed an entry publishes a new version.
        # A step whose groups were all below the threshold leaves the approved
        # table untouched, so already-open reads see no version movement.
        if report["advanced"]:
            self._version += 1
        report["version_after"] = self._version
        return report


@dataclass
class InferenceCall:
    """A read call bound to the snapshot taken when it started.

    The snapshot is never re-read, so concurrent promotion cannot change the
    parameters of a call that has already begun.
    """

    snapshot: InferenceSnapshot

    @property
    def version(self) -> int:
        return self.snapshot.version

    def normalize(self, group: str, x, gamma, beta, epsilon: float, *, dtype=np.float64):
        entry, _ = self.snapshot.resolve(group)
        mean = entry.mean.astype(dtype, copy=True)
        inv_std = (1.0 / np.sqrt(entry.var + float(epsilon))).astype(dtype)
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != mean.size:
            raise ValueError("input channel count does not match the approved parameters")
        q = (x - mean) * np.asarray(inv_std, dtype=np.float64)
        return q * np.asarray(gamma, dtype=np.float64) + np.asarray(beta, dtype=np.float64)

