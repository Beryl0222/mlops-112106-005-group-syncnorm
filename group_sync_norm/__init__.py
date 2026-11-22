"""Grouped synchronized BatchNorm as a tensor operator.

Public surface
==============

``GroupNormTrainer``    coordinator: forward, backward, promotion, inference.
``WorkerFrames``        the rows one partition holds for a step.
``RunningStats``        the approved-statistics table (``global`` fallback).
``InferenceCall``       an immutable snapshot-bound inference read.

Quick start::

    trainer = GroupNormTrainer(channels=2, epsilon=1e-5, decay=0.85, min_count=3)
    trainer.seed_running({"global": {"mean": [0, 0], "var": [1, 2]}})
    step = trainer.train_step(
        workers={"P0": WorkerFrames(x, mask, groups, dy), "P1": ...},
        expected_workers=["P0", "P1"],
        groups=["large", "rare"],
        promote=True,
    )
    grads = trainer.backward_last(workers={"P0": dy0, "P1": dy1})
    call = trainer.open_inference()
    y = call.normalize("rare", x, gamma, beta, epsilon=1e-5)
"""

from .state import (
    GLOBAL_GROUP,
    GroupEntry,
    InferenceCall,
    InferenceSnapshot,
    NotInferable,
    RunningStats,
)
from .trainer import (
    DuplicateContribution,
    GroupNormTrainer,
    LocalReduce,
    StepRejected,
    StepResult,
    WorkerFrames,
    merge_contributions,
    merge_group_stats,
    reduce_local,
)
from .wire import (
    ABSENT,
    BudgetExceeded,
    CONTEXT,
    DEFAULT_WIRE_SCALARS,
    GRAD,
    IncompleteExchange,
    NONE,
    OK,
    PHASE_INFERENCE,
    PHASE_TRAIN,
    PROPOSE,
    ProtocolError,
    RANGE_TOLERANCE_ULPS,
    REDUCE,
    RangeViolation,
    SKIPPED,
    GroupStats,
    agree_frame,
    decode,
    encode_context,
    encode_grad,
    encode_propose,
    encode_reduce,
    merge_many,
    merge_stats,
    restore_moments,
)

__all__ = [
    "ABSENT",
    "BudgetExceeded",
    "CONTEXT",
    "DEFAULT_WIRE_SCALARS",
    "DuplicateContribution",
    "GLOBAL_GROUP",
    "GRAD",
    "GroupEntry",
    "GroupNormTrainer",
    "GroupStats",
    "IncompleteExchange",
    "InferenceCall",
    "InferenceSnapshot",
    "LocalReduce",
    "NONE",
    "NotInferable",
    "OK",
    "PHASE_INFERENCE",
    "PHASE_TRAIN",
    "PROPOSE",
    "ProtocolError",
    "REDUCE",
    "RANGE_TOLERANCE_ULPS",
    "RangeViolation",
    "RunningStats",
    "SKIPPED",
    "StepRejected",
    "StepResult",
    "WorkerFrames",
    "agree_frame",
    "decode",
    "encode_context",
    "encode_grad",
    "encode_propose",
    "encode_reduce",
    "merge_contributions",
    "merge_group_stats",
    "merge_many",
    "merge_stats",
    "reduce_local",
    "restore_moments",
]
