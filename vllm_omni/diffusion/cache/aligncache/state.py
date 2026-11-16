# SPDX-License-Identifier: Apache-2.0
"""Pair transactions copy metadata; all retained tensors are owned, read-only snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import torch

from .config import AlignCacheConfig

PHASES = ("early", "mid", "late")


class Reason(str, Enum):
    WARMUP_TAIL = "warmup_tail"
    BOOTSTRAP = "bootstrap"
    MAX_AGE = "max_age"
    BUDGET_SKIP = "budget_skip"
    BUDGET_EXACT = "budget_exact"
    FORCE_EXACT = "force_exact"
    THRESHOLD_DISABLED = "threshold_disabled"
    ZERO_NORM = "zero_norm"
    NONFINITE_RISK = "nonfinite_risk"
    INVALID_PREDICTION = "invalid_prediction"
    NONFINITE_EXACT = "nonfinite_exact"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class ModelBinding:
    pipeline: Any
    transformer: Any
    token: object


@dataclass(frozen=True)
class GenerationSignature:
    epoch: int
    component: int
    condition: tuple[int, int]
    guidance_scale: float
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class SchedulePlan:
    timesteps: tuple[float, ...]
    scheduler_type: str
    scheduler_config: str
    scheduler_id: int | None = None
    schedule_id: int | None = None

    @property
    def evaluations(self):
        return len(self.timesteps)


@dataclass(frozen=True)
class StepContext:
    evaluation_index: int
    timestep: float
    component: int
    guidance_scale: float


@dataclass(frozen=True)
class Decision:
    phase: str
    reason: str
    exact: bool
    age: int
    budget: float | None
    threshold: float
    gain: float | None

    def __post_init__(self):
        Reason(self.reason)  # Reject accidental new/unversioned decision labels.


@dataclass
class RollingHistory:
    last: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    sampled: torch.Tensor | None = None
    ratio: float | None = None
    direction: float | None = None
    source: str | None = None
    previous_source: str | None = None
    scales: dict[str, float] = field(default_factory=lambda: dict.fromkeys(PHASES, 1.0))

    def fork(self):
        return replace(self, scales=self.scales.copy())


@dataclass(frozen=True)
class ExactAnchor:
    index: int
    input: torch.Tensor
    cond: torch.Tensor
    uncond: torch.Tensor
    cond_residual: torch.Tensor
    uncond_residual: torch.Tensor
    output_dtype: torch.dtype


@dataclass
class RequestState:
    epoch: int
    guidance: float = 5.0
    completed_pairs: int = 0
    budget: float = 0.0
    gain: float | None = None
    previous_input: torch.Tensor | None = None
    anchor: ExactAnchor | None = None
    cond: RollingHistory = field(default_factory=RollingHistory)
    uncond: RollingHistory = field(default_factory=RollingHistory)
    guided: RollingHistory = field(default_factory=RollingHistory)
    feedback_scale: dict[str, float] = field(default_factory=dict)
    ratio_ema: dict[str, float] = field(default_factory=dict)
    alpha: dict[str, float] = field(default_factory=dict)
    feedback_updates: int = 0
    anchor_reason: str = "no_anchor"

    @classmethod
    def create(cls, epoch: int, config: AlignCacheConfig):
        return cls(
            epoch=epoch,
            feedback_scale={p: config.feedback.scale_init[p] for p in PHASES},
            ratio_ema={p: 1 / max(config.feedback.scale_init[p], 1e-6) for p in PHASES},
            alpha={p: config.anchor.alpha_init[p] for p in PHASES},
        )

    def fork(self):
        return replace(
            self,
            cond=self.cond.fork(),
            uncond=self.uncond.fork(),
            guided=self.guided.fork(),
            feedback_scale=self.feedback_scale.copy(),
            ratio_ema=self.ratio_ema.copy(),
            alpha=self.alpha.copy(),
        )


@dataclass
class PairTransaction:
    decision: Decision
    state: RequestState
    cond_draft: torch.Tensor | None = None
    uncond_draft: torch.Tensor | None = None

    @classmethod
    def begin(cls, state, decision):
        return cls(decision, state.fork())


def storage_bytes(*objects) -> int:
    """Count shared storages once, including storage behind views."""
    seen_objects, storages = set(), {}

    def visit(value):
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                visit(getattr(value, name))

    for obj in objects:
        visit(obj)
    return sum(storages.values())
