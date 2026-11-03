# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core declarative types for parallel-strategy specification.

A :class:`StrategySpec` declares one parallelism scheme: the mesh axis it sizes
(kind + degree), how a batch is routed across that axis, and how per-worker
results are aggregated back. A *stack* of specs (one per axis) fully describes a
stage's parallel layout in a runtime-agnostic, data-only form that can be
translated into concrete engine sizing.

The mesh-axis kinds below enumerate every parallelism dimension the contract can
describe. Only a subset is wired end-to-end today (see ``translator.py``):

* **Wired** (translatable now): ``tp``, ``dp``, ``pp``, ``ep``,
  ``stage_replica``.
* **Reserved** (declared in the type system but not yet translatable — the
  translator raises ``AxisTranslationError`` for them): ``sp_ulysses``,
  ``sp_ring``, ``cfg``, ``vae_pp``, ``hsdp``, ``stage_pp``, ``cp``.

Reserved kinds fail fast at translation time rather than silently doing nothing,
so declaring one is an explicit "not yet" rather than a no-op.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from vllm_omni.config.composable_parallel.aggregation import AggregationPattern
from vllm_omni.config.composable_parallel.routing import RoutingPattern


class MeshAxisKind(str, Enum):
    """Enumerates every parallelism dimension the strategy contract can describe.

    Subclassing ``str`` keeps existing string comparisons and dict keys working
    (``MeshAxisKind.TP == "tp"`` and both hash the same), while making the type
    refactor-safe: a typo'd kind is an ``AttributeError`` on the enum rather than
    a silently-wrong bare string.
    """

    TP = "tp"
    DP = "dp"
    PP = "pp"
    EP = "ep"
    SP_ULYSSES = "sp_ulysses"
    SP_RING = "sp_ring"
    CFG = "cfg"
    VAE_PP = "vae_pp"
    HSDP = "hsdp"
    STAGE_PP = "stage_pp"
    STAGE_REPLICA = "stage_replica"
    CP = "cp"


# Bare string values, kept as a plain tuple so membership checks and error
# messages read as the kind names (``"tp"``) rather than enum reprs. ``str``-enum
# equality means both ``"tp"`` and ``MeshAxisKind.TP`` satisfy ``in`` checks.
MESH_AXIS_KINDS: tuple[str, ...] = tuple(k.value for k in MeshAxisKind)

HookCategory = Literal[
    "linear",
    "attention_pre",
    "attention_post",
    "ffn_pre",
    "ffn_post",
    "moe_dispatch",
    "other",
]

HOOK_CATEGORY_ORDER: dict[HookCategory, int] = {
    "linear": 0,
    "attention_pre": 1,
    "attention_post": 2,
    "ffn_pre": 3,
    "ffn_post": 4,
    "moe_dispatch": 5,
    "other": 6,
}


class SpecMergeConflictError(ValueError):
    """Raised when merged hook/kernel specs have conflicting declarations."""


@dataclass(frozen=True)
class MeshAxisSpec:
    """Declares one axis of a process mesh (kind + size)."""

    kind: MeshAxisKind
    size: int

    def __post_init__(self) -> None:
        if self.kind not in MESH_AXIS_KINDS:
            raise ValueError(f"MeshAxisSpec.kind must be one of {MESH_AXIS_KINDS}, got {self.kind!r}")
        if self.size <= 0:
            raise ValueError(f"MeshAxisSpec.size must be > 0, got {self.size}")


@dataclass(frozen=True)
class LayerHookSpec:
    """L2 hook slot referenced by StrategySpec; resolved by the model walker."""

    hook_id: str
    target: str | None = None
    category: HookCategory = "other"
    priority: int = 0
    axis_index: int = 0


@dataclass(frozen=True)
class KernelSpec:
    """L3 kernel slot referenced by StrategySpec; resolved by the kernel registry."""

    kernel_id: str
    target: str | None = None
    category: HookCategory = "other"
    priority: int = 0
    axis_index: int = 0
    group_axis_kind: MeshAxisKind | None = None
    requires_collective: bool = False


@dataclass(frozen=True)
class StrategySpec:
    """Declarative contract for one parallelism scheme (data + hook/kernel slots)."""

    name: str
    mesh_axis: MeshAxisSpec
    routing: RoutingPattern
    aggregation: AggregationPattern
    layer_hook_specs: tuple[LayerHookSpec, ...] = ()
    kernel_specs: tuple[KernelSpec, ...] = ()
    shard_extension: Mapping[str, Any] = field(default_factory=dict)
