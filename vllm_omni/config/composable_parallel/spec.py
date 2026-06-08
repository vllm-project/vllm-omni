# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core declarative types for parallel-strategy specification.

A :class:`StrategySpec` declares one parallelism scheme: the mesh axis it sizes
(kind + degree), how a batch is routed across that axis, and how per-worker
results are aggregated back. A *stack* of specs (one per axis) fully describes a
stage's parallel layout in a runtime-agnostic, data-only form that can be
translated into concrete engine sizing.

The mesh-axis kinds below enumerate every parallelism dimension the contract can
describe. Only a subset is translatable today (see ``translator.py``); the rest
are reserved for future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Tuple

from vllm_omni.config.composable_parallel.aggregation import AggregationPattern
from vllm_omni.config.composable_parallel.routing import RoutingPattern

MeshAxisKind = Literal[
    "tp",
    "dp",
    "pp",
    "ep",
    "sp_ulysses",
    "sp_ring",
    "cfg",
    "vae_pp",
    "hsdp",
    "stage_pp",
    "stage_replica",
    "cp",
]

MESH_AXIS_KINDS: tuple[MeshAxisKind, ...] = (
    "tp",
    "dp",
    "pp",
    "ep",
    "sp_ulysses",
    "sp_ring",
    "cfg",
    "vae_pp",
    "hsdp",
    "stage_pp",
    "stage_replica",
    "cp",
)

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


class SpecMergeConflict(ValueError):
    """Raised when merged hook/kernel specs have conflicting declarations."""


@dataclass(frozen=True)
class MeshAxisSpec:
    """Declares one axis of a process mesh (kind + size)."""

    kind: MeshAxisKind
    size: int

    def __post_init__(self) -> None:
        if self.kind not in MESH_AXIS_KINDS:
            raise ValueError(
                f"MeshAxisSpec.kind must be one of {MESH_AXIS_KINDS}, got {self.kind!r}"
            )
        if self.size <= 0:
            raise ValueError(f"MeshAxisSpec.size must be > 0, got {self.size}")


@dataclass(frozen=True)
class LayerHookSpec:
    """L2 hook slot referenced by StrategySpec; resolved by the model walker."""

    hook_id: str
    target: Optional[str] = None
    category: HookCategory = "other"
    priority: int = 0
    axis_index: int = 0


@dataclass(frozen=True)
class KernelSpec:
    """L3 kernel slot referenced by StrategySpec; resolved by the kernel registry."""

    kernel_id: str
    target: Optional[str] = None
    category: HookCategory = "other"
    priority: int = 0
    axis_index: int = 0
    group_axis_kind: Optional[MeshAxisKind] = None
    requires_collective: bool = False


@dataclass(frozen=True)
class StrategySpec:
    """Declarative contract for one parallelism scheme (data + hook/kernel slots)."""

    name: str
    mesh_axis: MeshAxisSpec
    routing: RoutingPattern
    aggregation: AggregationPattern
    layer_hook_specs: Tuple[LayerHookSpec, ...] = ()
    kernel_specs: Tuple[KernelSpec, ...] = ()
    shard_extension: Mapping[str, Any] = field(default_factory=dict)
