# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Declarative model capabilities for component layerwise offload.

Models declare an ``_offload_plan`` class attribute on the pipeline. Both the
ordinary and distributed layerwise backends consume the same plan instead of
embedding model-specific block paths in backend code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from torch import nn


class BackingKind(str, Enum):
    """Component-agnostic host-backing dimension for streamed weights.

    A component's ``backing`` decides where its offloaded blocks live between
    GPU residency windows. The value is the *resolved* result of the model
    declaration intersected with the user configuration and the parallel /
    quantization constraints; anything a resolver cannot serve fails closed to
    ``PINNED`` before a backend touches the mmap path.
    """

    PINNED = "pinned"  # fully pinned host RAM (the historical default)
    CHECKPOINT_MMAP = "checkpoint_mmap"  # rank-local file-backed mmap
    DISK = "disk"  # reserved: durable runtime-layout artifact (future #9536)


@dataclass(frozen=True)
class OffloadPlan:
    """Optional declarative metadata for component layerwise offload.

    Models declare this as a class attribute ``_offload_plan`` on the
    pipeline class. When present, both layerwise backends use it instead of
    model-specific backend branches, making new integrations data-driven.

    If not declared, the offloader falls back to
    ``_layerwise_offload_blocks_attrs`` on each DiT module class.

    :func:`~vllm_omni.diffusion.offloader.plan_resolver.resolve_offload_plan`
    is the only consumer; backends read the resolved artifact it returns.

    Attributes:
        block_attrs: Maps DiT path → tuple of block-list attribute names.
            e.g. ``{"transformer": ("gen_layers",),
                    "transformer.language_model": ("layers",)}``
        offload_submodules: Maps child name → block-list attribute name,
            for large non-DiT submodules within a DiT that should be
            independently offloaded with their own hooks.
            e.g. ``{"context_encoder": "layers"}``
        resident_dit_paths: DiT paths whose leading blocks may be kept on the
            device when ``dlo_resident_layers`` is nonzero. Keeping this
            model-declared avoids applying a consumer-GPU tuning knob to
            auxiliary or dual DiTs unintentionally.
        encoder_component_types: Maps encoder paths to public selector types
            (currently text_encoder). This declaration is used before the
            compatibility name heuristic.
        encoder_block_attrs: Maps encoder paths to streamable block-list paths.
        encoder_dlo_weight_replication: Encoder paths whose loader-produced
            block tensors are identical across the DiT DLO group. Only these
            encoders may use multi-rank AllGather transfer; this must not be
            declared for encoder-TP shards.
        one_shot_component_paths: Component paths that run a single forward per
            prompt, so their weights only need to be read sequentially once.
            One-shot components qualify for sequential-read streaming; a
            ``DISK`` backing declaration additionally requires a backend that
            can serve it, and remains reserved in this release.
        component_backing: Maps component paths to the requested
            :class:`BackingKind`. Missing entries resolve to the backend
            default (DiT→``CHECKPOINT_MMAP``, everything else→``PINNED``).
    """

    on_demand_component_paths: frozenset[str] = field(default_factory=frozenset)

    block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    offload_submodules: dict[str, str] = field(default_factory=dict)
    resident_dit_paths: frozenset[str] = field(default_factory=frozenset)
    encoder_component_types: dict[str, str] = field(default_factory=dict)
    encoder_block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    encoder_dlo_weight_replication: frozenset[str] = field(default_factory=frozenset)

    one_shot_component_paths: frozenset[str] = field(default_factory=frozenset)
    component_backing: dict[str, BackingKind] = field(default_factory=dict)


def get_offload_plan(pipeline: nn.Module) -> OffloadPlan | None:
    """Retrieve the OffloadPlan declared by the pipeline, if any."""
    return getattr(pipeline, "_offload_plan", None)
