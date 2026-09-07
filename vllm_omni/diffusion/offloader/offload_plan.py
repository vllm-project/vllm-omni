# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative model capabilities for component layerwise offload.

Models declare an ``_offload_plan`` class attribute on the pipeline. Both the
ordinary and distributed layerwise backends consume the same plan instead of
embedding model-specific block paths in backend code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class OffloadPlan:
    """Optional declarative metadata for component layerwise offload.

    Models declare this as a class attribute ``_offload_plan`` on the
    pipeline class. When present, both layerwise backends use it instead of
    model-specific backend branches, making new integrations data-driven.

    If not declared, the offloader falls back to:
    1. ``_layerwise_offload_blocks_attrs`` on each DiT module class.
    2. Heuristic search for ``layers`` / ``blocks`` / ``h`` attributes.

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
    """

    on_demand_component_paths: frozenset[str] = field(default_factory=frozenset)

    block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    offload_submodules: dict[str, str] = field(default_factory=dict)
    resident_dit_paths: frozenset[str] = field(default_factory=frozenset)
    encoder_component_types: dict[str, str] = field(default_factory=dict)
    encoder_block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    encoder_dlo_weight_replication: frozenset[str] = field(default_factory=frozenset)


def get_offload_plan(pipeline: nn.Module) -> OffloadPlan | None:
    """Retrieve the OffloadPlan declared by the pipeline, if any."""
    return getattr(pipeline, "_offload_plan", None)
