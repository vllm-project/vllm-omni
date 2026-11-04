# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion packed-layout plan type definitions.

Mirrors :mod:`vllm_omni.diffusion.distributed.sp_plan`: models declare *where*
the chunk-stacked layout starts, ends and has to be expanded, while the hooks
in :mod:`vllm_omni.diffusion.hooks.mixfusion` implement *how*.

A ``_mixfusion_plan`` maps module names (same syntax as ``_sp_plan``: ``""``
for the root, dotted paths, one ``*`` per ModuleList wildcard) to the types
below. The layout depends on the per-step batch, so the MixFusionDriver arms a
:class:`~vllm_omni.diffusion.forward_context.MixFusionStep` and the root hook
asks the model's :class:`~vllm_omni.diffusion.mixfusion.adapter.MixFusionAdapter`
for the concrete
:class:`~vllm_omni.diffusion.mixfusion.execution_plan.PackedExecutionPlan`.

Example::

    _mixfusion_plan = {
        "": MixFusionRoot(),  # pack before forward, restore the list after
        # Hand out adapter-computed packed hidden states / RoPE tables.
        "image_rope_prepare": MixFusionReplaceOutput(boundary="rope"),
        "transformer_blocks.*.img_mod": MixFusionExpandRows(),  # per chunk
        "norm_out": MixFusionExpandArg(arg_name="conditioning_embedding"),
    }
"""

from __future__ import annotations

from dataclasses import dataclass

# =============================================================================
# MixFusion plan types
# =============================================================================


@dataclass(frozen=True)
class MixFusionRoot:
    """Root (``""``) boundary: pack before forward, reassemble after it.

    It is the only boundary that sees both the per-request latent list and the
    text stream (whose padded width participates in every RoPE table), so both
    directions delegate to the model's :class:`MixFusionAdapter`.
    """


@dataclass(frozen=True)
class MixFusionReplaceOutput:
    """Return an adapter-computed value instead of running this forward.

    The value is ``PackedInputs.boundary_outputs[boundary]``, computed once by
    the adapter at the root where the text stream is still visible; the
    module's original forward never runs on the packed batch.
    """

    boundary: str


@dataclass(frozen=True)
class MixFusionExpandRows:
    """Output side: expand a per-request ``[B, ...]`` tensor to per-chunk rows.

    E.g. AdaLN modulation parameters consumed by the chunk-stacked image stream.
    """


@dataclass(frozen=True)
class MixFusionExpandArg:
    """Input side: expand forward argument ``arg_name`` to per-chunk rows.

    E.g. ``norm_out(x, conditioning_embedding)`` consumes request-level
    conditioning alongside chunk-stacked hidden states. The argument is bound
    by signature; ``expected_dims`` rejects signature drift instead of silently
    expanding the wrong tensor.
    """

    arg_name: str
    expected_dims: int = 2
