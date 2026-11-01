# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion packed-layout plan type definitions.

Mirrors :mod:`vllm_omni.diffusion.distributed.sp_plan`: models only declare
*where* the chunk-stacked MixFusion layout starts, ends and has to be
expanded; the hooks in :mod:`vllm_omni.diffusion.hooks.mixfusion` implement
*how*. The declaration keeps ``forward`` free of any MixFusion branching.

A ``_mixfusion_plan`` is a dictionary mapping module names (same syntax as
``_sp_plan``: ``""`` for the root, dotted paths, one ``*`` for a ModuleList
wildcard) to one of the declarative types below.

Unlike an ``_sp_plan``, the packed layout is data-dependent per denoise step
(chunk count and request shapes), so there is no static config object: the
pipeline arms a :class:`~vllm_omni.diffusion.forward_context.MixFusionStep`
on the forward context and the root hook derives the concrete
:class:`~vllm_omni.diffusion.mixfusion.layout.MixFusionPlan` from the batch.

Example::

    class MyTransformer(nn.Module):
        _mixfusion_plan = {
            # Pack the per-request latent list before forward, restore the
            # per-request list after it.
            "": MixFusionRoot(pack_method="prepare_mixfusion_inputs"),
            # Hand out the packed hidden states / RoPE tables computed at
            # the root instead of recomputing them on the packed batch.
            "image_rope_prepare": MixFusionRopePrepare(),
            # Expand per-request conditioning rows to per-chunk rows.
            "transformer_blocks.*.img_mod": MixFusionExpandRows(),
            "norm_out": MixFusionExpandArg(arg_index=1),
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

    The root is the only boundary that sees both the per-request latent list
    and the text stream (whose padded width participates in every RoPE
    table), so the packing work has to start there. ``pre_forward`` calls the
    transformer method named by ``pack_method`` and replaces the
    ``hidden_states`` keyword with the chunk-stacked batch;
    ``post_forward`` reassembles the per-request outputs.

    Args:
        pack_method: Name of the transformer method that turns a list of
            per-request latents into the packed inputs, e.g.
            ``QwenImageTransformer2DModel.prepare_mixfusion_inputs``.
    """

    pack_method: str = "prepare_mixfusion_inputs"


@dataclass(frozen=True)
class MixFusionRopePrepare:
    """``image_rope_prepare`` boundary: return the root-computed packed tensors.

    The packed hidden states (already projected by ``img_in``) and the RoPE
    tables are produced by the root hook, because only that boundary also
    sees the text stream. Recomputing them here would either run on the
    wrong layout or repeat the projection / ``pos_embed`` work.
    """


@dataclass(frozen=True)
class MixFusionExpandRows:
    """Output side: expand a per-request ``[B, ...]`` tensor to per-chunk rows.

    Use on modules whose per-request output feeds chunk-level consumers
    (e.g. AdaLN modulation parameters on the chunk-stacked image stream).
    """


@dataclass(frozen=True)
class MixFusionExpandArg:
    """Input side: expand positional argument ``arg_index`` to per-chunk rows.

    Use on modules that consume a per-request conditioning tensor alongside
    chunk-stacked hidden states (e.g. ``norm_out(x, temb)``).

    Args:
        arg_index: Index of the positional argument to expand.
        expected_dims: Rank the argument must have before expansion; guards
            against signature drift turning the expansion into a silent
            reshape of the wrong tensor.
    """

    arg_index: int
    expected_dims: int = 2
