# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion packed-layout plan type definitions.

Mirrors :mod:`vllm_omni.diffusion.distributed.sp_plan`: models declare *where*
the chunk-stacked layout starts, ends and has to be expanded, while the hooks
in :mod:`vllm_omni.diffusion.hooks.mixfusion` and the framework's
:class:`~vllm_omni.diffusion.mixfusion.chunked.ChunkedJointAdapter` implement
*how*. A ``_mixfusion_plan`` maps module names (same syntax as ``_sp_plan``:
``""`` for the root, dotted paths, one ``*`` per ModuleList wildcard) to the
types below.

Example (Qwen-Image)::

    _mixfusion_plan = {
        # Per-request latent list in, one prediction per request out.
        "": MixFusionRoot(
            request=MixFusionImageRequest(default_true_cfg_scale=4.0),
            disabled_if=("zero_cond_t",),
            required_inputs=("img_shapes", "txt_seq_lens"),
            unsupported_inputs=("image_latents",),
        ),
        # Run the model's own projection + RoPE boundary once per request and
        # chunk-stack its outputs (the _sp_plan "split_output" boundary).
        "image_rope_prepare": MixFusionChunkOutputs(
            per_request_args=("hidden_states", "img_shapes"),
            outputs=(ChunkedOutput(seq_dim=1), ChunkedOutput(seq_dim=0), PerRequestOutput()),
        ),
        "transformer_blocks.*.img_mod": MixFusionExpandRows(),  # per chunk
        "norm_out": MixFusionExpandArg(arg_name="conditioning_embedding"),
    }

Joint attention is the one place a model branches on MixFusion: under an armed
step it passes the plan in ``AttentionMetadata.extra[MIXFUSION_PLAN_KEY]`` and
``Attention.forward`` runs the packed path.
"""

from __future__ import annotations

from dataclasses import dataclass

# =============================================================================
# MixFusion plan types
# =============================================================================


@dataclass(frozen=True)
class MixFusionImageRequest:
    """Declare a text-to-image request's grid before tensors are allocated.

    Attribute paths are resolved on the loaded pipeline for VAE/default size,
    and on the plan's transformer for channels. Integer constants work too.
    ``patch_size`` describes latent packing, not the attention kernel.
    Other input layouts need their own declaration before admission can relax
    their shapes; this declaration accepts plain text prompts only.
    """

    vae_scale_factor: str | int = "vae_scale_factor"
    default_sample_size: str | int = "default_sample_size"
    channels: str | int = "in_channels"
    patch_size: int = 2
    default_true_cfg_scale: float = 1.0


@dataclass(frozen=True)
class MixFusionRoot:
    """Root (``""``) boundary: pack before forward, reassemble after it.

    The fields name the forward's per-request latent list, its request-level
    text stream (one row per image) with the optional text padding mask, and
    the output attribute holding the image prediction. The defaults follow the
    diffusers DiT convention. The joint attention streams are named ``"text"``
    and ``"image"``.

    ``request`` describes host-side admission; omitting it keeps dense
    scheduling. The remaining declarations guard runtime mode and inputs
    without adding configuration/preparation/validation methods to the model.
    """

    ragged_arg: str = "hidden_states"
    text_arg: str = "encoder_hidden_states"
    text_mask_arg: str | None = "encoder_hidden_states_mask"
    output_attr: str = "sample"
    request: MixFusionImageRequest | None = None
    # Truthy transformer attributes disable packed admission at startup.
    disabled_if: tuple[str, ...] = ()
    # InputBatch fields checked by the common adapter before execution.
    required_inputs: tuple[str, ...] = ()
    unsupported_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkedOutput:
    """Chunk-stack one output along its sequence dimension.

    ``seq_dim=0``: each request returns ``[S, ...]`` and the result is
    ``[num_chunks, chunk_size, ...]``. ``seq_dim=1``: each request returns
    ``[1, S, ...]`` and the result has the same ``[num_chunks, chunk_size,
    ...]`` layout.
    """

    seq_dim: int


@dataclass(frozen=True)
class PerRequestOutput:
    """Stack one output per request (e.g. a text RoPE table).

    ``batched=False``: each request returns ``[L, ...]`` and the result is
    ``[num_requests, L, ...]``; ``batched=True``: each returns ``[1, L, ...]``.
    """

    batched: bool = False


@dataclass(frozen=True)
class MixFusionChunkOutputs:
    """Run this module once per request and chunk-stack its outputs.

    The packed-path counterpart of an ``_sp_plan`` ``split_output`` boundary:
    the module still receives the per-request latent list, so each request's
    call sees its own geometry. ``per_request_args`` are sliced per request
    (a per-request list gives its item as a batch of one, a per-image tensor
    its row); every other argument is passed through unchanged. ``outputs``
    says how each returned value is combined, in return order.
    """

    per_request_args: tuple[str, ...]
    outputs: tuple[ChunkedOutput | PerRequestOutput, ...]


@dataclass(frozen=True)
class MixFusionReplaceOutput:
    """Return an adapter-computed value instead of running this forward.

    Only for a custom ``MixFusionAdapter`` (the escape hatch for layouts the
    declarative plan cannot express): the value is
    ``PackedInputs.boundary_outputs[boundary]``, computed by the adapter at
    the root; the module's original forward never runs on the packed batch.
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
