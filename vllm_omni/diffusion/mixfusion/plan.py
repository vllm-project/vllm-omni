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
        "": MixFusionRoot(),
        # Run the model's own projection + RoPE boundary once per request and
        # chunk-stack its outputs (the _sp_plan "split_output" boundary).
        "image_rope_prepare": MixFusionChunkOutputs(
            per_request_args=("hidden_states", "img_shapes"),
            outputs=(ChunkedOutput(seq_dim=1), ChunkedOutput(seq_dim=0), PerRequestOutput()),
        ),
        "transformer_blocks.*.img_mod": MixFusionExpandRows(),  # per chunk
        "norm_out": MixFusionExpandArg(arg_name="conditioning_embedding"),
    }

The model's joint attention hands its named streams to
``Attention.forward_streams``; nothing else in ``forward`` branches on
MixFusion.
"""

from __future__ import annotations

from dataclasses import dataclass

# =============================================================================
# MixFusion plan types
# =============================================================================


@dataclass(frozen=True)
class MixFusionRoot:
    """Root (``""``) boundary: pack before forward, reassemble after it.

    The fields name the forward's per-request latent list, its request-level
    text stream (one row per image) with the optional text padding mask, and
    the output attribute holding the image prediction. The defaults follow the
    diffusers DiT convention. The joint attention streams are named ``"text"``
    and ``"image"``.
    """

    ragged_arg: str = "hidden_states"
    text_arg: str = "encoder_hidden_states"
    text_mask_arg: str | None = "encoder_hidden_states_mask"
    output_attr: str = "sample"


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
