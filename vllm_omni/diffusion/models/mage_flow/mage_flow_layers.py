# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Core layers for the padded request-batch Mage-Flow transformer.

The checkpoint-facing module names intentionally match the upstream Mage
implementation. The attention kernel itself is replaced with vLLM-Omni's
backend-selectable ``Attention`` layer.
"""

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.normalization import (
    AdaLayerNormContinuous as DiffusersAdaLayerNormContinuous,
)
from diffusers.models.normalization import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

# Mage's FFN is the Qwen-Image tensor-parallel FeedForward: same ``net.0.proj``/
# ``net.2`` diffusers weight layout, same tanh-approximate GELU, same sharding.
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import FeedForward


def _sequence_parallel_active() -> bool:
    """True while execution is inside an ``_sp_plan``-sharded region."""
    return is_forward_context_available() and get_forward_context().sp_active


def _mask_tokens(hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero out padded tokens. ``None`` means every token is real."""
    return hidden_states if mask is None else hidden_states * mask[..., None]


def _request_isolated_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep BF16 GEMM shapes identical to standalone request execution."""
    if hidden_states.shape[0] == 1:
        return module(hidden_states)
    if attention_mask is None:
        return torch.cat(
            [module(hidden_states[index : index + 1]) for index in range(hidden_states.shape[0])],
            dim=0,
        )

    output = None
    for index in range(hidden_states.shape[0]):
        valid_tokens = attention_mask[index]
        sample_output = module(
            hidden_states[index : index + 1, valid_tokens],
        )
        if output is None:
            output = sample_output.new_zeros(
                (
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    *sample_output.shape[2:],
                )
            )
        output[index, valid_tokens] = sample_output[0]
    assert output is not None
    return output


def apply_rotary_emb_mage_flow(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply Mage's adjacent-pair complex RoPE to ``[B, S, H, D]`` Q/K."""
    if hidden_states.shape[-1] % 2:
        raise ValueError("Mage-Flow RoPE requires an even attention head dimension")
    expected_unbatched = (hidden_states.shape[1], hidden_states.shape[-1] // 2)
    expected_batched = (hidden_states.shape[0], *expected_unbatched)
    if freqs_cis.shape not in (expected_unbatched, expected_batched):
        raise ValueError(
            "RoPE shape does not match image tokens: "
            f"got {tuple(freqs_cis.shape)}, expected {expected_unbatched} "
            f"or {expected_batched}"
        )
    rotated = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    frequencies = freqs_cis[None, :, None, :] if freqs_cis.ndim == 2 else freqs_cis[:, :, None, :]
    output = torch.view_as_real(rotated * frequencies).flatten(-2)
    return output.to(hidden_states.dtype)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Mage's timestep embedding, including its checkpoint-critical rounding.

    The cast to ``timesteps.dtype`` below is load-bearing: under BF16 it diverges from
    ``diffusers``' fp32 frequency table by ~1.13, so the two are not interchangeable.
    """
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1-D tensor")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    frequencies = torch.exp(exponent).to(timesteps.dtype)
    embedding = scale * timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class MageFlowTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.time_proj(timestep)
        return _request_isolated_forward(
            self.timestep_embedder,
            projected.to(dtype=hidden_states.dtype),
        )


class MageFlowEmbedRope(nn.Module):
    """Generate the upstream three-axis, centered image RoPE frequencies."""

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        scale_rope: bool = True,
        max_positions: int = 4096,
    ) -> None:
        super().__init__()
        if any(dim % 2 for dim in axes_dim):
            raise ValueError(f"all axes_dim entries must be even, got {axes_dim}")
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # DiffusersPipelineLoader may instantiate modules under a CUDA or meta
        # default-device context. Pin construction to CPU explicitly; relying
        # on torch.arange()'s ambient device would silently reintroduce the
        # CUDA pow/polar rounding that this cache is intended to avoid.
        positive_indices = torch.arange(max_positions, device="cpu")
        negative_indices = torch.arange(max_positions, device="cpu").flip(0) * -1 - 1

        # Match Qwen-Image and upstream Mage: build the canonical complex64
        # tables on CPU, then lazily move the complete tables to the device
        # requested by forward(). CPU and CUDA pow/polar differ in their last
        # few bits, which is enough to change BF16 rounding after RoPE.
        #
        # These must remain plain attributes rather than registered buffers.
        # The pipeline applies ``module.to(dtype=torch.bfloat16)``; applying
        # that conversion to a complex buffer discards its imaginary phase.
        self.pos_freqs = torch.cat(
            [
                self._rope_params(
                    positive_indices,
                    axis_dim,
                    self.theta,
                )
                for axis_dim in self.axes_dim
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(
                    negative_indices,
                    axis_dim,
                    self.theta,
                )
                for axis_dim in self.axes_dim
            ],
            dim=1,
        )

    @staticmethod
    def _rope_params(
        index: torch.Tensor,
        dim: int,
        theta: int,
    ) -> torch.Tensor:
        frequencies = torch.outer(
            index,
            1.0
            / torch.pow(
                theta,
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                    device="cpu",
                ).div(dim),
            ),
        )
        return torch.polar(torch.ones_like(frequencies), frequencies)

    @staticmethod
    def _canonical_device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device(
                "cuda",
                torch.accelerator.current_device_index(),
            )
        return device

    @lru_cache(maxsize=16)
    def _compute_grid(
        self,
        device: torch.device,
        frame: int,
        height: int,
        width: int,
        frame_offset: int = 0,
    ) -> torch.Tensor:
        if frame < 1 or height < 1 or width < 1:
            raise ValueError(f"image grid dimensions must be positive, got {(frame, height, width)}")
        if self.pos_freqs.device != device:
            # As in Qwen-Image, mutate the single canonical table pair to the
            # active device. The final grid cache is keyed by device, so a
            # cached cuda:0 tensor can never satisfy a cuda:1 request.
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        positive = self.pos_freqs.split(
            [axis_dim // 2 for axis_dim in self.axes_dim],
            dim=1,
        )
        negative = self.neg_freqs.split(
            [axis_dim // 2 for axis_dim in self.axes_dim],
            dim=1,
        )
        if frame_offset + frame > positive[0].shape[0] or height > positive[1].shape[0] or width > positive[2].shape[0]:
            raise ValueError(f"image grid {(frame, height, width)} exceeds the RoPE cache")

        frame_freqs = (
            positive[0][frame_offset : frame_offset + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        )
        if self.scale_rope:
            height_freqs = torch.cat(
                [
                    negative[1][-(height - height // 2) :],
                    positive[1][: height // 2],
                ]
            )
            width_freqs = torch.cat(
                [
                    negative[2][-(width - width // 2) :],
                    positive[2][: width // 2],
                ]
            )
        else:
            height_freqs = positive[1][:height]
            width_freqs = positive[2][:width]

        height_freqs = height_freqs.view(1, height, 1, -1).expand(frame, height, width, -1)
        width_freqs = width_freqs.view(1, 1, width, -1).expand(frame, height, width, -1)
        # Cache the final contiguous device tensor, not an expanded view. This
        # bounds both recomputation and repeated CPU-to-device transfers.
        return (
            torch.cat(
                [frame_freqs, height_freqs, width_freqs],
                dim=-1,
            )
            .reshape(frame * height * width, -1)
            .clone()
            .contiguous()
        )

    def forward(
        self,
        image_grid_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        grids = [image_grid_fhw] if isinstance(image_grid_fhw, tuple) else image_grid_fhw
        if not grids:
            raise ValueError("image_grid_fhw must contain one grid")
        target_device = self._canonical_device(self.pos_freqs.device if device is None else device)
        return torch.cat(
            [
                self._compute_grid(
                    target_device,
                    frame,
                    height,
                    width,
                    frame_offset=index,
                )
                for index, (frame, height, width) in enumerate(grids)
            ],
            dim=0,
        )


class MageFlowImageRopePrepare(nn.Module):
    """Project image tokens and build their RoPE table behind one boundary.

    Sequence parallelism shards the image stream, and the RoPE frequencies must
    be sharded in lockstep with it — a token and its rotation have to land on
    the same rank. Emitting both from a single module gives ``_sp_plan`` one
    place to split them together; ``_sp_plan`` hooks only fire at ``nn.Module``
    boundaries, so the inline computation could not be intercepted.

    ``img_in`` and ``pos_embed`` stay owned by the transformer and are merely
    referenced here, so ``named_parameters()`` keeps reporting the checkpoint's
    own names.
    """

    def __init__(self, img_in: nn.Module, pos_embed: nn.Module) -> None:
        super().__init__()
        self.img_in = img_in
        self.pos_embed = pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_attention_mask: torch.Tensor | None,
        grids_per_sample: list[list[tuple[int, int, int]]],
        max_image_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = hidden_states.device
        frequencies = []
        for grids in grids_per_sample:
            # RoPE owns a bounded device-aware LRU. Pass the execution device
            # explicitly so cached GPU tensors cannot cross device boundaries.
            sample_frequencies = self.pos_embed(grids, device=device)
            padding = max_image_tokens - sample_frequencies.shape[0]
            if padding < 0:
                raise ValueError("Mage-Flow image grid exceeds the padded token length")
            if padding:
                sample_frequencies = torch.cat(
                    [
                        sample_frequencies,
                        torch.ones(
                            padding,
                            sample_frequencies.shape[1],
                            dtype=sample_frequencies.dtype,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            frequencies.append(sample_frequencies)
        image_rotary_emb = torch.stack(frequencies)
        hidden_states = _request_isolated_forward(
            self.img_in,
            hidden_states,
            image_attention_mask,
        )
        return hidden_states, image_rotary_emb


class MageJointAttention(nn.Module):
    """Joint attention over padded per-request ``[text, image]`` tokens."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        added_kv_proj_dim: int,
        bias: bool = True,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        # Fused Q/K/V projections. The official checkpoint ships separate
        # ``to_q``/``to_k``/``to_v`` (and ``add_*_proj``) tensors; they are
        # merged at load time via the transformer's stacked_params_mapping.
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=dim_head,
            total_num_heads=heads,
            bias=bias,
            return_bias=False,
            prefix=f"{prefix}.to_qkv" if prefix else "to_qkv",
        )
        self.add_qkv_proj = QKVParallelLinear(
            hidden_size=added_kv_proj_dim,
            head_size=dim_head,
            total_num_heads=heads,
            bias=bias,
            return_bias=False,
            prefix=f"{prefix}.add_qkv_proj" if prefix else "add_qkv_proj",
        )

        # Per-GPU head counts. Everything downstream of the QKV projection
        # operates on the local shard, so the attention layer and every
        # ``unflatten`` below must use these, not the global ``heads``.
        self.local_heads = self.to_qkv.num_heads
        self.local_kv_heads = self.to_qkv.num_kv_heads
        self.q_size = self.local_heads * dim_head
        self.kv_size = self.local_kv_heads * dim_head

        # RMSNorm normalizes over ``dim_head`` only, so it never reduces across
        # the head dimension and stays correct under TP without extra comms.
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim,
                    query_dim,
                    bias=True,
                    input_is_parallel=True,
                    return_bias=False,
                    prefix=f"{prefix}.to_out.0" if prefix else "to_out.0",
                ),
                nn.Dropout(0.0),
            ]
        )
        self.to_add_out = RowParallelLinear(
            self.inner_dim,
            added_kv_proj_dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=f"{prefix}.to_add_out" if prefix else "to_add_out",
        )

        # ``num_kv_heads`` is left unset: Mage-Flow is pure MHA, so local_kv_heads
        # always equals local_heads and the default already matches.
        self.attention = Attention(
            num_heads=self.local_heads,
            head_size=dim_head,
            softmax_scale=dim_head**-0.5,
            causal=False,
            prefix=prefix,
            role="mage_flow.joint",
            role_category="self",
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
        qkv_proj: nn.Module,
        q_norm: nn.Module,
        k_norm: nn.Module,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = _request_isolated_forward(
            qkv_proj,
            hidden_states,
            attention_mask,
        )
        query, key, value = qkv.split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        query = q_norm(query.unflatten(-1, (self.local_heads, self.dim_head)))
        key = k_norm(key.unflatten(-1, (self.local_kv_heads, self.dim_head)))
        value = value.unflatten(-1, (self.local_kv_heads, self.dim_head))
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("Mage-Flow image and text batch sizes must match")
        # ``None`` stays ``None``: it means "every token is real", the same thing
        # the enclosing block assumes when it skips its own masking. Expanding it
        # into an all-ones mask here would contradict that and, under sequence
        # parallelism -- where the caller passes ``None`` precisely because a
        # full-sequence mask no longer lines up with this rank's shard -- would
        # add a batch of no-op multiplies per layer.
        if image_attention_mask is not None and image_attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("image_attention_mask must match image token dimensions")
        if encoder_attention_mask is not None and encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_attention_mask must match encoder token dimensions")

        img_q, img_k, img_v = self._project(
            hidden_states,
            self.to_qkv,
            self.norm_q,
            self.norm_k,
            image_attention_mask,
        )
        txt_q, txt_k, txt_v = self._project(
            encoder_hidden_states,
            self.add_qkv_proj,
            self.norm_added_q,
            self.norm_added_k,
            encoder_attention_mask,
        )
        img_q = apply_rotary_emb_mage_flow(img_q, image_rotary_emb)
        img_k = apply_rotary_emb_mage_flow(img_k, image_rotary_emb)

        if _sequence_parallel_active():
            # Only the image stream is sharded; the text stream stays replicated.
            # Concatenating here would mix a sharded tensor with a full one, so
            # the text side is handed to the attention backend as joint tensors
            # and spliced in after the all-to-all instead. The masks are all-valid
            # under SP (the transformer rejects padded sequences there), and a
            # full-sequence mask would not line up with this rank's shard anyway.
            image_mask = image_attention_mask
            encoder_mask = encoder_attention_mask
            joint_output = self.attention(
                img_q,
                img_k,
                img_v,
                attn_metadata=AttentionMetadata(
                    joint_query=txt_q,
                    joint_key=txt_k,
                    joint_value=txt_v,
                    joint_strategy="front",
                ),
            )
        else:
            # Padding is handed to the attention backend as a joint mask, the
            # same contract Qwen-Image and Flux use: the backend unpads, runs
            # varlen attention and repads with zeros.
            #
            # Two absent masks stay absent rather than being materialised into
            # an all-ones mask, which saves building it and the multiplies that
            # consume it.
            #
            # An earlier version of this comment justified that numerically,
            # claiming an all-valid mask shifts the output because CUDNN_ATTN
            # forwards it to SDPA rather than short-circuiting to the dense
            # kernel (citing SSIM 0.9977). Comparing mask against no-mask on
            # that backend (sm_120, cuDNN 9.19) instead produced bit-identical
            # images, so treat this as a work reduction only, and re-measure on
            # your own backend before relying on the stronger claim.
            image_mask = image_attention_mask
            encoder_mask = encoder_attention_mask
            attn_metadata = None
            if image_mask is not None or encoder_mask is not None:
                if image_mask is None:
                    image_mask = hidden_states.new_ones(hidden_states.shape[:2], dtype=torch.bool)
                if encoder_mask is None:
                    encoder_mask = encoder_hidden_states.new_ones(encoder_hidden_states.shape[:2], dtype=torch.bool)
                attn_metadata = AttentionMetadata(
                    attn_mask=torch.cat([encoder_mask, image_mask], dim=1).to(torch.bool),
                )
            joint_output = self.attention(
                torch.cat([txt_q, img_q], dim=1),
                torch.cat([txt_k, img_k], dim=1),
                torch.cat([txt_v, img_v], dim=1),
                attn_metadata=attn_metadata,
            )

        txt_output, img_output = joint_output.split(
            [encoder_hidden_states.shape[1], hidden_states.shape[1]],
            dim=1,
        )
        # ``to_out``/``to_add_out`` are row-parallel: their outputs are already
        # all-reduced back to the full query/context dim, not the per-GPU shard.
        img_output = self.to_out[1](self.to_out[0](img_output.flatten(2, 3)))
        txt_output = self.to_add_out(txt_output.flatten(2, 3))
        return (
            _mask_tokens(img_output, image_mask),
            _mask_tokens(txt_output, encoder_mask),
        )


class MageFlowDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # The modulation projections stay replicated. Their 6*dim output is
        # consumed as six separate [B, dim] shift/scale/gate vectors that
        # modulate the *full* (unsharded) residual stream, so column-sharding
        # them would break the chunk() layout. Qwen-Image makes the same call.
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                dim,
                6 * dim,
                bias=True,
                return_bias=False,
                prefix=f"{prefix}.img_mod.1" if prefix else "img_mod.1",
            ),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = MageJointAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=True,
            eps=eps,
            prefix=f"{prefix}.attn" if prefix else "",
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            prefix=f"{prefix}.img_mlp" if prefix else "img_mlp",
        )

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                dim,
                6 * dim,
                bias=True,
                return_bias=False,
                prefix=f"{prefix}.txt_mod.1" if prefix else "txt_mod.1",
            ),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            prefix=f"{prefix}.txt_mlp" if prefix else "txt_mlp",
        )

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = modulation.chunk(3, dim=-1)
        return (
            hidden_states * (1 + scale[:, None]) + shift[:, None],
            gate[:, None],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_attention_kwargs:
            unsupported = ", ".join(sorted(joint_attention_kwargs))
            raise ValueError(f"Mage-Flow dense attention does not accept attention kwargs: {unsupported}")
        img_mod1, img_mod2 = _request_isolated_forward(
            self.img_mod,
            temb,
        ).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = _request_isolated_forward(
            self.txt_mod,
            temb,
        ).chunk(2, dim=-1)

        img_normed, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_normed, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        img_attn, txt_attn = self.attn(
            img_normed,
            txt_normed,
            image_rotary_emb,
            image_attention_mask=image_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn
        hidden_states = _mask_tokens(hidden_states, image_attention_mask)
        encoder_hidden_states = _mask_tokens(encoder_hidden_states, encoder_attention_mask)

        img_normed, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        txt_normed, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        img_mlp_output = _request_isolated_forward(
            self.img_mlp,
            img_normed,
            image_attention_mask,
        )
        hidden_states = hidden_states + img_gate2 * img_mlp_output
        txt_mlp_output = _request_isolated_forward(
            self.txt_mlp,
            txt_normed,
            encoder_attention_mask,
        )
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        hidden_states = _mask_tokens(hidden_states, image_attention_mask)
        encoder_hidden_states = _mask_tokens(encoder_hidden_states, encoder_attention_mask)
        return encoder_hidden_states, hidden_states


class AdaLayerNormContinuous(DiffusersAdaLayerNormContinuous):
    """diffusers' ``AdaLayerNormContinuous`` with request-isolated modulation.

    The modulation projection is the only difference: it runs one request at a
    time so its GEMM shape matches standalone execution.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        conditioning_embedding = self.silu(conditioning_embedding).to(hidden_states.dtype)
        scale, shift = _request_isolated_forward(
            self.linear,
            conditioning_embedding,
        ).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
