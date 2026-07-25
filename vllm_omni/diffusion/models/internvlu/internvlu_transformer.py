# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native InternVL-U generation transformer.

The released checkpoint stores a dual-stream MMDiT under
``generation_decoder/decoder.*`` plus a learned ``encoder_padding_token`` at
the component root; the pipeline's ``load_weights`` maps both onto this single
``InternVLUTransformer2DModel``.  The implementation uses vLLM-Omni parallel
linear layers and its attention dispatcher.  The first supported configuration
is TP1; the pipeline validates that restriction explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_dit import ForwardPattern
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTAdapterConfig
from vllm_omni.diffusion.layers.norm import LayerNorm
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    FeedForward,
    QwenTimestepProjEmbeddings,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    from vllm_omni.diffusion.data import OmniDiffusionConfig


class UnifiedMSRoPE(nn.Module):
    """Unified three-axis rotary embedding used by the released decoder."""

    def __init__(
        self,
        theta: float = 10000.0,
        axes_dim: Sequence[int] = (16, 56, 56),
    ) -> None:
        super().__init__()
        if len(axes_dim) != 3 or any(dim % 2 for dim in axes_dim):
            raise ValueError(f"axes_dim must contain three even values, got {axes_dim}")
        self.theta = float(theta)
        self.axes_dim = tuple(int(dim) for dim in axes_dim)
        offsets = [0]
        inv_freqs = []
        for dim in self.axes_dim:
            inv_freqs.append(
                1.0
                / torch.pow(
                    torch.tensor(self.theta, dtype=torch.float32),
                    torch.arange(0, dim, 2, dtype=torch.float32) / dim,
                )
            )
            offsets.append(offsets[-1] + dim // 2)
        self.axis_offsets = tuple(offsets)
        self.register_buffer(
            "all_inv_freqs",
            torch.cat(inv_freqs),
            persistent=False,
        )

    def forward(self, position_ids_3d: torch.Tensor) -> torch.Tensor:
        position_ids_3d = position_ids_3d.float()
        pieces = []
        for axis_idx in range(3):
            start = self.axis_offsets[axis_idx]
            end = self.axis_offsets[axis_idx + 1]
            inv_freq = self.all_inv_freqs[start:end].to(position_ids_3d.device)
            phase = position_ids_3d[..., axis_idx, None] * inv_freq
            pieces.append(torch.polar(torch.ones_like(phase), phase))
        return torch.cat(pieces, dim=-1)


def _create_grid_positions(
    frame: int,
    height: int,
    width: int,
    *,
    cumulative_frame: int,
    scale_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """Create the centered, optionally scaled positions used by InternVL-U."""
    if frame <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"grid dimensions must be positive, got {(frame, height, width)}")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    h_neg_count = height - height // 2
    w_neg_count = width - width // 2
    h_indices = torch.cat(
        (
            torch.arange(-h_neg_count, 0, dtype=torch.float32, device=device),
            torch.arange(0, height // 2, dtype=torch.float32, device=device),
        )
    )
    w_indices = torch.cat(
        (
            torch.arange(-w_neg_count, 0, dtype=torch.float32, device=device),
            torch.arange(0, width // 2, dtype=torch.float32, device=device),
        )
    )

    if scale_factor != 1.0:
        target_h_neg = int(h_neg_count / scale_factor)
        target_h_pos = int((height // 2) / scale_factor)
        target_w_neg = int(w_neg_count / scale_factor)
        target_w_pos = int((width // 2) / scale_factor)

        h_min, h_max = -h_neg_count, height // 2 - 1
        if h_max != h_min:
            h_indices = -target_h_neg + (h_indices - h_min) * ((target_h_pos - 1) + target_h_neg) / (h_max - h_min)

        w_min, w_max = -w_neg_count, width // 2 - 1
        if w_max != w_min:
            w_indices = -target_w_neg + (w_indices - w_min) * ((target_w_pos - 1) + target_w_neg) / (w_max - w_min)

    f_indices = torch.arange(
        cumulative_frame,
        cumulative_frame + frame,
        dtype=torch.float32,
        device=device,
    )
    f_grid, h_grid, w_grid = torch.meshgrid(
        f_indices,
        h_indices,
        w_indices,
        indexing="ij",
    )
    return torch.stack(
        (f_grid.flatten(), h_grid.flatten(), w_grid.flatten()),
        dim=-1,
    )


def _true_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    mask = mask.bool().flatten()
    if mask.numel() == 0:
        return []
    padded = F.pad(mask.to(torch.int8), (1, 1))
    changes = padded[1:] - padded[:-1]
    starts = torch.nonzero(changes == 1, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(changes == -1, as_tuple=False).flatten().tolist()
    return list(zip(starts, ends, strict=True))


def build_position_ids_3d(
    latent_grids: Sequence[Sequence[int]],
    encoder_image_token_mask: torch.Tensor,
    image_fhw_cond: torch.Tensor,
    *,
    latent_scale: float = 1.0,
    vlm_image_scale: float = 0.25,
) -> torch.Tensor:
    """Build exact positions for generated latents, reference latents and VLM.

    ``latent_grids`` starts with the generated latent grid and may contain
    reference VAE grids.  They form the first contiguous image segment.  Each
    true span in ``encoder_image_token_mask`` forms one later VLM-image segment
    and must have a corresponding row in ``image_fhw_cond``.
    """
    device = encoder_image_token_mask.device
    if not latent_grids:
        raise ValueError("latent_grids must include the generated latent grid")

    latent_positions = []
    cumulative_frame = 0
    for grid in latent_grids:
        if len(grid) != 3:
            raise ValueError(f"latent grid must be [f,h,w], got {grid}")
        frame, height, width = (int(value) for value in grid)
        latent_positions.append(
            _create_grid_positions(
                frame,
                height,
                width,
                cumulative_frame=cumulative_frame,
                scale_factor=latent_scale,
                device=device,
            )
        )
        cumulative_frame += frame
    first_block = torch.cat(latent_positions, dim=0)

    text_image_spans = _true_spans(encoder_image_token_mask)
    if len(text_image_spans) != int(image_fhw_cond.shape[0]):
        raise ValueError(
            f"VLM image span/grid mismatch: found {len(text_image_spans)} spans but {image_fhw_cond.shape[0]} grids"
        )

    total_length = first_block.shape[0] + encoder_image_token_mask.numel()
    full_image_mask = torch.cat(
        (
            torch.ones(first_block.shape[0], dtype=torch.bool, device=device),
            encoder_image_token_mask.bool().flatten(),
        )
    )
    blocks = [first_block]
    for grid, (start, end) in zip(
        image_fhw_cond.tolist(),
        text_image_spans,
        strict=True,
    ):
        frame, height, width = (int(value) for value in grid)
        block = _create_grid_positions(
            frame,
            height,
            width,
            cumulative_frame=0,
            scale_factor=vlm_image_scale,
            device=device,
        )
        if block.shape[0] != end - start:
            raise ValueError(
                "VLM image grid does not match its token span: "
                f"grid={(frame, height, width)} has {block.shape[0]} tokens, "
                f"span has {end - start}"
            )
        blocks.append(block)

    spans = _true_spans(full_image_mask)
    if len(spans) != len(blocks):
        raise ValueError(f"image segment mismatch: found {len(spans)} spans for {len(blocks)} blocks")

    positions = torch.zeros(total_length, 3, dtype=torch.float32, device=device)
    current_text_pos = 0
    cursor = 0
    for (start, end), block in zip(spans, blocks, strict=True):
        if start > cursor:
            text_positions = torch.arange(
                current_text_pos,
                current_text_pos + start - cursor,
                dtype=torch.float32,
                device=device,
            )
            positions[cursor:start] = text_positions[:, None].expand(-1, 3)
            current_text_pos += start - cursor

        adjusted = block.clone()
        adjusted[:, 0] += current_text_pos
        positions[start:end] = adjusted
        current_text_pos = int(adjusted.max().item()) + 1
        cursor = end

    if cursor < total_length:
        text_positions = torch.arange(
            current_text_pos,
            current_text_pos + total_length - cursor,
            dtype=torch.float32,
            device=device,
        )
        positions[cursor:] = text_positions[:, None].expand(-1, 3)
    return positions


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Apply complex RoPE to [B,S,H,D] query/key tensors."""
    complex_states = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    rotated = complex_states * frequencies.unsqueeze(-2)
    return torch.view_as_real(rotated).flatten(-2).to(dtype=hidden_states.dtype)


class InternVLUJointAttention(nn.Module):
    """Separate image/context projections with doubled, gated queries."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if dim != num_heads * head_dim:
            raise ValueError(f"dim must equal num_heads * head_dim, got {dim}, {num_heads}, {head_dim}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        def column(name: str, output_size: int) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                dim,
                output_size,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.{name}",
            )

        # The Q checkpoint tensors interleave [query, gate] within every head.
        self.to_q = column("to_q", 2 * dim)
        self.to_k = column("to_k", dim)
        self.to_v = column("to_v", dim)
        self.add_q_proj = column("add_q_proj", 2 * dim)
        self.add_k_proj = column("add_k_proj", dim)
        self.add_v_proj = column("add_v_proj", dim)

        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )
        self.to_add_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_add_out",
        )
        self.attention = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=False,
            num_kv_heads=num_heads,
            prefix=prefix,
            role="self",
        )

    def _project_query(
        self,
        projection: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_gate = projection(hidden_states)
        local_heads = query_gate.shape[-1] // (2 * self.head_dim)
        query_gate = query_gate.unflatten(-1, (local_heads, 2 * self.head_dim))
        query, gate = query_gate.split(self.head_dim, dim=-1)
        return query, gate.flatten(-2)

    def _project_kv(
        self,
        projection: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        projected = projection(hidden_states)
        local_heads = projected.shape[-1] // self.head_dim
        return projected.unflatten(-1, (local_heads, self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_frequencies: torch.Tensor,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        encoder_mask = encoder_token_mask.unsqueeze(-1).to(dtype=dtype)
        image_mask = 1.0 - encoder_mask

        context_query, context_gate = self._project_query(self.add_q_proj, hidden_states)
        context_key = self._project_kv(self.add_k_proj, hidden_states)
        context_value = self.add_v_proj(hidden_states)

        image_query, image_gate = self._project_query(self.to_q, hidden_states)
        image_key = self._project_kv(self.to_k, hidden_states)
        image_value = self.to_v(hidden_states)

        context_query = self.norm_added_q(context_query)
        context_key = self.norm_added_k(context_key)
        image_query = self.norm_q(image_query)
        image_key = self.norm_k(image_key)

        stream_mask = encoder_mask.unsqueeze(-1)
        joint_query = context_query * stream_mask + image_query * (1.0 - stream_mask)
        joint_key = context_key * stream_mask + image_key * (1.0 - stream_mask)
        joint_value = (context_value * encoder_mask + image_value * image_mask).unflatten(
            -1, (joint_query.shape[-2], self.head_dim)
        )

        joint_query = apply_rotary_emb(joint_query, rotary_frequencies)
        joint_key = apply_rotary_emb(joint_key, rotary_frequencies)
        attended = self.attention(
            joint_query,
            joint_key,
            joint_value,
            AttentionMetadata(attn_mask=attention_mask),
        ).flatten(-2)

        context_output = self.to_add_out(attended)
        image_output = self.to_out(attended)
        output = context_output * encoder_mask + image_output * image_mask
        gate = context_gate * encoder_mask + image_gate * image_mask
        return output * torch.sigmoid(gate)


class InternVLUTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                dim,
                6 * dim,
                bias=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.img_mod.1",
            ),
        )
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                dim,
                6 * dim,
                bias=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.txt_mod.1",
            ),
        )
        self.img_norm1 = LayerNorm(dim, elementwise_affine=False)
        self.img_norm2 = LayerNorm(dim, elementwise_affine=False)
        self.txt_norm1 = LayerNorm(dim, elementwise_affine=False)
        self.txt_norm2 = LayerNorm(dim, elementwise_affine=False)
        self.attn = InternVLUJointAttention(
            dim,
            num_heads,
            head_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.img_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.img_mlp",
        )
        self.txt_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.txt_mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        rotary_frequencies: torch.Tensor,
        encoder_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        encoder_mask = encoder_token_mask.unsqueeze(-1).to(dtype=dtype)
        image_mask = 1.0 - encoder_mask

        image_mod = self.img_mod(temb)
        text_mod = self.txt_mod(temb)
        image_mod = image_mod[:, None].expand(-1, hidden_states.shape[1], -1)
        text_mod = text_mod[:, None].expand(-1, hidden_states.shape[1], -1)
        modulation = image_mod * image_mask + text_mod * encoder_mask
        mod1, mod2 = modulation.chunk(2, dim=-1)
        shift1, scale1, gate1 = mod1.chunk(3, dim=-1)
        shift2, scale2, gate2 = mod2.chunk(3, dim=-1)

        normed = self.img_norm1(hidden_states) * image_mask + self.txt_norm1(hidden_states) * encoder_mask
        modulated = normed * (1.0 + scale1) + shift1
        attention_output = self.attn(
            modulated,
            encoder_token_mask=encoder_token_mask,
            attention_mask=attention_mask,
            rotary_frequencies=rotary_frequencies,
        )
        hidden_states = hidden_states + gate1 * attention_output

        normed = self.img_norm2(hidden_states) * image_mask + self.txt_norm2(hidden_states) * encoder_mask
        modulated = normed * (1.0 + scale2) + shift2
        mlp_output = self.img_mlp(modulated) * image_mask + self.txt_mlp(modulated) * encoder_mask
        return hidden_states + gate2 * mlp_output


class InternVLUAdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            dim,
            2 * dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        embedding = self.linear(self.silu(temb).to(dtype=dtype)).float()
        scale, shift = embedding.chunk(2, dim=-1)
        normalized = F.layer_norm(
            hidden_states.float(),
            (self.dim,),
            weight=None,
            bias=None,
            eps=1e-6,
        )
        return (normalized * (1.0 + scale[:, None]) + shift[:, None]).to(dtype)


def _patchify(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    if latent.ndim != 3:
        raise ValueError(f"latent must be [C,H,W], got {tuple(latent.shape)}")
    channels, height, width = latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"latent spatial size {(height, width)} must be divisible by {patch_size}")
    return (
        latent.reshape(
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, channels * patch_size * patch_size)
    )


def _unpatchify(
    patches: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    return (
        patches.reshape(
            patches.shape[0],
            height // patch_size,
            width // patch_size,
            channels,
            patch_size,
            patch_size,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(patches.shape[0], channels, height, width)
    )


class InternVLUTransformer2DModel(nn.Module):
    """InternVL-U MMDiT with generated/reference/context token packing."""

    # Blocks take and return one packed sequence (image/reference/context
    # tokens mixed via masks), so cache-dit's single-stream pattern applies.
    # All three CFG branches run in one batched forward per step, so there is
    # no separate-CFG call to track.
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "transformer_blocks": ForwardPattern.Pattern_3,
        },
    )

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 12,
        joint_attention_dim: int = 4096,
        axes_dims_rope: Sequence[int] = (16, 56, 56),
        video_position_scale_factor: float = 1.0,
        vlm_cond_position_scale_factor: float = 0.25,
        max_sequence_length: int = 768,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Hyper-parameters arrive as explicit kwargs validated against the
        # checkpoint's generation_decoder config; od_config carries the shared
        # runtime configuration (parallelism, cache backends).
        self.od_config = od_config
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.head_dim = int(attention_head_dim)
        self.num_heads = int(num_attention_heads)
        self.inner_dim = self.head_dim * self.num_heads
        self.joint_attention_dim = int(joint_attention_dim)
        self.video_position_scale_factor = float(video_position_scale_factor)
        self.vlm_cond_position_scale_factor = float(vlm_cond_position_scale_factor)
        self.max_sequence_length = int(max_sequence_length)
        if sum(int(value) for value in axes_dims_rope) != self.head_dim:
            raise ValueError(
                f"sum(axes_dims_rope) must equal attention_head_dim, got {axes_dims_rope} and {self.head_dim}"
            )

        self.config = SimpleNamespace(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            attention_head_dim=self.head_dim,
            num_attention_heads=self.num_heads,
            joint_attention_dim=self.joint_attention_dim,
            axes_dims_rope=tuple(axes_dims_rope),
            video_position_scale_factor=self.video_position_scale_factor,
            vlm_cond_position_scale_factor=self.vlm_cond_position_scale_factor,
            max_sequence_length=self.max_sequence_length,
        )
        # Learned pad embedding for conditioning shorter than
        # max_sequence_length (checkpoint key ``encoder_padding_token``).
        self.encoder_padding_token = nn.Parameter(torch.zeros(self.joint_attention_dim))
        self.pos_embed = UnifiedMSRoPE(theta=10000.0, axes_dim=axes_dims_rope)
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            quant_config=quant_config,
        )
        self.txt_norm = RMSNorm(self.joint_attention_dim, eps=1e-6)
        self.img_in = ReplicatedLinear(
            self.in_channels,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.img_in",
        )
        self.txt_in = ReplicatedLinear(
            self.joint_attention_dim,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.txt_in",
        )
        self.transformer_blocks = nn.ModuleList(
            [
                InternVLUTransformerBlock(
                    self.inner_dim,
                    self.num_heads,
                    self.head_dim,
                    quant_config=quant_config,
                    prefix=f"{prefix}.transformer_blocks.{index}",
                )
                for index in range(self.num_layers)
            ]
        )
        self.norm_out = InternVLUAdaLayerNormContinuous(
            self.inner_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.norm_out",
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_out",
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_forward_input(
        self,
        encoder_hidden_states: Sequence[torch.Tensor],
        encoder_image_token_mask: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad variable-length conditioning branches to one batched tensor.

        Sequences are padded with the learned ``encoder_padding_token`` up to
        at least ``max_sequence_length`` and are never truncated.
        """
        if len(encoder_hidden_states) != len(encoder_image_token_mask):
            raise ValueError("hidden-state and image-mask branch counts differ")
        if not encoder_hidden_states:
            raise ValueError("at least one conditioning branch is required")
        max_length = max(
            self.max_sequence_length,
            max(int(states.shape[0]) for states in encoder_hidden_states),
        )
        batch_size = len(encoder_hidden_states)
        hidden_size = self.encoder_padding_token.shape[0]
        padded = self.encoder_padding_token[None, None].expand(batch_size, max_length, hidden_size).clone()
        attention_mask = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.bool,
            device=self.encoder_padding_token.device,
        )
        image_mask = torch.zeros_like(attention_mask)
        for index, (states, mask) in enumerate(zip(encoder_hidden_states, encoder_image_token_mask, strict=True)):
            if states.ndim != 2 or states.shape[-1] != hidden_size:
                raise ValueError(f"conditioning must be [T,{hidden_size}], got {tuple(states.shape)}")
            if mask.ndim != 1 or mask.shape[0] != states.shape[0]:
                raise ValueError(f"image mask must be [T] aligned with conditioning, got {tuple(mask.shape)}")
            length = states.shape[0]
            padded[index, :length] = states.to(
                device=padded.device,
                dtype=padded.dtype,
            )
            attention_mask[index, :length] = True
            image_mask[index, :length] = mask.to(
                device=image_mask.device,
                dtype=torch.bool,
            )
        return padded, attention_mask, image_mask

    def _pack_inputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_image_token_mask: torch.Tensor,
        reference_latents: Sequence[Sequence[torch.Tensor]],
        image_fhw_cond: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        batch_size, channels, height, width = hidden_states.shape
        if channels * self.patch_size**2 != self.in_channels:
            raise ValueError(
                f"latent channel/patch contract mismatch: {channels} * {self.patch_size}^2 != {self.in_channels}"
            )
        if not (
            len(reference_latents) == len(image_fhw_cond) == batch_size and encoder_hidden_states.shape[0] == batch_size
        ):
            raise ValueError("branch batch sizes do not match")
        max_reference_count = max(len(branch) for branch in reference_latents)
        reference_limit = int(
            min(height // self.patch_size, width // self.patch_size) // 2 / self.video_position_scale_factor
        )
        if max_reference_count > 0 and max_reference_count >= reference_limit:
            raise ValueError(
                "InternVL-U reference-image count must be less than "
                f"{reference_limit} for latent size {(height, width)}, "
                f"got {max_reference_count}"
            )

        projected_text = self.txt_in(self.txt_norm(encoder_hidden_states))
        rows: list[torch.Tensor] = []
        row_encoder_masks: list[torch.Tensor] = []
        row_frequencies: list[torch.Tensor] = []
        generated_token_count = (height // self.patch_size) * (width // self.patch_size)

        for branch_idx in range(batch_size):
            generated_patches = _patchify(hidden_states[branch_idx], self.patch_size)
            latent_parts = [generated_patches]
            latent_grids: list[tuple[int, int, int]] = [(1, height // self.patch_size, width // self.patch_size)]
            for reference in reference_latents[branch_idx]:
                reference = reference.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                latent_parts.append(_patchify(reference, self.patch_size))
                latent_grids.append(
                    (
                        1,
                        reference.shape[-2] // self.patch_size,
                        reference.shape[-1] // self.patch_size,
                    )
                )
            image_tokens = self.img_in(torch.cat(latent_parts, dim=0))

            valid_text_length = int(encoder_attention_mask[branch_idx].sum().item())
            if valid_text_length <= 0:
                raise ValueError("each branch must contain at least one valid VLM token")
            text_tokens = projected_text[branch_idx, :valid_text_length]
            text_image_mask = encoder_image_token_mask[branch_idx, :valid_text_length].bool()
            branch_fhw = image_fhw_cond[branch_idx].to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            positions = build_position_ids_3d(
                latent_grids,
                text_image_mask,
                branch_fhw,
                latent_scale=self.video_position_scale_factor,
                vlm_image_scale=self.vlm_cond_position_scale_factor,
            )
            row = torch.cat((image_tokens, text_tokens), dim=0)
            if positions.shape[0] != row.shape[0]:
                raise AssertionError("position/token packing length mismatch")
            rows.append(row)
            row_encoder_masks.append(
                torch.cat(
                    (
                        torch.zeros(
                            image_tokens.shape[0],
                            dtype=torch.bool,
                            device=hidden_states.device,
                        ),
                        torch.ones(
                            text_tokens.shape[0],
                            dtype=torch.bool,
                            device=hidden_states.device,
                        ),
                    )
                )
            )
            row_frequencies.append(self.pos_embed(positions))

        max_length = max(row.shape[0] for row in rows)
        packed = hidden_states.new_zeros(batch_size, max_length, self.inner_dim)
        valid_mask = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        encoder_mask = torch.zeros_like(valid_mask)
        frequencies = torch.ones(
            batch_size,
            max_length,
            self.head_dim // 2,
            dtype=torch.complex64,
            device=hidden_states.device,
        )
        for index, row in enumerate(rows):
            length = row.shape[0]
            packed[index, :length] = row
            valid_mask[index, :length] = True
            encoder_mask[index, :length] = row_encoder_masks[index]
            frequencies[index, :length] = row_frequencies[index]
        return packed, valid_mask, encoder_mask, frequencies, generated_token_count

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_image_token_mask: torch.Tensor,
        reference_latents: Sequence[Sequence[torch.Tensor]] | None = None,
        image_fhw_cond: Sequence[torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        batch_size = hidden_states.shape[0]
        reference_latents = reference_latents or [[] for _ in range(batch_size)]
        image_fhw_cond = image_fhw_cond or [
            torch.empty(0, 3, dtype=torch.long, device=hidden_states.device) for _ in range(batch_size)
        ]
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)
        (
            packed,
            valid_mask,
            encoder_mask,
            frequencies,
            generated_token_count,
        ) = self._pack_inputs(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_image_token_mask,
            reference_latents,
            image_fhw_cond,
        )
        for block in self.transformer_blocks:
            packed = block(
                packed,
                temb,
                attention_mask=valid_mask,
                rotary_frequencies=frequencies,
                encoder_token_mask=encoder_mask,
            )

        generated = packed[:, :generated_token_count]
        generated = self.norm_out(generated, temb)
        output_patches = self.proj_out(generated)
        output = _unpatchify(
            output_patches,
            channels=self.out_channels,
            height=hidden_states.shape[-2],
            width=hidden_states.shape[-1],
            patch_size=self.patch_size,
        )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


__all__ = [
    "InternVLUTransformer2DModel",
    "InternVLUTransformerBlock",
    "UnifiedMSRoPE",
    "apply_rotary_emb",
    "build_position_ids_3d",
]
