# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Helios (https://github.com/BestWishYsh/Helios)

import math
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    return F.avg_pool3d(x, kernel_size, stride=kernel_size)


def apply_rotary_emb_helios(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply Helios-style rotary embeddings.

    freqs_cis contains [cos_t, cos_y, cos_x, sin_t, sin_y, sin_x] concatenated
    along the last dimension, with shape [B, seq, D*2] where D = DT+DY+DX.
    hidden_states has shape [B, seq, H, head_dim].
    """
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


class DistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor parallel ranks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()

        input_dtype = x.dtype
        x_float = x.float()
        local_sum_sq = (x_float**2).sum(dim=-1, keepdim=True)
        local_count = x.shape[-1]

        if tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            tensor_model_parallel_all_reduce(global_sum_sq)
            global_count = local_count * tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = local_count

        rms = torch.sqrt(global_sum_sq / global_count + self.eps)

        output = (x_float / rms) * self.weight.float()
        return output.to(input_dtype)


class ColumnParallelGELU(nn.Module):
    """Column parallel linear with GELU activation."""

    def __init__(self, dim_in: int, dim_out: int, *, approximate: str = "tanh", bias: bool = True):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class HeliosFeedForward(nn.Module):
    """TP-enabled FeedForward network for Helios."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dim_out: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dim_out = dim_out or dim

        self.net_0 = ColumnParallelGELU(dim, inner_dim, approximate="tanh", bias=bias)
        self.net_1 = nn.Identity()
        self.net_2 = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=bias,
            input_is_parallel=True,
            return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_1(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class HeliosRotaryPosEmbed(nn.Module):
    """Helios-style 3D rotary position embeddings using explicit frame indices."""

    def __init__(self, rope_dim: tuple[int, int, int], theta: float):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.register_buffer("freqs_base_t", self._get_freqs_base(self.DT), persistent=False)
        self.register_buffer("freqs_base_y", self._get_freqs_base(self.DY), persistent=False)
        self.register_buffer("freqs_base_x", self._get_freqs_base(self.DX), persistent=False)

    def _get_freqs_base(self, dim):
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

    @torch.no_grad()
    def get_frequency_batched(self, freqs_base, pos):
        freqs = torch.einsum("d,bthw->dbthw", freqs_base, pos)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    @lru_cache(maxsize=32)
    def _get_spatial_meshgrid(self, height, width, device_str):
        device = torch.device(device_str)
        grid_y_coords = torch.arange(height, device=device, dtype=torch.float32)
        grid_x_coords = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y_coords, grid_x_coords, indexing="ij")
        return grid_y, grid_x

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        batch_size = frame_indices.shape[0]
        num_frames = frame_indices.shape[1]

        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        grid_y, grid_x = self._get_spatial_meshgrid(height, width, str(device))

        grid_t = frame_indices[:, :, None, None].expand(batch_size, num_frames, height, width)
        grid_y_batch = grid_y[None, None, :, :].expand(batch_size, num_frames, -1, -1)
        grid_x_batch = grid_x[None, None, :, :].expand(batch_size, num_frames, -1, -1)

        freqs_cos_t, freqs_sin_t = self.get_frequency_batched(self.freqs_base_t, grid_t)
        freqs_cos_y, freqs_sin_y = self.get_frequency_batched(self.freqs_base_y, grid_y_batch)
        freqs_cos_x, freqs_sin_x = self.get_frequency_batched(self.freqs_base_x, grid_x_batch)

        result = torch.cat(
            [freqs_cos_t, freqs_cos_y, freqs_cos_x, freqs_sin_t, freqs_sin_y, freqs_sin_x],
            dim=0,
        )

        return result.permute(1, 0, 2, 3, 4)


class HeliosTimeTextEmbedding(nn.Module):
    """Combined time and text condition embeddings for Helios."""

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        is_return_encoder_hidden_states: bool = True,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if encoder_hidden_states is not None and is_return_encoder_hidden_states:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class HeliosOutputNorm(nn.Module):
    """Output normalization that extracts only original_context_length tokens."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, original_context_length: int):
        temb = temb[:, -original_context_length:, :]
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2).to(hidden_states.device), scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = (self.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        return hidden_states


class HeliosSelfAttention(nn.Module):
    """Optimized self-attention for Helios with history amplification support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=True,
        )

        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        self.tp_inner_dim = self.num_heads * head_dim

        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps)

        self.to_out = RowParallelLinear(
            self.inner_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
        )
        self.dropout = nn.Dropout(dropout)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

        self.is_amplify_history = is_amplify_history
        if is_amplify_history:
            tp_size = get_tensor_model_parallel_world_size()
            local_heads = num_heads // tp_size
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(1))
            elif history_scale_mode == "per_head":
                self.history_key_scale = nn.Parameter(torch.ones(local_heads))
            else:
                raise ValueError(f"Unknown history_scale_mode: {history_scale_mode}")
            self.history_scale_mode = history_scale_mode
            self.max_scale = 10.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        original_context_length: int | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.to_qkv(hidden_states)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.unflatten(2, (self.num_heads, self.head_dim))
        key = key.unflatten(2, (self.num_kv_heads, self.head_dim))
        value = value.unflatten(2, (self.num_kv_heads, self.head_dim))

        if rotary_emb is not None:
            query = apply_rotary_emb_helios(query, rotary_emb)
            key = apply_rotary_emb_helios(key, rotary_emb)

        if self.is_amplify_history and original_context_length is not None:
            history_seq_len = hidden_states.shape[1] - original_context_length
            if history_seq_len > 0:
                scale_key = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
                if self.history_scale_mode == "per_head":
                    scale_key = scale_key.view(1, 1, -1, 1)
                key = torch.cat(
                    [key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]],
                    dim=1,
                )

        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = self.to_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class HeliosCrossAttention(nn.Module):
    """Optimized cross-attention for Helios."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.to_q = ColumnParallelLinear(
            dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
        )
        self.to_k = ColumnParallelLinear(
            dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
        )
        self.to_v = ColumnParallelLinear(
            dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
        )

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp_size
        self.tp_inner_dim = self.num_heads * head_dim

        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps)

        self.to_out = RowParallelLinear(
            self.inner_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
        )
        self.dropout = nn.Dropout(dropout)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        query = self.norm_q(query)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        key = self.norm_k(key)

        query = query.unflatten(2, (self.num_heads, self.head_dim))
        key = key.unflatten(2, (self.num_heads, self.head_dim))
        value = value.unflatten(2, (self.num_heads, self.head_dim))

        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = self.to_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class HeliosTransformerBlock(nn.Module):
    """Transformer block with guidance cross-attention and history support."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        cross_attn_norm: bool = False,
        guidance_cross_attn: bool = False,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
    ):
        super().__init__()

        head_dim = dim // num_heads

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = HeliosSelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
        )

        # 2. Cross-attention
        self.attn2 = HeliosCrossAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = HeliosFeedForward(dim=dim, inner_dim=ffn_dim, dim_out=dim)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # 4. Guidance cross-attention flag
        self.guidance_cross_attn = guidance_cross_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        original_context_length: int | None = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, rotary_emb, original_context_length)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention (with optional guidance: only current chunk attends to text)
        if self.guidance_cross_attn and original_context_length is not None:
            history_seq_len = hidden_states.shape[1] - original_context_length

            history_hidden_states, current_hidden_states = (
                hidden_states[:, :history_seq_len],
                hidden_states[:, history_seq_len:],
            )
            norm_hidden_states = self.norm2(current_hidden_states.float()).type_as(current_hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            current_hidden_states = current_hidden_states + attn_output
            hidden_states = torch.cat([history_hidden_states, current_hidden_states], dim=1)
        else:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class HeliosTransformer3DModel(nn.Module):
    """Optimized Helios Transformer model for video generation using vLLM layers.

    Helios extends the Wan2.2 architecture with multi-term memory patches,
    guidance cross-attention, and chunked video generation support.
    """

    _repeated_blocks = ["HeliosTransformerBlock"]
    _layerwise_offload_blocks_attr = "blocks"
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    _sp_plan = {
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True, auto_pad=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True, auto_pad=True),
        },
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        rope_dim: tuple[int, ...] = (44, 42, 42),
        rope_theta: float = 10000.0,
        guidance_cross_attn: bool = True,
        zero_history_timestep: bool = True,
        has_multi_term_memory_patch: bool = True,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "text_dim": text_dim,
                "freq_dim": freq_dim,
                "ffn_dim": ffn_dim,
                "num_layers": num_layers,
                "cross_attn_norm": cross_attn_norm,
                "qk_norm": qk_norm,
                "eps": eps,
                "added_kv_proj_dim": added_kv_proj_dim,
                "rope_dim": rope_dim,
                "rope_theta": rope_theta,
                "guidance_cross_attn": guidance_cross_attn,
                "zero_history_timestep": zero_history_timestep,
                "has_multi_term_memory_patch": has_multi_term_memory_patch,
                "is_amplify_history": is_amplify_history,
                "history_scale_mode": history_scale_mode,
            },
        )()

        self.inner_dim = inner_dim
        self.zero_history_timestep = zero_history_timestep
        self.has_multi_term_memory_patch = has_multi_term_memory_patch

        # 1. Patch & position embedding
        self.rope = HeliosRotaryPosEmbed(rope_dim=rope_dim, theta=rope_theta)
        self.patch_embedding = Conv3dLayer(
            in_channels=in_channels,
            out_channels=inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Multi-term memory patches
        if has_multi_term_memory_patch:
            self.patch_short = Conv3dLayer(
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
            )
            self.patch_mid = Conv3dLayer(
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=(2, 4, 4),
                stride=(2, 4, 4),
            )
            self.patch_long = Conv3dLayer(
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=(4, 8, 8),
                stride=(4, 8, 8),
            )

        # 3. Condition embeddings
        self.condition_embedder = HeliosTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                HeliosTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    eps,
                    cross_attn_norm,
                    guidance_cross_attn=guidance_cross_attn,
                    is_amplify_history=is_amplify_history,
                    history_scale_mode=history_scale_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = HeliosOutputNorm(inner_dim, eps)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        indices_hidden_states: torch.Tensor | None = None,
        indices_latents_history_short: torch.Tensor | None = None,
        indices_latents_history_mid: torch.Tensor | None = None,
        indices_latents_history_long: torch.Tensor | None = None,
        latents_history_short: torch.Tensor | None = None,
        latents_history_mid: torch.Tensor | None = None,
        latents_history_long: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        # 1. Process noisy latents
        hidden_states = self.patch_embedding(hidden_states)
        _, _, post_patch_num_frames, post_patch_height, post_patch_width = hidden_states.shape

        if indices_hidden_states is None:
            indices_hidden_states = torch.arange(0, post_patch_num_frames).unsqueeze(0).expand(batch_size, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        rotary_emb = self.rope(
            frame_indices=indices_hidden_states,
            height=post_patch_height,
            width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = rotary_emb.flatten(2).transpose(1, 2)
        original_context_length = hidden_states.shape[1]

        # 2. Process short history latents
        if latents_history_short is not None and indices_latents_history_short is not None:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.patch_short(latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rotary_emb_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rotary_emb_history_short = rotary_emb_history_short.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_short, rotary_emb], dim=1)

        # 3. Process mid history latents
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.patch_mid(latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rotary_emb_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rotary_emb_history_mid = pad_for_3d_conv(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = center_down_sample_3d(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = rotary_emb_history_mid.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_mid, rotary_emb], dim=1)

        # 4. Process long history latents
        if latents_history_long is not None and indices_latents_history_long is not None:
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.patch_long(latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rotary_emb_history_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rotary_emb_history_long = pad_for_3d_conv(rotary_emb_history_long, (4, 4, 4))
            rotary_emb_history_long = center_down_sample_3d(rotary_emb_history_long, (4, 4, 4))
            rotary_emb_history_long = rotary_emb_history_long.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_long, rotary_emb], dim=1)

        history_context_length = hidden_states.shape[1] - original_context_length

        # 5. Compute timestep embeddings
        if indices_hidden_states is not None and self.zero_history_timestep:
            timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if indices_hidden_states is not None and not self.zero_history_timestep:
            main_repeat_size = hidden_states.shape[1]
        else:
            main_repeat_size = original_context_length
        temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
        timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

        if indices_hidden_states is not None and self.zero_history_timestep:
            temb = torch.cat([temb_t0, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # 6. Transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                original_context_length,
            )

        # 7. Output normalization
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states = self.proj_out(hidden_states)

        # 8. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with QKV fusion, FFN remapping, and TP norm sharding."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            # Handle QKV fusion for self-attention
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # FFN remapping: net.0 -> net_0, net.2 -> net_2
                if ".ffn.net.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.0.", ".ffn.net_0.")
                elif ".ffn.net.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.2.", ".ffn.net_2.")

                # Output projection: to_out.0 -> to_out
                if ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")

                if lookup_name not in params_dict:
                    logger.warning(f"Skipping weight {original_name} -> {lookup_name}")
                    continue

                param = params_dict[lookup_name]

                # TP sharding for RMSNorm weights
                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [
                        ".attn1.norm_q.",
                        ".attn1.norm_k.",
                        ".attn2.norm_q.",
                        ".attn2.norm_k.",
                    ]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                # TP sharding for history_key_scale (per_head mode)
                if tp_size > 1 and ".history_key_scale" in lookup_name and loaded_weight.dim() == 1:
                    shard_size = loaded_weight.shape[0] // tp_size
                    if shard_size > 0:
                        loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
