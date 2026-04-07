# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from Diffusers implementation:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos.py


from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as DistributedRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def apply_rotary_emb_cosmos(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensors.

    Args:
        hidden_states: Input tensor of shape [B, S, H, D]
        freqs_cos: Cosine frequencies
        freqs_sin: Sine frequencies

    Returns:
        Tensor with rotary embeddings applied
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class CosmosRotaryPosEmbed(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for 3D video data with NTK-aware scaling.
    """

    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
    ):
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        fps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)

        dim_h_range = (
            torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        )
        dim_w_range = (
            torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        )
        dim_t_range = (
            torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        )

        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        if fps is None:
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)

        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
    """
    Learnable 3D positional embeddings for temporal, height, and width dimensions.
    """

    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]

        emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)

        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))

        return (emb / norm).type_as(hidden_states)


class CosmosPatchEmbed(nn.Module):
    """
    Patch embedding for 3D video inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            out_channels,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size,
            num_channels,
            num_frames // p_t,
            p_t,
            height // p_h,
            p_h,
            width // p_w,
            p_w,
        )

        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)
        hidden_states = hidden_states.type_as(self.proj.weight)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosTimestepEmbedding(nn.Module):
    """
    MLP for processing timestep embeddings.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps.to(self.linear_1.weight.dtype))
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(nn.Module):
    """
    Complete timestep conditioning module.
    """

    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()

        self.time_proj = Timesteps(
            embedding_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) without gate.
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()

        self.embedding_dim = in_features
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()

        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb[..., : 2 * self.embedding_dim]

        shift, scale = embedded_timestep.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class CosmosAdaLayerNormZero(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization (AdaLN-Zero).
    """

    def __init__(self, in_features: int, hidden_features: int | None = None):
        super().__init__()

        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()

        if hidden_features is None:
            self.linear_1 = nn.Identity()
        else:
            self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)

        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb

        shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states, gate


class CosmosAttention(nn.Module):
    """
    Cosmos Attention module wrapping vLLM parallel layers.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 16,
        kv_heads: int | None = None,
        dim_head: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_dim: int = None,
    ):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        from vllm_omni.diffusion.attention.layer import Attention

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.is_cross_attention = True
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = heads // tp_size
        self.tp_inner_dim = self.num_heads_per_partition * dim_head

        self.norm_q = DistributedRMSNorm(self.dim_head, eps=eps)
        self.norm_k = DistributedRMSNorm(self.dim_head, eps=eps)

        if self.is_cross_attention:
            self.to_q = ColumnParallelLinear(
                query_dim,
                self.inner_dim,
                bias=bias,
                gather_output=False,
                return_bias=False,
            )
            self.to_k = ColumnParallelLinear(
                self.cross_attention_dim,
                self.inner_kv_dim,
                bias=bias,
                gather_output=False,
                return_bias=False,
            )
            self.to_v = ColumnParallelLinear(
                self.cross_attention_dim,
                self.inner_kv_dim,
                bias=bias,
                gather_output=False,
                return_bias=False,
            )
        else:
            self._qkv_proj = QKVParallelLinear(
                hidden_size=query_dim,
                head_size=dim_head,
                total_num_heads=heads,
                bias=bias,
            )
            self.to_q = self._qkv_proj
            self.to_k = self._qkv_proj
            self.to_v = self._qkv_proj

        self.to_out = RowParallelLinear(
            self.inner_dim,
            self.out_dim,
            bias=bias,
            input_is_parallel=True,
            return_bias=False,
        )
        self.dropout = nn.Dropout(dropout)

        self._attn = Attention(
            num_heads=self.num_heads_per_partition,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
            num_kv_heads=self.num_heads_per_partition,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.heads, -1)).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        hidden_states = self._attn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


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


class CosmosFeedForward(nn.Module):
    """
    Cosmos FeedForward module wrapping vLLM parallel layers.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = ColumnParallelGELU(dim, inner_dim, approximate="tanh", bias=bias)
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            RowParallelLinear(
                inner_dim,
                dim,
                bias=bias,
                input_is_parallel=True,
                return_bias=False,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CosmosTransformerBlock(nn.Module):
    """
    Cosmos Transformer Block with self-attention, cross-attention, and feedforward.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int = 1024,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        od_config: OmniDiffusionConfig | None = None,
        block_index: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.block_index = block_index
        self.od_config = od_config

        self.norm1 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)

        self.attn1 = CosmosAttention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=0.0,
            bias=False,
            eps=1e-5,
        )

        self.norm2 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)

        self.attn2 = CosmosAttention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=0.0,
            bias=False,
            eps=1e-5,
        )

        self.norm3 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)

        self.ff = CosmosFeedForward(
            dim=hidden_size,
            mult=mlp_ratio,
            dropout=0.0,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_timestep = embedded_timestep.type_as(hidden_states)
        image_rotary_emb = (freqs_cos, freqs_sin)

        norm_hidden, gate1 = self.norm1(hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate1 * attn_output

        norm_hidden, gate2 = self.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(norm_hidden, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + gate2 * attn_output

        norm_hidden, gate3 = self.norm3(hidden_states, embedded_timestep, temb)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + gate3 * ff_output

        return hidden_states


class CosmosTransformer3DModel(nn.Module):
    """
    Cosmos Predict 2.5 3D Video Transformer.
    """

    _repeated_blocks = ["CosmosTransformerBlock"]
    _layerwise_offload_blocks_attr = "blocks"

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 4096,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: str | None = "learnable",
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int | None = None,
        encoder_hidden_states_channels: int = 1024,
        **kwargs,
    ):
        super().__init__()

        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config else None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.hidden_size = num_attention_heads * attention_head_dim
        self.concat_padding_mask = concat_padding_mask

        self.patch_embed = CosmosPatchEmbed(
            in_channels=18,
            out_channels=self.hidden_size,
            patch_size=patch_size,
            bias=False,
        )

        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim,
            max_size=max_size,
            patch_size=patch_size,
            base_fps=24,
            rope_scale=rope_scale,
        )

        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=self.hidden_size,
                max_size=max_size,
                patch_size=patch_size,
                eps=1e-6,
            )
        else:
            self.learnable_pos_embed = None

        self.time_embed = CosmosEmbedding(
            embedding_dim=self.hidden_size,
            condition_dim=self.hidden_size,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    hidden_size=self.hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=encoder_hidden_states_channels,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    od_config=od_config,
                    block_index=i,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = CosmosAdaLayerNorm(self.hidden_size, adaln_lora_dim)

        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(self.hidden_size, patch_volume * out_channels, bias=False)

        if use_crossattn_projection and crossattn_proj_in_channels is not None:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, encoder_hidden_states_channels, bias=True),
                nn.GELU(),
            )
        else:
            self.crossattn_proj = None

        logger.info(
            f"Initialized CosmosTransformer3DModel: "
            f"{num_layers} layers, {num_attention_heads} heads, "
            f"{attention_head_dim} head_dim, hidden_size={self.hidden_size}"
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        latents = hidden_states

        if condition_mask is not None:
            latents = torch.cat([latents, condition_mask], dim=1)

        if padding_mask is not None:
            from torchvision import transforms

            batch_size, num_channels, num_frames, height, width = latents.shape
            padding_mask_resized = transforms.functional.resize(
                padding_mask,
                list(latents.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [latents, padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)],
                dim=1,
            )
        else:
            hidden_states = latents

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        latent_shape = hidden_states.shape

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)

        if self.learnable_pos_embed is not None:
            learnable_pos = self.learnable_pos_embed(
                torch.zeros(latent_shape, device=hidden_states.device, dtype=hidden_states.dtype)
            )
            hidden_states = hidden_states + learnable_pos

        freqs_cos, freqs_sin = self.rope(
            torch.zeros(latent_shape, device=hidden_states.device),
            fps=fps,
        )

        p_t, p_h, p_w = self.od_config.tf_model_config.params["patch_size"]
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)

            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )
        else:
            raise ValueError(f"Expected timestep to have shape [B, 1, T, 1, 1] or [T], but got {timestep.shape}")

        if self.crossattn_proj is not None:
            encoder_hidden_states = self.crossattn_proj(encoder_hidden_states)

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states cannot be None")

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                freqs_cos,
                freqs_sin,
                temb,
            )

        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self._unpatchify(hidden_states, latent_shape)

        return hidden_states

    def _unpatchify(
        self,
        hidden_states: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = latent_shape
        p_t, p_h, p_w = self.patch_embed.patch_size

        num_patches_t = num_frames // p_t
        num_patches_h = height // p_h
        num_patches_w = width // p_w

        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (num_patches_t, num_patches_h, num_patches_w))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = []

        weight_name_remapping = {}

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            name = weight_name_remapping.get(name, name)
            original_name = name
            lookup_name = name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")

                if lookup_name not in params_dict:
                    logger.warning(f"Skipping weight {original_name} -> {lookup_name}")
                    continue

                param = params_dict[lookup_name]

                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [
                        ".attn1.norm_q.",
                        ".attn1.norm_k.",
                        ".attn2.norm_q.",
                        ".attn2.norm_k.",
                        ".attn2.norm_added_k.",
                    ]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
