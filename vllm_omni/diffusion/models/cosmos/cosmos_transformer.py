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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


class CosmosPatchEmbed(nn.Module):
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
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()

        self.time_proj = Timesteps(
            embedding_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = RMSNorm(embedding_dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.embedding_dim = in_features

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
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
    ) -> torch.Tensor:
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


class CosmosSelfAttention(nn.Module):
    """
    Optimized self-attention module using vLLM layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # QK normalization using vLLM's RMSNorm
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        # Fused QKV projection using vLLM's optimized layer
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=bias,
            disable_tp=True,
        )

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=bias),
                nn.Dropout(dropout),
            ]
        )

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Fused QKV projection
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, -1)).transpose(1, 2)

        # QK normalization
        q_shape, k_shape = query.shape, key.shape
        query = self.norm_q(query.reshape(-1, self.head_dim)).view(q_shape)
        key = self.norm_k(key.reshape(-1, self.head_dim)).view(k_shape)

        # Apply RoPE
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # Attention
        hidden_states = self.attn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class CosmosCrossAttention(nn.Module):
    """
    Optimized cross-attention module using vLLM layers.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_heads: int = 16,
        kv_heads: int | None = None,
        head_dim: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_dim: int | None = None,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else head_dim * num_heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else head_dim * kv_heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cross_attention_dim = cross_attention_dim

        # QK normalization
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        # Query projection
        self.to_q = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)

        # Separate K and V projections for cross-attention
        self.to_k = ReplicatedLinear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = ReplicatedLinear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, self.out_dim, bias=bias),
                nn.Dropout(dropout),
            ]
        )

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        # Query projection
        query, _ = self.to_q(hidden_states)

        # KV projection from encoder
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)

        query = query.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, -1)).transpose(1, 2)

        # QK normalization
        q_shape, k_shape = query.shape, key.shape
        query = self.norm_q(query.reshape(-1, self.head_dim)).view(q_shape)
        key = self.norm_k(key.reshape(-1, self.head_dim)).view(k_shape)

        # Apply RoPE
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # Attention
        hidden_states = self.attn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class CosmosTransformerBlock(nn.Module):
    """
    Cosmos Transformer Block with self-attention, cross-attention, and feedforward.
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        out_bias: bool = False,
    ):
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn1 = CosmosSelfAttention(
            dim=hidden_size,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=0.0,
            bias=False,
            eps=1e-5,
        )

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn2 = CosmosCrossAttention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=0.0,
            bias=False,
            eps=1e-5,
        )

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu", bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        temb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_timestep = embedded_timestep.type_as(hidden_states)

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # Self Attention
        norm_hidden, gate = self.norm1(hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output

        # Cross Attention
        norm_hidden, gate = self.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(norm_hidden, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + gate * attn_output

        # Feed Forward
        norm_hidden, gate = self.norm3(hidden_states, embedded_timestep, temb)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + gate * ff_output

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
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

        # Apply sequence scaling in temporal dimension
        if fps is None:
            # Images
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            # Videos
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
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


class CosmosTransformer3DModel(nn.Module):
    r"""
    A Transformer model for video-like data used in [Cosmos](https://github.com/NVIDIA/Cosmos).

    This is an optimized version of the diffusers CosmosTransformer3DModel that uses
    vLLM's efficient QKVParallelLinear and RMSNorm implementations.

    Args:
        patch_size: 3D patch dimensions for patchifying the input latent tensors (t_patch, h_patch, w_patch).
        num_attention_heads: The number of heads to use for multi-head attention.
        attention_head_dim: The number of channels in each attention head.
        in_channels: The number of channels in the input.
        out_channels: The number of channels in the output.
        num_layers: The number of layers of transformer blocks to use.
        mlp_ratio: The ratio of the hidden layer size to the input size in the feedforward network.
        text_embed_dim: Input dimension of text embeddings from the text encoder.
        adaln_lora_dim: The hidden dimension of the Adaptive LayerNorm LoRA layer.
        max_size: The maximum size of the input latent tensors in the temporal, height, and width dimensions.
        rope_scale: The scaling factor to use for RoPE in the temporal, height, and width dimensions.
        concat_padding_mask: Whether to concatenate the padding mask to the input latent tensors.
        extra_pos_embed_type: The type of extra positional embeddings to use. Can be one of `None` or `learnable`.
    """

    _repeated_blocks = ["CosmosTransformerBlock"]
    _layerwise_offload_blocks_attr = "transformer_blocks"
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    def __init__(
        self,
        *,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: str | None = "learnable",
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        encoder_hidden_states_channels: int = 1024,
        **kwargs,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # Store config for compatibility
        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "mlp_ratio": mlp_ratio,
                "text_embed_dim": text_embed_dim,
                "adaln_lora_dim": adaln_lora_dim,
                "max_size": max_size,
                "rope_scale": rope_scale,
                "concat_padding_mask": concat_padding_mask,
                "extra_pos_embed_type": extra_pos_embed_type,
                "use_crossattn_projection": use_crossattn_projection,
                "crossattn_proj_in_channels": crossattn_proj_in_channels,
                "encoder_hidden_states_channels": encoder_hidden_states_channels,
            },
        )()

        # Patch Embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(
            in_channels=patch_embed_in_channels,
            out_channels=hidden_size,
            patch_size=patch_size,
            bias=False,
        )

        # Positional Embedding
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
        )

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=max_size,
                patch_size=patch_size,
            )

        # Time Embedding
        self.time_embed = CosmosEmbedding(
            embedding_dim=hidden_size,
            condition_dim=hidden_size,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    out_bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # Output norm & projection
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False)

        if self.config.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, encoder_hidden_states_channels, bias=True),
                nn.GELU(),
            )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        fps: int | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # oncatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            from torchvision import transforms

            padding_mask_resized = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        # Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # Patchify input
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, T*H*W, C]

        # Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, T*H*W, C]
        else:
            raise ValueError(f"Expected timestep to have shape [B, 1, T, 1, 1] or [T], but got {timestep.shape}")

        # Process encoder hidden states
        if self.config.use_crossattn_projection:
            encoder_hidden_states = self.crossattn_proj(encoder_hidden_states)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                image_rotary_emb=image_rotary_emb,
                temb=temb,
                extra_pos_emb=extra_pos_emb,
            )

        # Output norm & projection & unpatchify
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map separate Q/K/V weights from checkpoint to fused QKV parameter in model
        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]

        weight_name_remapping = {}

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            name = weight_name_remapping.get(name, name)
            original_name = name
            lookup_name = name

            # Check if this weight should be loaded into a stacked parameter (QKV fusion)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict:
                    logger.warning(f"Skipping weight {original_name} -> {lookup_name}")
                    continue

                param = params_dict[lookup_name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
