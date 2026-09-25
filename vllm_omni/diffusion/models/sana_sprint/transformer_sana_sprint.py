# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from Diffusers' Apache-2.0 licensed SanaTransformer2DModel.

from collections.abc import Iterable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.normalization import RMSNorm
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention


class SanaAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        qk_norm: str | None,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
        linear: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.linear = linear
        inner_dim = heads * dim_head
        context_dim = cross_attention_dim or query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=bias)
        if qk_norm not in (None, "rms_norm_across_heads"):
            raise ValueError(f"Unsupported Sana-Sprint QK normalization: {qk_norm}")
        self.norm_q = RMSNorm(inner_dim, eps=1e-5) if qk_norm else None
        self.norm_k = RMSNorm(inner_dim, eps=1e-5) if qk_norm else None
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=out_bias), nn.Dropout(dropout)])
        if not linear:
            self.attn = Attention(
                num_heads=heads,
                head_size=dim_head,
                causal=False,
                softmax_scale=dim_head**-0.5,
                role="cross",
                qkv_layout="BSND",
                skip_sequence_parallel=True,
                disable_kv_quant=True,
                allow_fp32_fallback=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = self.to_q(hidden_states), self.to_k(context), self.to_v(context)
        if self.norm_q is not None:
            q, k = self.norm_q(q), self.norm_k(k)
        if self.linear:
            # ReLU kernel attention, including its normalization, accumulates in FP32.
            q = F.relu(q.transpose(1, 2).unflatten(1, (self.heads, -1))).float()
            k = F.relu(k.transpose(1, 2).unflatten(1, (self.heads, -1)).transpose(2, 3)).float()
            v = v.transpose(1, 2).unflatten(1, (self.heads, -1)).float()
            v = F.pad(v, (0, 0, 0, 1), value=1.0)
            out = (v @ k) @ q
            out = out[:, :, :-1] / (out[:, :, -1:] + 1e-15)
            out = out.flatten(1, 2).transpose(1, 2).to(hidden_states.dtype)
        else:
            q, k, v = (x.unflatten(2, (self.heads, -1)) for x in (q, k, v))
            # Flash's shared Q/K padding mask cannot represent independent text padding.
            metadata = AttentionMetadata(attn_mask=attention_mask)
            out = self.attn.sdpa_fallback.forward(q, k, v, metadata).flatten(2, 3).to(q.dtype)
        out = self.to_out[1](self.to_out[0](out))
        return out.clamp(-65504, 65504) if self.linear and out.dtype == torch.float16 else out


class GLUMBConv(nn.Module):
    def __init__(self, dim: int, expand_ratio: float) -> None:
        super().__init__()
        hidden_channels = int(expand_ratio * dim)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(dim, hidden_channels * 2, 1)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.nonlinearity(self.conv_inverted(hidden_states))
        hidden_states, gate = self.conv_depth(hidden_states).chunk(2, dim=1)
        return self.conv_point(hidden_states * self.nonlinearity(gate))


class SanaModulatedNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-06) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, scale_shift_table: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (scale_shift_table[None] + temb[:, None].to(scale_shift_table.device)).chunk(2, dim=1)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class SanaCombinedTimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self, timestep: torch.Tensor, guidance: torch.Tensor, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        guidance_proj = self.guidance_condition_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=hidden_dtype))
        conditioning = timesteps_emb + guidance_emb
        return (self.linear(self.silu(conditioning)), conditioning)


class SanaTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: int = 20,
        cross_attention_head_dim: int = 112,
        cross_attention_dim: int = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-06,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
        qk_norm: str | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            linear=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = SanaAttention(
            query_dim=dim,
            qk_norm=qk_norm,
            cross_attention_dim=cross_attention_dim,
            heads=num_cross_attention_heads,
            dim_head=cross_attention_head_dim,
            dropout=dropout,
            bias=True,
            out_bias=attention_out_bias,
            linear=False,
        )
        self.ff = GLUMBConv(dim, mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output
        attn_output = self.attn2(
            hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask
        )
        hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output
        return hidden_states


class SanaSprintTransformer2DModel(nn.Module):
    _repeated_blocks = ["SanaTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(
        self,
        *,
        in_channels: int = 32,
        out_channels: int | None = 32,
        num_attention_heads: int = 36,
        attention_head_dim: int = 32,
        num_layers: int = 28,
        num_cross_attention_heads: int = 16,
        cross_attention_head_dim: int = 72,
        cross_attention_dim: int | None = 1152,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: float | None = None,
        guidance_embeds: bool = True,
        guidance_embeds_scale: float = 0.1,
        qk_norm: str | None = "rms_norm_across_heads",
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not guidance_embeds:
            raise ValueError("Sana-Sprint requires guidance embeddings.")
        self.config = SimpleNamespace(
            in_channels=in_channels,
            sample_size=sample_size,
            patch_size=patch_size,
            guidance_embeds_scale=guidance_embeds_scale,
        )
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_type="sincos" if interpolation_scale is not None else None,
        )
        self.time_embed = SanaCombinedTimestepGuidanceEmbeddings(inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * (out_channels or in_channels))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
        batch, _, height, width = hidden_states.shape
        p = self.config.patch_size
        height, width = height // p, width // p
        hidden_states = self.patch_embed(hidden_states)
        timestep, embedded_timestep = self.time_embed(timestep, guidance, hidden_states.dtype)
        encoder_hidden_states = self.caption_norm(self.caption_projection(encoder_hidden_states))
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                height=height,
                width=width,
            )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states).reshape(batch, height, width, p, p, -1)
        return hidden_states.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, height * p, width * p)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
