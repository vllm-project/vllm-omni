# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module

ANIMA_TRANSFORMER_CONFIG = {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}


# NOTE: We import and use diffusers' `apply_rotary_emb` instead of a custom native implementation
# to prevent numerical drift in bfloat16. Diffusers upcasts queries, keys, and rotary frequency
# tensors to float32 before computing the rotation, and casts back to bfloat16 at the end.
# Performing the entire computation in bfloat16 accumulates precision errors across the 28
# transformer blocks, which is heavily amplified by Classifier-Free Guidance (CFG).


class CosmosPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], out_channels, bias=bias)

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
        return self.proj(hidden_states)


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps)
        emb = self.activation(emb)
        return self.linear_2(emb)


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
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

        return hidden_states * (1 + scale) + shift


class CosmosAdaLayerNormZero(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()
        self.linear_1 = (
            nn.Identity() if hidden_features is None else nn.Linear(in_features, hidden_features, bias=False)
        )
        self.linear_2 = nn.Linear(hidden_features or in_features, 3 * in_features, bias=False)

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


class CosmosGELU(nn.Module):
    def __init__(self, dim: int, inner_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states))


class CosmosFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, bias: bool = False) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList(
            [
                CosmosGELU(dim, inner_dim, bias=bias),
                nn.Identity(),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CosmosAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        out_bias: bool = False,
        img_context: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        cross_attention_dim = query_dim if cross_attention_dim is None else cross_attention_dim

        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = inner_dim
        self.query_dim = query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-6)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=out_bias), nn.Identity()])
        self.attn = Attention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
            num_kv_heads=heads,
            prefix=prefix,
        )

        self.img_context = img_context
        if img_context:
            self.q_img = nn.Linear(query_dim, inner_dim, bias=False)
            self.k_img = nn.Linear(query_dim, inner_dim, bias=False)
            self.v_img = nn.Linear(query_dim, inner_dim, bias=False)
            self.q_img_norm = nn.RMSNorm(dim_head, eps=1e-6)
            self.k_img_norm = nn.RMSNorm(dim_head, eps=1e-6)
            self.img_attn = Attention(
                num_heads=heads,
                head_size=dim_head,
                softmax_scale=1.0 / (dim_head**0.5),
                causal=False,
                num_kv_heads=heads,
                prefix=f"{prefix}.img_attn" if prefix else "img_attn",
            )

    def _attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]

        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.dim_head)
        key = self.to_k(encoder_hidden_states).view(batch_size, encoder_seq_len, self.heads, self.dim_head)
        value = self.to_v(encoder_hidden_states).view(batch_size, encoder_seq_len, self.heads, self.dim_head)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            # We use diffusers' apply_rotary_emb to leverage its internal float32 rotation upcasting
            # logic, resolving the bfloat16 cumulative precision drift vs. the reference pipeline.
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1, use_real_unbind_dim=-2)

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)
        return hidden_states.flatten(2, 3).type_as(query)

    def _image_attention(
        self,
        hidden_states: torch.Tensor,
        img_context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        img_seq_len = img_context.shape[1]

        query = self.q_img(hidden_states).view(batch_size, seq_len, self.heads, self.dim_head)
        key = self.k_img(img_context).view(batch_size, img_seq_len, self.heads, self.dim_head)
        value = self.v_img(img_context).view(batch_size, img_seq_len, self.heads, self.dim_head)

        query = self.q_img_norm(query)
        key = self.k_img_norm(key)

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        hidden_states = self.img_attn(query, key, value, attn_metadata=attn_metadata)
        return hidden_states.flatten(2, 3).type_as(query)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None] | None = None,
        attention_mask: torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.img_context:
            if not isinstance(encoder_hidden_states, tuple):
                raise ValueError("Expected encoder_hidden_states as (text_context, img_context).")
            text_context, img_context = encoder_hidden_states
            text_mask, img_mask = attention_mask if attention_mask is not None else (None, None)
            query_states = hidden_states
            hidden_states = self._attention(query_states, text_context, text_mask, image_rotary_emb)
            if img_context is not None:
                hidden_states = hidden_states + self._image_attention(query_states, img_context, img_mask)
        else:
            hidden_states = self._attention(hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        out_bias: bool = False,
        img_context: bool = False,
        before_proj: bool = False,
        after_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.img_context = img_context
        self.attn1 = CosmosAttention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_bias=out_bias,
            prefix=f"{prefix}.attn1" if prefix else "attn1",
        )
        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn2 = CosmosAttention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_bias=out_bias,
            img_context=img_context,
            prefix=f"{prefix}.attn2" if prefix else "attn2",
        )
        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = CosmosFeedForward(hidden_size, mult=mlp_ratio, bias=out_bias)

        self.before_proj = nn.Linear(hidden_size, hidden_size) if before_proj else None
        self.after_proj = nn.Linear(hidden_size, hidden_size) if after_proj else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        controlnet_residual: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        block_idx: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.before_proj is not None:
            hidden_states = self.before_proj(hidden_states) + latents

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate * ff_output

        if controlnet_residual is not None:
            if self.after_proj is not None:
                raise ValueError("controlnet_residual is incompatible with after_proj.")
            hidden_states = hidden_states + controlnet_residual

        if self.after_proj is not None:
            return hidden_states, self.after_proj(hidden_states)

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
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

    def forward(self, hidden_states: torch.Tensor, fps: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        dim_h_range = torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: self.dim_h // 2] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: self.dim_w // 2] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: self.dim_t // 2] / self.dim_t
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
        return torch.cos(freqs), torch.sin(freqs)


class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.eps = eps
        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]

        emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = (emb_t + emb_h + emb_w).flatten(1, 3)

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


class AnimaTransformer3DModel(nn.Module):
    _repeated_blocks = ["CosmosTransformerBlock"]
    _hsdp_shard_conditions = [is_transformer_block_module]

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: str | None = "learnable",
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        encoder_hidden_states_channels: int = 1024,
        controlnet_block_every_n: int | None = None,
        img_context_dim_in: int | None = None,
        img_context_num_tokens: int = 256,
        img_context_dim_out: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            text_embed_dim=text_embed_dim,
            adaln_lora_dim=adaln_lora_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
            concat_padding_mask=concat_padding_mask,
            extra_pos_embed_type=extra_pos_embed_type,
            use_crossattn_projection=use_crossattn_projection,
            crossattn_proj_in_channels=crossattn_proj_in_channels,
            encoder_hidden_states_channels=encoder_hidden_states_channels,
            controlnet_block_every_n=controlnet_block_every_n,
            img_context_dim_in=img_context_dim_in,
            img_context_num_tokens=img_context_num_tokens,
            img_context_dim_out=img_context_dim_out,
            extra_config=kwargs,
        )

        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
        )
        self.learnable_pos_embed: CosmosLearnablePositionalEmbed | None = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=max_size,
                patch_size=patch_size,
            )
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    out_bias=False,
                    img_context=img_context_dim_in is not None and img_context_dim_in > 0,
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False)

        self.crossattn_proj: nn.Module | None = None
        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, encoder_hidden_states_channels, bias=True),
                nn.GELU(),
            )

        self.img_context_proj: nn.Module | None = None
        if img_context_dim_in:
            self.img_context_proj = nn.Sequential(
                nn.Linear(img_context_dim_in, img_context_dim_out, bias=True),
                nn.GELU(),
            )

        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        fps: float | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        batch_size, _, num_frames, height, width = hidden_states.shape

        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            if padding_mask is None:
                raise ValueError("padding_mask is required when concat_padding_mask=True.")
            padding_mask_resized = F.interpolate(
                padding_mask,
                size=list(hidden_states.shape[-2:]),
                mode="nearest",
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)],
                dim=1,
            )

        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim != 2:
            attention_mask = attention_mask.squeeze(1).squeeze(1)

        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.learnable_pos_embed is not None else None

        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)

        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            if timestep.shape != (batch_size, 1, num_frames, 1, 1):
                raise ValueError(f"Expected timestep shape [B, 1, T, 1, 1], got {timestep.shape}.")
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )
        else:
            raise ValueError(f"Expected timestep shape [B, 1, T, 1, 1] or [T], got {timestep.shape}.")

        text_context, img_context = (
            encoder_hidden_states if isinstance(encoder_hidden_states, tuple) else (encoder_hidden_states, None)
        )
        if self.config.use_crossattn_projection:
            if self.crossattn_proj is None:
                raise ValueError("crossattn_proj is required when use_crossattn_projection=True.")
            text_context = self.crossattn_proj(text_context)
        if img_context is not None and self.config.img_context_dim_in:
            if self.img_context_proj is None:
                raise ValueError("img_context_proj is required when img_context_dim_in is set.")
            img_context = self.img_context_proj(img_context)
        processed_encoder_hidden_states = (
            (text_context, img_context) if isinstance(encoder_hidden_states, tuple) else text_context
        )

        controlnet_block_index_map = {}
        if block_controlnet_hidden_states is not None:
            n_blocks = len(self.transformer_blocks)
            controlnet_block_index_map = {
                block_idx: block_controlnet_hidden_states[idx]
                for idx, block_idx in list(enumerate(range(0, n_blocks, self.config.controlnet_block_every_n)))
            }

        for block_idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                processed_encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                extra_pos_emb,
                attention_mask,
                controlnet_block_index_map.get(block_idx),
            )

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
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name not in params:
                continue
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(name)
        return loaded
