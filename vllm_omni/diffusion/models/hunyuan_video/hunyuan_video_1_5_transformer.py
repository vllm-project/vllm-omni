# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply RoPE to queries/keys."""
    cos, sin = freqs_cis
    # [1, 1, S, D/2]
    cos = cos[None, None]
    sin = sin[None, None]

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class HunyuanVideo15PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideo15AdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15TimeEmbedding(nn.Module):
    """Time embedding for HunyuanVideo 1.5."""

    def __init__(self, embedding_dim: int, use_meanflow: bool = False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_meanflow = use_meanflow
        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder_r = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None and self.use_meanflow:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(timesteps_proj_r.to(dtype=timestep.dtype))
            timesteps_emb = timesteps_emb + timesteps_emb_r

        return timesteps_emb


class HunyuanVideo15IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / math.sqrt(attention_head_dim),
            causal=False,
        )
        self.to_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=attention_bias)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=attention_bias)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, int(mlp_width_ratio * hidden_size), bias=attention_bias),
            nn.SiLU(),
            nn.Linear(int(mlp_width_ratio * hidden_size), hidden_size, bias=attention_bias),
        )

        self.norm_out = HunyuanVideo15AdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        qkv = self.to_qkv(norm_hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        attn_output = self.attn(
            query=query,
            key=key,
            value=value,
            attn_metadata=attn_metadata,
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = self.to_out(attn_output)

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideo15IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideo15IndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideo15TokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideo15RotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: list[int], theta: float = 256.0) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
        return freqs_cos, freqs_sin


class HunyuanVideo15ByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_features)
        self.act_fn = nn.GELU()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class HunyuanVideo15ImageProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(in_channels, in_channels)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(in_channels, hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class HunyuanVideo15Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=True)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=True)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=True)

        self.add_q_proj = nn.Linear(hidden_size, self.inner_dim, bias=True)
        self.add_k_proj = nn.Linear(hidden_size, self.inner_dim, bias=True)
        self.add_v_proj = nn.Linear(hidden_size, self.inner_dim, bias=True)

        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
            self.norm_k = RMSNorm(self.head_dim, eps=1e-6)
            self.norm_added_q = RMSNorm(self.head_dim, eps=1e-6)
            self.norm_added_k = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.norm_added_q = nn.Identity()
            self.norm_added_k = nn.Identity()

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.inner_dim)])
        self.to_add_out = nn.Linear(self.inner_dim, self.inner_dim)

        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / math.sqrt(attention_head_dim),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (self.num_heads, -1))
            encoder_key = encoder_key.unflatten(2, (self.num_heads, -1))
            encoder_value = encoder_value.unflatten(2, (self.num_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        attn_metadata = None
        if attention_mask is not None:
            batch_size = query.shape[0]
            attention_mask = F.pad(attention_mask, (query.shape[1] - attention_mask.shape[1], 0), value=True)
            attention_mask = attention_mask.bool()
            attn_mask = attention_mask.view(batch_size, 1, 1, attention_mask.shape[1]).repeat(
                1, 1, attention_mask.shape[1], 1
            )
            attn_mask = (attn_mask & attn_mask.transpose(2, 3)).bool()
            attn_metadata = AttentionMetadata(attn_mask=attn_mask)

        joint_hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(query.dtype)

        if encoder_hidden_states is None:
            hidden_states = self.to_out[0](joint_hidden_states)
            return hidden_states, encoder_hidden_states

        hidden_states, encoder_hidden_states = (
            joint_hidden_states[:, : hidden_states.shape[1]],
            joint_hidden_states[:, hidden_states.shape[1] :],
        )

        hidden_states = self.to_out[0](hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class HunyuanVideo15TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = HunyuanVideo15Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm=qk_norm,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, int(mlp_ratio * hidden_size)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(mlp_ratio * hidden_size), hidden_size),
        )

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = nn.Sequential(
            nn.Linear(hidden_size, int(mlp_ratio * hidden_size)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(mlp_ratio * hidden_size), hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideo15Transformer3DModel(nn.Module):
    """
    HunyuanVideo1.5 transformer adapted for vLLM.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int, ...] = (16, 56, 56),
        target_size: int = 640,
        task_type: str = "i2v",
        use_meanflow: bool = False,
    ) -> None:
        super().__init__()

        model_config = od_config.tf_model_config
        in_channels = model_config.get("in_channels", in_channels)
        out_channels = model_config.get("out_channels", out_channels)
        num_attention_heads = model_config.get("num_attention_heads", num_attention_heads)
        attention_head_dim = model_config.get("attention_head_dim", attention_head_dim)
        num_layers = model_config.get("num_layers", num_layers)
        num_refiner_layers = model_config.get("num_refiner_layers", num_refiner_layers)
        mlp_ratio = model_config.get("mlp_ratio", mlp_ratio)
        patch_size = model_config.get("patch_size", patch_size)
        patch_size_t = model_config.get("patch_size_t", patch_size_t)
        qk_norm = model_config.get("qk_norm", qk_norm)
        text_embed_dim = model_config.get("text_embed_dim", text_embed_dim)
        text_embed_2_dim = model_config.get("text_embed_2_dim", text_embed_2_dim)
        image_embed_dim = model_config.get("image_embed_dim", image_embed_dim)
        rope_theta = model_config.get("rope_theta", rope_theta)
        rope_axes_dim = tuple(model_config.get("rope_axes_dim", rope_axes_dim))
        target_size = model_config.get("target_size", target_size)
        task_type = model_config.get("task_type", task_type)
        use_meanflow = model_config.get("use_meanflow", use_meanflow)

        self.text_embed_2_dim = text_embed_2_dim
        self.image_embed_dim = image_embed_dim

        self.config = model_config
        self.config.target_size = target_size
        self.config.task_type = task_type
        self.config.patch_size = patch_size
        self.config.patch_size_t = patch_size_t

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.x_embedder = HunyuanVideo15PatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.image_embedder = HunyuanVideo15ImageProjection(image_embed_dim, inner_dim)

        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(text_embed_2_dim, 2048, inner_dim)

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=use_meanflow)

        self.cond_type_embed = nn.Embedding(3, inner_dim)
        self.rope = HunyuanVideo15RotaryPosEmbed(patch_size, patch_size_t, list(rope_axes_dim), rope_theta)

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideo15TransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        image_rotary_emb = self.rope(hidden_states)
        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        if encoder_hidden_states_2 is None:
            encoder_hidden_states_2 = torch.zeros(
                (batch_size, 1, self.text_embed_2_dim), device=hidden_states.device, dtype=hidden_states.dtype
            )
            encoder_attention_mask_2 = torch.zeros(
                (batch_size, 1), device=hidden_states.device, dtype=encoder_attention_mask.dtype
            )

        if image_embeds is None:
            image_embeds = torch.zeros(
                (batch_size, 1, self.image_embed_dim), device=hidden_states.device, dtype=hidden_states.dtype
            )

        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)
        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)
        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],
                        text_2[text_mask_2],
                        text[text_mask],
                        image[~image_mask],
                        torch.zeros_like(text_2[~text_mask_2]),
                        torch.zeros_like(text[~text_mask]),
                    ],
                    dim=0,
                )
            )

            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
