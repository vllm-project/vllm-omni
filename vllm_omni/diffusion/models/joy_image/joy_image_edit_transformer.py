# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2025 The JoyImage Team and The HuggingFace Team
# Adapted from Hugging Face Diffusers commit
# 23ba73e1d2079c4b89959484ed0ca1c22e7ef998:
# src/diffusers/models/transformers/transformer_joyimage.py

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def _as_3tuple(value: int | list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected a 3D patch size, got {value!r}.")
    return (int(value[0]), int(value[1]), int(value[2]))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def _apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = xq.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq.shape)]
    cos = freqs_cis[0].view(*shape).to(xq.device)
    sin = freqs_cis[1].view(*shape).to(xq.device)
    xq_out = (xq.float() * cos + _rotate_half(xq) * sin).type_as(xq)
    xk_out = (xk.float() * cos + _rotate_half(xk) * sin).type_as(xk)
    return xq_out, xk_out


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.layer_norm(
            hidden_states.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return hidden_states.to(dtype)


class GeluApproximate(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class FeedForward(nn.Module):
    """Small local equivalent of the Diffusers FeedForward used by JoyImage."""

    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                GeluApproximate(dim, inner_dim),
                nn.Identity(),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = timesteps[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        caption = self.linear_1(caption)
        caption = self.act_1(caption)
        return self.linear_2(caption)


class JoyImageModulate(nn.Module):
    def __init__(self, hidden_size: int, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(torch.zeros(1, factor, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        if hidden_states.ndim != 3:
            hidden_states = hidden_states.unsqueeze(1)
        return [item.squeeze(1) for item in (self.modulate_table + hidden_states).chunk(self.factor, dim=1)]


class JoyImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.img_attn_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.img_attn_q_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.img_attn_k_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.img_attn_proj = nn.Linear(inner_dim, dim, bias=True)

        self.txt_attn_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.txt_attn_q_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.txt_attn_k_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.txt_attn_proj = nn.Linear(inner_dim, dim, bias=True)

    def _stream_qkv(
        self,
        qkv_proj: nn.Linear,
        q_norm: nn.Module,
        k_norm: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = qkv_proj(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, self.head_dim))
        key = key.unflatten(-1, (self.heads, self.head_dim))
        value = value.unflatten(-1, (self.heads, self.head_dim))
        return q_norm(query), k_norm(key), value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[
            tuple[torch.Tensor, torch.Tensor] | None,
            tuple[torch.Tensor, torch.Tensor] | None,
        ]
        | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_query, img_key, img_value = self._stream_qkv(
            self.img_attn_qkv,
            self.img_attn_q_norm,
            self.img_attn_k_norm,
            hidden_states,
        )
        txt_query, txt_key, txt_value = self._stream_qkv(
            self.txt_attn_qkv,
            self.txt_attn_q_norm,
            self.txt_attn_k_norm,
            encoder_hidden_states,
        )

        if image_rotary_emb is not None:
            vis_freqs, txt_freqs = image_rotary_emb
            if vis_freqs is not None:
                img_query, img_key = _apply_rotary_emb(img_query, img_key, vis_freqs)
            if txt_freqs is not None:
                txt_query, txt_key = _apply_rotary_emb(txt_query, txt_key, txt_freqs)

        joint_query = torch.cat([img_query, txt_query], dim=1)
        joint_key = torch.cat([img_key, txt_key], dim=1)
        joint_value = torch.cat([img_value, txt_value], dim=1)

        joint_hidden_states = F.scaled_dot_product_attention(
            joint_query.transpose(1, 2),
            joint_key.transpose(1, 2),
            joint_value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        img_attn_output = joint_hidden_states[:, : hidden_states.shape[1], :]
        txt_attn_output = joint_hidden_states[:, hidden_states.shape[1] :, :]
        return self.img_attn_proj(img_attn_output), self.txt_attn_proj(txt_attn_output)


class JoyImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_width_ratio)

        self.img_mod = JoyImageModulate(dim, factor=6)
        self.img_norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim)

        self.txt_mod = JoyImageModulate(dim, factor=6)
        self.txt_norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim)

        self.attn = JoyImageAttention(
            dim,
            num_attention_heads,
            attention_head_dim,
            eps=eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[
            tuple[torch.Tensor, torch.Tensor] | None,
            tuple[torch.Tensor, torch.Tensor] | None,
        ],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(temb)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(temb)

        img_normed = self.img_norm1(hidden_states)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        img_modulated = img_normed * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        txt_modulated = txt_normed * (1 + txt_mod1_scale.unsqueeze(1)) + txt_mod1_shift.unsqueeze(1)
        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + img_attn * img_mod1_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + txt_attn * txt_mod1_gate.unsqueeze(1)

        img_ffn_normed = self.img_norm2(hidden_states)
        txt_ffn_normed = self.txt_norm2(encoder_hidden_states)
        img_ffn_input = img_ffn_normed * (1 + img_mod2_scale.unsqueeze(1)) + img_mod2_shift.unsqueeze(1)
        txt_ffn_input = txt_ffn_normed * (1 + txt_mod2_scale.unsqueeze(1)) + txt_mod2_shift.unsqueeze(1)
        hidden_states = hidden_states + self.img_mlp(img_ffn_input) * img_mod2_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.txt_mlp(txt_ffn_input) * txt_mod2_gate.unsqueeze(1)
        return hidden_states, encoder_hidden_states


class JoyImageTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class JoyImageEditTransformer3DModel(nn.Module):
    _skip_layerwise_casting_patterns = ["img_in", "condition_embedder", "norm"]
    _no_split_modules = ["JoyImageTransformerBlock"]
    _repeated_blocks = ["JoyImageTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["double_blocks"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int | list[int] | tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        text_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mlp_ratio: float | None = None,
        num_layers: int = 40,
        rope_dim_list: list[int] | tuple[int, ...] | None = None,
        rope_type: str = "rope",
        theta: int = 256,
        **_: Any,
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = _as_3tuple(patch_size)
        self.rope_type = rope_type
        self.theta = theta

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})."
            )
        attention_head_dim = hidden_size // num_attention_heads
        if rope_dim_list is None:
            if attention_head_dim == 128:
                rope_dim_list = [16, 56, 56]
            else:
                base = max(2, (attention_head_dim // 3) // 2 * 2)
                rope_dim_list = [base, base, attention_head_dim - base * 2]
        self.rope_dim_list = [int(dim) for dim in rope_dim_list]
        if sum(self.rope_dim_list) != attention_head_dim:
            raise ValueError("sum(rope_dim_list) must equal hidden_size // num_attention_heads.")

        self.img_in = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.condition_embedder = JoyImageTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_dim,
        )
        block_mlp_ratio = mlp_ratio if mlp_ratio is not None else mlp_width_ratio
        self.double_blocks = nn.ModuleList(
            [
                JoyImageTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=block_mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, self.out_channels * math.prod(self.patch_size))
        self.gradient_checkpointing = False

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        *,
        od_config: OmniDiffusionConfig | None = None,
        **overrides: Any,
    ) -> JoyImageEditTransformer3DModel:
        with open(config_path) as config_file:
            config = json.load(config_file)
        config.update(overrides)
        return cls(od_config=od_config, **config)

    def get_rotary_pos_embed(
        self,
        vis_rope_size: list[int],
        txt_rope_size: int | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        target_ndim = 3
        if len(vis_rope_size) != target_ndim:
            vis_rope_size = [1] * (target_ndim - len(vis_rope_size)) + list(vis_rope_size)

        grid = torch.stack(
            torch.meshgrid(
                *[torch.linspace(0, size, size + 1, dtype=torch.float32)[:size] for size in vis_rope_size],
                indexing="ij",
            ),
            dim=0,
        )

        vis_cos: list[torch.Tensor] = []
        vis_sin: list[torch.Tensor] = []
        for axis, dim in enumerate(self.rope_dim_list):
            pos = grid[axis].reshape(-1)
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
            freqs = torch.outer(pos.float(), freqs)
            vis_cos.append(freqs.cos().repeat_interleave(2, dim=1))
            vis_sin.append(freqs.sin().repeat_interleave(2, dim=1))
        vis_freqs = (torch.cat(vis_cos, dim=1), torch.cat(vis_sin, dim=1))

        if txt_rope_size is None:
            return vis_freqs, None

        text_positions = torch.arange(txt_rope_size) + grid.reshape(-1).max().item() + 1
        txt_cos: list[torch.Tensor] = []
        txt_sin: list[torch.Tensor] = []
        for dim in self.rope_dim_list:
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
            freqs = torch.outer(text_positions.float(), freqs)
            txt_cos.append(freqs.cos().repeat_interleave(2, dim=1))
            txt_sin.append(freqs.sin().repeat_interleave(2, dim=1))
        txt_freqs = (torch.cat(txt_cos, dim=1), torch.cat(txt_sin, dim=1))
        return vis_freqs, txt_freqs

    def unpatchify(self, image_tokens: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        channels = self.out_channels
        patch_t, patch_h, patch_w = self.patch_size
        if t * h * w != image_tokens.shape[1]:
            raise ValueError(f"Expected t*h*w ({t * h * w}) to equal token count ({image_tokens.shape[1]}).")
        image_tokens = image_tokens.reshape(image_tokens.shape[0], t, h, w, patch_t, patch_h, patch_w, channels)
        image_tokens = image_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return image_tokens.reshape(
            image_tokens.shape[0],
            channels,
            t * patch_t,
            h * patch_h,
            w * patch_w,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> tuple[torch.Tensor] | dict[str, torch.Tensor]:
        is_multi_item = hidden_states.ndim == 6
        num_items = 0
        frames_per_item = hidden_states.shape[3] if is_multi_item else None
        if is_multi_item:
            num_items = hidden_states.shape[1]
            if num_items > 1:
                if self.patch_size[0] != 1:
                    raise ValueError("For multi-item input, patch_size[0] must be 1.")
                hidden_states = torch.cat([hidden_states[:, -1:], hidden_states[:, :-1]], dim=1)
            batch_size, num_items, channels, frames, height, width = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(
                batch_size,
                channels,
                num_items * frames,
                height,
                width,
            )
        elif hidden_states.ndim != 5:
            raise ValueError(
                "JoyImageEditTransformer3DModel expects (B, C, T, H, W) or "
                f"(B, N, C, T, H, W), got {tuple(hidden_states.shape)}."
            )

        batch_size, _, original_frames, original_height, original_width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        if original_frames % patch_t or original_height % patch_h or original_width % patch_w:
            raise ValueError(
                "Latent shape must be divisible by patch_size; "
                f"got {(original_frames, original_height, original_width)} and {self.patch_size}."
            )
        patch_frames = original_frames // patch_t
        patch_height = original_height // patch_h
        patch_width = original_width // patch_w

        image_tokens = self.img_in(hidden_states).flatten(2).transpose(1, 2)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)

        _, vector_embedding, text_tokens = self.condition_embedder(
            timestep,
            encoder_hidden_states.to(device=hidden_states.device),
        )
        if vector_embedding.shape[-1] > self.hidden_size:
            vector_embedding = vector_embedding.unflatten(1, (6, -1))

        vis_freqs, txt_freqs = self.get_rotary_pos_embed(
            vis_rope_size=[patch_frames, patch_height, patch_width],
            txt_rope_size=text_tokens.shape[1] if self.rope_type == "mrope" else None,
        )
        vis_freqs = tuple(item.to(hidden_states.device) for item in vis_freqs)
        if txt_freqs is not None:
            txt_freqs = tuple(item.to(hidden_states.device) for item in txt_freqs)

        attention_mask = None
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(device=hidden_states.device, dtype=torch.bool)
            image_mask = torch.ones(
                batch_size,
                image_tokens.shape[1],
                device=hidden_states.device,
                dtype=torch.bool,
            )
            attention_mask = torch.cat([image_mask, encoder_hidden_states_mask], dim=1)[:, None, None, :]

        for block in self.double_blocks:
            image_tokens, text_tokens = block(
                hidden_states=image_tokens,
                encoder_hidden_states=text_tokens,
                temb=vector_embedding,
                image_rotary_emb=(vis_freqs, txt_freqs),
                attention_mask=attention_mask,
            )

        image_tokens = self.proj_out(self.norm_out(image_tokens))
        output = self.unpatchify(image_tokens, patch_frames, patch_height, patch_width)

        if is_multi_item:
            channels = output.shape[1]
            output = output.reshape(
                batch_size,
                channels,
                num_items,
                frames_per_item,
                original_height,
                original_width,
            )
            output = output.permute(0, 2, 1, 3, 4, 5)
            if num_items > 1:
                output = torch.cat([output[:, 1:], output[:, :1]], dim=1)

        if return_dict:
            return {"sample": output}
        return (output,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            mapped_name = self._map_weight_name(name)
            param = params.get(mapped_name)
            if param is None:
                logger.warning(
                    "Skipping JoyAI-Image-Edit transformer weight %s mapped to "
                    "%s -- not found in model parameters. This may indicate an "
                    "incomplete implementation or checkpoint mismatch.",
                    name,
                    mapped_name,
                )
                continue
            default_weight_loader(param, loaded_weight)
            loaded.add(mapped_name)
        return loaded

    @staticmethod
    def _map_weight_name(name: str) -> str:
        if name.startswith("transformer."):
            name = name[len("transformer.") :]
        prefix_replacements = {
            "patch_embedding.": "img_in.",
            "time_embed.": "condition_embedder.time_embedder.",
            "time_text_embed.": "condition_embedder.",
            "text_embedder.": "condition_embedder.text_embedder.",
            "final_layer.linear.": "proj_out.",
            "final_layer.norm_final.": "norm_out.",
        }
        for source, target in prefix_replacements.items():
            if name.startswith(source):
                return target + name[len(source) :]
        return name
