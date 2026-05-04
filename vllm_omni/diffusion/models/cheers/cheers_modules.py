# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Cheers generation-specific modules ported from HF modeling_umm.py.

Includes: Siglip2VisionTransformer, UMMTextModel (custom Qwen2),
TimestepEmbedder, DiTAttention, ModulatedAttentionBlock, FinalLayer,
DiTRotaryEmbedding, CheersGenProjector, HiGate, CheersHiProjector,
VAE components, and UndProjector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling

# ── KV Cache ───────────────────────────────────────────────────────


class NaiveCache:
    """Simple KV cache matching Bagel's NaiveCache interface.

    Stores 3-D tensors ``(seq_len, kv_heads, head_dim)`` per layer,
    identical to the format produced by the vLLM-Omni KV transfer.
    """

    def __init__(self, num_layers: int):
        self.key_cache: dict[int, Tensor | None] = {k: None for k in range(num_layers)}
        self.value_cache: dict[int, Tensor | None] = {k: None for k in range(num_layers)}

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @property
    def seq_lens(self) -> int:
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0


def _kv_seq_len(cache) -> int:
    """Return the sequence length stored in *cache* (NaiveCache / SimpleNamespace)."""
    if cache is None:
        return 0
    kc = cache.key_cache
    first = kc[0] if isinstance(kc, (list, dict)) else None
    return first.shape[0] if first is not None else 0


def _crop_kv_cache(cache, n: int) -> None:
    """Remove the last *n* tokens from every layer in *cache* (in-place)."""
    kc = cache.key_cache
    vc = cache.value_cache
    indices = range(len(kc)) if isinstance(kc, list) else kc.keys()
    for i in indices:
        if kc[i] is not None:
            kc[i] = kc[i][:-n]
            vc[i] = vc[i][:-n]


# ── Utilities ──────────────────────────────────────────────────────


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    shift = shift.to(torch.float32)
    scale = scale.to(torch.float32)
    if len(x.shape) != len(shift.shape):
        return (x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)).to(input_dtype)
    return (x * (1 + scale) + shift).to(input_dtype)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ── RMSNorm ────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ── MLP ────────────────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ── DiT Rotary Embedding ──────────────────────────────────────────


class DiTRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ── DiT Attention ─────────────────────────────────────────────────


class DiTAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: repeat KV heads
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


# ── Modulated Attention Block (DiT block) ──────────────────────────


class ModulatedAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = DiTAttention(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
        )
        self.mlp = MLP(hidden_size, hidden_size * 4)
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        hidden_states: Tensor,
        adaln_input: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(
            6, dim=-1
        )
        residual = hidden_states
        hidden_states = modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        hidden_states, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + gate_msa * hidden_states

        residual = hidden_states
        hidden_states = modulate(self.post_attention_layernorm(hidden_states), shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ── Final Layer ────────────────────────────────────────────────────


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, adaln_input: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ── Gen Projector (7-layer DiT) ────────────────────────────────────


class CheersGenProjector(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        patch_size: int,
        output_dim: int,
        layers_num: int,
    ):
        super().__init__()
        self.diffusion_head_a = nn.ModuleList(
            [
                ModulatedAttentionBlock(
                    hidden_size=embed_dim,
                    layer_idx=layer_idx,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                )
                for layer_idx in range(layers_num)
            ]
        )
        self.diffusion_head_b = FinalLayer(
            hidden_size=embed_dim,
            patch_size=patch_size,
            out_channels=output_dim,
        )
        self.rotary_emb = DiTRotaryEmbedding(
            dim=embed_dim // num_attention_heads,
            max_position_embeddings=2048,
            base=10000,
        )

    def forward(self, x: Tensor, time_embeds: Tensor, position_ids: Tensor | None = None) -> Tensor:
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        hidden_states = x
        for layer in self.diffusion_head_a:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                adaln_input=time_embeds,
                position_ids=position_ids,
            )[0]
        return self.diffusion_head_b(hidden_states, time_embeds)


# ── HiGate (semantic residual fusion) ─────────────────────────────


class HiGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, low_info: Tensor, high_info: Tensor) -> Tensor:
        hi_gate = self.gate(low_info)
        output = low_info + hi_gate * high_info
        return self.layer_norm(output)


# ── Hi Projector (3-layer DiT with doubled RoPE) ──────────────────


class CheersHiProjector(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        patch_size: int,
        output_dim: int,
        layers_num: int,
    ):
        super().__init__()
        self.diffusion_head_a = nn.ModuleList(
            [
                ModulatedAttentionBlock(
                    hidden_size=embed_dim,
                    layer_idx=layer_idx,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                )
                for layer_idx in range(layers_num)
            ]
        )
        self.diffusion_head_b = FinalLayer(
            hidden_size=embed_dim,
            patch_size=patch_size,
            out_channels=output_dim,
        )
        self.rotary_emb = DiTRotaryEmbedding(
            dim=embed_dim // num_attention_heads,
            max_position_embeddings=2048,
            base=10000,
        )

    def forward(self, x: Tensor, time_embeds: Tensor) -> Tensor:
        # HiProjector uses half-length position ids, then doubles cos/sin
        position_ids = torch.arange(x.shape[1] // 2, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, position_ids)
        cos = cos.repeat(1, 2, 1)
        sin = sin.repeat(1, 2, 1)
        position_embeddings = (cos, sin)
        hidden_states = x
        for layer in self.diffusion_head_a:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                adaln_input=time_embeds,
                position_ids=position_ids,
            )[0]
        return self.diffusion_head_b(hidden_states, time_embeds)


# ── Timestep Embedder (dual MLP) ──────────────────────────────────


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size_1: int,
        hidden_size_2: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.mlp_1 = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size_1, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size_1, hidden_size_1, bias=True),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size_2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size_2, hidden_size_2, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        return self.mlp_1(t_freq), self.mlp_2(t_freq)


# ── VAE components ─────────────────────────────────────────────────


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        return x + self.proj_out(h_)


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = _swish(self.norm1(x))
        h = self.conv1(h)
        h = _swish(self.norm2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class _Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class CheersVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("ch", 128) if isinstance(config, dict) else getattr(config, "ch", 128)
        ch_mult = (
            config.get("ch_mult", [1, 2, 4, 4])
            if isinstance(config, dict)
            else getattr(config, "ch_mult", [1, 2, 4, 4])
        )
        num_res_blocks = (
            config.get("num_res_blocks", 2) if isinstance(config, dict) else getattr(config, "num_res_blocks", 2)
        )
        z_channels = config.get("z_channels", 32) if isinstance(config, dict) else getattr(config, "z_channels", 32)
        in_channels = config.get("in_channels", 3) if isinstance(config, dict) else getattr(config, "in_channels", 3)
        num_resolutions = len(ch_mult)

        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != num_resolutions - 1:
                down.downsample = _Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, 1, 1)
        self._num_resolutions = num_resolutions
        self._num_res_blocks = num_res_blocks

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self._num_resolutions):
            for i_block in range(self._num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = _swish(self.norm_out(h))
        h = self.conv_out(h)
        return self.quant_conv(h)


class CheersVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("ch", 128) if isinstance(config, dict) else getattr(config, "ch", 128)
        ch_mult = (
            config.get("ch_mult", [1, 2, 4, 4])
            if isinstance(config, dict)
            else getattr(config, "ch_mult", [1, 2, 4, 4])
        )
        num_res_blocks = (
            config.get("num_res_blocks", 2) if isinstance(config, dict) else getattr(config, "num_res_blocks", 2)
        )
        z_channels = config.get("z_channels", 32) if isinstance(config, dict) else getattr(config, "z_channels", 32)
        out_ch = config.get("out_ch", 3) if isinstance(config, dict) else getattr(config, "out_ch", 3)
        num_resolutions = len(ch_mult)

        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        block_in = ch * ch_mult[num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)
        self._num_resolutions = num_resolutions
        self._num_res_blocks = num_res_blocks

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.to(upscale_dtype)
        for i_level in reversed(range(self._num_resolutions)):
            for i_block in range(self._num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = _swish(self.norm_out(h))
        return self.conv_out(h)


class CheersVAEModel(nn.Module):
    """Full VAE model with encoder, decoder, and BatchNorm normalization."""

    def __init__(self, enc_config, dec_config, z_channels: int = 32):
        super().__init__()
        self.encoder = CheersVAEEncoder(enc_config)
        self.decoder = CheersVAEDecoder(dec_config)
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=1e-4,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + 1e-4)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(
            mean,
            "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.normalize(z)

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)


class CheersVAEDecoderProjector(nn.Module):
    """Decoder-only projector for latent→pixel-space conversion during denoising."""

    def __init__(self, dec_config, z_channels: int = 32):
        super().__init__()
        self.decoder = CheersVAEDecoder(dec_config)
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=1e-4,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def forward(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + 1e-4)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        z = z * s + m
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)


# ── Understanding Projector ────────────────────────────────────────


class CheersUndProjector(nn.Module):
    """Maps SigLIP features to LLM dimension with 2x2 spatial compression."""

    def __init__(self, image_embed_dim: int, text_embed_dim: int, compression_factor: tuple[int, int] = (2, 2)):
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.compression_factor = compression_factor
        self.layernorm = nn.LayerNorm(image_embed_dim)
        hidden_size = image_embed_dim * (compression_factor[0] * compression_factor[1])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, text_embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layernorm(x)
        height = width = int(x.size(1) ** 0.5)
        x = x.permute(0, 2, 1).unflatten(-1, (height, width))
        batch_size, dim, height, width = x.shape
        unfolded = x.unfold(2, self.compression_factor[0], self.compression_factor[0]).unfold(
            3, self.compression_factor[1], self.compression_factor[1]
        )
        unfolded = unfolded.contiguous().view(
            batch_size,
            dim,
            -1,
            self.compression_factor[0] * self.compression_factor[1],
        )
        unfolded = (
            unfolded.permute(0, 2, 3, 1)
            .contiguous()
            .view(
                batch_size,
                -1,
                dim * self.compression_factor[0] * self.compression_factor[1],
            )
        )
        return self.mlp(unfolded)


# ── Siglip2 Vision Transformer ────────────────────────────────────
# Ported from HF Cheers modeling_umm.py — NOT the standard
# transformers SiglipVisionModel.  Key difference: this is
# Siglip2VisionTransformer with its own embedding, attention, and
# layernorm scheme.


@dataclass
class Siglip2VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    output_attentions: bool = False
    output_hidden_states: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Siglip2VisionConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            3, self.embed_dim, config.patch_size, stride=config.patch_size, padding="valid"
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(new_height, new_width), mode="bicubic", align_corners=False
        )
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def forward(self, pixel_values: Tensor, interpolate_pos_encoding: bool = False) -> Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class _Siglip2Attention(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None, output_attentions: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim).contiguous()
        return self.out_proj(attn_output), None


class _Siglip2MLP(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = _Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _Siglip2MLP(config)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, ...]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_w = self.self_attn(hidden_states, attention_mask, output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,) + ((attn_w,) if output_attentions else ())


class _Siglip2Encoder(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([_Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        encoder_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        hidden_states = inputs_embeds
        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            out = layer(hidden_states, attention_mask, output_attentions)
            hidden_states = out[0]
            if output_attentions:
                all_attns = all_attns + (out[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attns)


class _Siglip2PoolingHead(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _Siglip2MLP(config)

    def forward(self, hidden_state: Tensor) -> Tensor:
        probe = self.probe.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        return (residual + self.mlp(hidden_state))[:, 0]


class Siglip2VisionTransformer(nn.Module):
    """Custom Siglip2 vision transformer matching HF Cheers modeling_umm.py."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = _Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_head = True
        self.head = _Siglip2PoolingHead(config)

    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions or False,
            output_hidden_states=output_hidden_states or False,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = self.head(last_hidden_state) if self.use_head else None
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ── UMMTextModel (custom Qwen2 with bool-mask SDPA) ───────────────
# Ported from HF Cheers modeling_umm.py.
# Key difference from standard transformers Qwen2ForCausalLM:
#   - UMMTextModel.forward() passes attention_mask directly to layers
#     (no _update_causal_mask transformation)
#   - Qwen2SdpaAttention converts mask to torch.bool before SDPA
#     (True = attend, False = mask)


@dataclass
class CheersQwen2Config:
    vocab_size: int = 152064
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    attention_dropout: float = 0.0
    pad_token_id: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> CheersQwen2Config:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _qwen2_apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


class _Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: CheersQwen2Config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class _Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _Qwen2SdpaAttention(nn.Module):
    """Qwen2 attention with bool-mask SDPA — matches Cheers' custom implementation."""

    def __init__(self, config: CheersQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = _Qwen2RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None, object | None]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = _qwen2_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # NaiveCache: read 3-D (seq, heads, dim), concat, write back.
        # SDPA needs 4-D (batch, heads, seq, dim), so we convert at the boundary.
        if past_key_value is not None:
            cached_k = past_key_value.key_cache[self.layer_idx]
            if cached_k is not None:
                # 3D→4D: (S, H, D) → (1, H, S, D)
                past_k = cached_k.unsqueeze(0).transpose(1, 2)
                past_v = past_key_value.value_cache[self.layer_idx].unsqueeze(0).transpose(1, 2)
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)

        if use_cache and past_key_value is not None:
            # 4D→3D: (1, H, S, D) → (S, H, D)
            past_key_value.key_cache[self.layer_idx] = key_states.squeeze(0).transpose(0, 1)
            past_key_value.value_cache[self.layer_idx] = value_states.squeeze(0).transpose(0, 1)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = attention_mask is None and q_len > 1
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask.to(torch.bool).to(value_states.device)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), None, past_key_value


class _Qwen2MLP(nn.Module):
    def __init__(self, config: CheersQwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: CheersQwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = _Qwen2SdpaAttention(config, layer_idx)
        self.mlp = _Qwen2MLP(config)
        self.input_layernorm = _Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_w, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_w,)
        if use_cache:
            outputs += (present_kv,)
        return outputs


class UMMTextModel(nn.Module):
    """Custom Qwen2 LLM backbone matching Cheers' UMMTextModel.

    Key difference from transformers Qwen2ForCausalLM:
      - forward() passes attention_mask directly to layers (no causal mask transform)
      - Attention converts mask to bool for SDPA
    """

    def __init__(self, config: CheersQwen2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([_Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = _Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _Qwen2RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        cache_position: Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = NaiveCache(len(self.layers))
        if cache_position is None:
            past_seen = _kv_seq_len(past_key_values)
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Pass attention_mask directly — no causal mask transformation
        causal_mask = attention_mask
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
