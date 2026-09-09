# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The vLLM-Omni team.
# Copyright (c) rednote-hilab. All rights reserved.
# Adapted from:
# https://github.com/rednote-hilab/dots.tts/tree/a393d2e/src/dots_tts/modules/backbone
#
# Merged from two upstream files (preserved as-is, inference behaviour
# unchanged; only internal imports were folded inline):
#   - modules/backbone/layers.py  -> Dropout, Mlp, rotate_half,
#                                    apply_rotary_pos_emb, RotaryEmbedding,
#                                    MultiHeadAttention
#                                    (Conv1d / ConvTranspose1d are NOT
#                                    re-vendored here — they already live in
#                                    dots_tts_vocoder.py)
#   - modules/backbone/dit.py     -> modulate, TimestepEmbedder, FinalLayer,
#                                    DiTBlock, DiT
#
# Per-section upstream attributions are preserved at each section header.
# Nothing in this file diverges from upstream behaviour.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ============================================================================
# Building blocks (adapted from modules/backbone/layers.py)
# ----------------------------------------------------------------------------
# Conv1d / ConvTranspose1d from layers.py are already vendored in
# dots_tts_vocoder.py (they are used by the BigVGAN-style AudioVAE).  The
# pieces below are the ones consumed by the DiT flow-matching head.
# ============================================================================


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False, force_drop: bool = False, **_kwargs):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x, **_kwargs):
        return F.dropout(
            x,
            p=self.p,
            training=True if self.force_drop else self.training,
            inplace=self.inplace,
        )


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size=4096,
        act_layer=nn.GELU,
        dropout=0.0,
        **_kwargs,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x, _mask=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.autocast(enabled=False, device_type="cuda")
def apply_rotary_pos_emb(pos, t):
    if pos.dim() == 3:
        pos = pos.unsqueeze(1)
    return t * pos.cos() + rotate_half(t) * pos.sin()


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self._theta = float(theta)

    def _apply(self, fn):
        inv_freq = self.inv_freq
        super()._apply(fn)
        self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    @torch.autocast(enabled=False, device_type="cuda")
    def forward(self, t):
        inv_freq = self.inv_freq
        if inv_freq.device != t.device:
            raise RuntimeError(f"RotaryEmbedding buffer device mismatch: inv_freq={inv_freq.device} input={t.device}.")
        t = t.to(dtype=inv_freq.dtype)
        if t.dim() == 1:
            freqs = torch.einsum("i , j -> i j", t, inv_freq)
        else:
            freqs = torch.einsum("bi, j -> bij", t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        dropout: float = 0.0,
        norm_layer: str = "LayerNorm",
        rotary_bias: bool = False,
        rotary_theta: float | None = 50000,
        **_kwargs,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rotary_bias = rotary_bias

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        norm_layer = getattr(nn, norm_layer)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

        if self.rotary_bias:
            self.rotary = RotaryEmbedding(self.head_dim, theta=rotary_theta)

    def forward(self, q, k=None, v=None, mask=None, pos_ids=None, **_kwargs):
        k = k or q
        v = v or q
        B, L, _ = q.shape
        _, S, _ = v.shape
        if mask is not None:
            if mask.ndim == 2:  # [B, L]
                assert L == S
                mask = rearrange(mask, "b j -> b 1 1 j")
                mask = mask.expand(-1, self.num_heads, L, -1)
            elif mask.ndim == 3:  # [B, L, S]
                assert mask.size(1) == L and mask.size(2) == S
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply rotary
        if self.rotary_bias:
            if L == S:
                if pos_ids is None:
                    rotary_emb = self.rotary(torch.arange(L, device=q.device))
                else:
                    rotary_emb = self.rotary(pos_ids)
                q, k = (apply_rotary_pos_emb(rotary_emb, tensor) for tensor in (q, k))
            else:
                q_rotary_emb = self.rotary(torch.arange(L, device=q.device))
                k_rotary_emb = self.rotary(torch.arange(S, device=k.device))
                q = apply_rotary_pos_emb(q_rotary_emb, q)
                k = apply_rotary_pos_emb(k_rotary_emb, k)

        attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)

        if mask is not None:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_dropout(self.o_proj(out))

    def decode_step(self, x, *, cache, positions: torch.Tensor):
        if x.size(1) <= 0:
            raise ValueError("MultiHeadAttention.decode_step expects a non-empty input.")
        if positions.ndim != 1 or positions.size(0) != x.size(1):
            raise ValueError("MultiHeadAttention.decode_step positions must match the decode block length.")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        block_len = q.size(2)

        if self.rotary_bias:
            rotary_emb = self.rotary(positions)
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        cached_k, cached_v = cache
        cached_k.index_copy_(2, positions, k)
        cached_v.index_copy_(2, positions, v)

        cache_capacity = cached_k.size(2)
        key_positions = torch.arange(
            cache_capacity,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0)
        query_positions = positions.unsqueeze(1)
        causal_mask = key_positions <= query_positions
        valid_mask = key_positions <= positions[-1]
        attn_bias = torch.zeros(
            q.size(0),
            self.num_heads,
            block_len,
            cache_capacity,
            dtype=q.dtype,
            device=q.device,
        )
        attn_bias.masked_fill_(
            (causal_mask & valid_mask).unsqueeze(0).unsqueeze(0).logical_not(),
            float("-inf"),
        )

        out = F.scaled_dot_product_attention(
            q,
            cached_k,
            cached_v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_dropout(self.o_proj(out)), cache


# ============================================================================
# DiT flow-matching head (adapted from modules/backbone/dit.py)
# ----------------------------------------------------------------------------
# DiT.__init__ takes a ``transformer_config`` object exposing ``.to_dict()``,
# ``.hidden_size`` and ``.num_layers``.  Upstream constructs it from a pydantic
# config class higher up the stack; vLLM-Omni callers can pass anything that
# satisfies the same minimal interface (e.g. a stdlib @dataclass with a
# ``to_dict`` method or SimpleNamespace).  See dots_tts_talker.py for the
# concrete soar-checkpoint config factory.
# ============================================================================


def modulate(x, shift, scale, **_kwargs):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, c, **_kwargs):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        hidden_size: int = 1024,
        modulation: bool = False,
        eps: float = 1e-5,
        **_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.attn = attention
        self.ffn = ffn
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, condition=None, mask=None, **kwargs):
        if condition is None:
            assert not self.modulation, "Without global condition, must set modulation to False"
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(condition).chunk(
                6, dim=1
            )

        if condition is not None:
            pack_indices = kwargs.get("pack_indices")
            if pack_indices is not None:
                gate_attn = gate_attn[pack_indices]
                gate_ffn = gate_ffn[pack_indices]
            else:
                gate_attn = gate_attn.unsqueeze(1)
                gate_ffn = gate_ffn.unsqueeze(1)

        if condition is not None:
            x = x + gate_attn * self.attn(
                modulate(self.norm1(x), shift_attn, scale_attn, **kwargs),
                mask=mask,
                **kwargs,
            )
        else:
            x = x + self.attn(self.norm1(x), mask=mask, **kwargs)

        if condition is not None:
            x = x + gate_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn, **kwargs))
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        transformer_config,
        *,
        mode: str = "flow_matching",
    ):
        super().__init__()
        if mode not in {"flow_matching", "meanflow"}:
            raise ValueError(f"DiT mode must be 'flow_matching' or 'meanflow', got {mode!r}.")

        transformer_kwargs = transformer_config.to_dict()
        model_dim = transformer_config.hidden_size
        self.mode = mode
        self.num_layers = transformer_config.num_layers

        self.input_layer = nn.Linear(in_dim, model_dim)
        self.time_embedder = TimestepEmbedder(model_dim)
        if mode == "meanflow":
            self.duration_embedder = TimestepEmbedder(model_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            attn_block = MultiHeadAttention(**transformer_kwargs, name=f"layer_{i}")
            ffn_block = Mlp(act_layer=lambda: nn.GELU(approximate="tanh"), **transformer_kwargs)
            self.blocks.append(DiTBlock(attention=attn_block, ffn=ffn_block, **transformer_kwargs))

        self.output_layer = FinalLayer(model_dim, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.linear.weight, 0)
        nn.init.constant_(self.output_layer.linear.bias, 0)

    def forward(
        self,
        x,
        timesteps,
        duration: torch.Tensor | None = None,
        mask=None,
        attn_mask=None,
        g_cond: torch.Tensor | None = None,
        **kwargs,
    ):
        t = self.time_embedder(timesteps)
        c = t
        duration_embedder = getattr(self, "duration_embedder", None)
        if duration_embedder is not None and duration is not None:
            c = c + duration_embedder(duration)
        if g_cond is not None:
            c = c + g_cond

        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, c, mask=attn_mask, **kwargs)
        return self.output_layer(x, c, **kwargs)
