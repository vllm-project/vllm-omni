# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.2 DiT blocks and the two FastWAM experts (video and action).

Inference subset of ``wan_video_dit.py`` and ``action_dit.py``. The float math is
kept op-for-op identical to the reference eager path so a port can be compared
bitwise: RMSNorm is computed in float32 and rounded before the weight multiply,
RoPE runs in complex128, sinusoidal embeddings in float64, and attention is
``F.scaled_dot_product_attention`` with the same boolean masks the reference
passes (they are all-True at inference, but they select the same SDPA backend).
"""

# Adapted from FastWAM (https://github.com/yuantianyuan01/FastWAM @7faa711,
# src/fastwam/models/wan22/). The original portions carry the following MIT license:
# Copyright (c) 2026 The FastWAM Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """1-D RoPE table (complex128), always built on CPU like the reference."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device="cpu")[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device="cpu"), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    x_heads = x.view(batch_size, seq_len, num_heads, -1)
    x_out = torch.view_as_complex(x_heads.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2))
    return torch.view_as_real(x_out * freqs).flatten(2).to(x.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, mask: torch.Tensor | None
) -> torch.Tensor:
    # Plain SDPA with a boolean mask, as in the reference; vLLM-Omni's Attention
    # layer is not used because MoT attention mixes two experts' key/value sets
    # under a block mask and the reference numerics are the parity target.
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(dtype) * self.weight


class _Attention(nn.Module):
    """Q/K/V/O projections with full-width RMSNorm on q and k (shared by self- and cross-attention)."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, attn_hidden_dim)
        self.o = nn.Linear(attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(attn_hidden_dim, eps=eps)


class CrossAttention(_Attention):
    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        return self.o(attention(q, k, v, self.num_heads, ctx_mask))


class FastWAMBlock(nn.Module):
    """One transformer layer of either expert, evaluated the way FastWAM's MoT runs it.

    ``forward`` is the whole layer: modulated self-attention whose keys/values are
    optionally prefixed with another expert's cached keys/values (the action
    expert attends to the video expert's first-frame K/V), ungated cross-attention
    to the text+proprio context, and the gated GELU FFN. It returns the updated
    tokens plus this layer's post-RoPE keys and values so the video prefill can
    cache them. Being one ``nn.Module.forward``, the layer is the unit vLLM-Omni's
    regional ``torch.compile`` wraps (see ``FastWAMModel._repeated_blocks``).
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = _Attention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, hidden_dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        self_attn_mask: torch.Tensor,
        prefix_k: torch.Tensor | None = None,
        prefix_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # t_mod is [B, S, 6, D] (one modulation per token, video) or [B, 6, D] (action).
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        mods = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            mods = tuple(m.squeeze(2) for m in mods)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods

        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        q = self.self_attn.norm_q(self.self_attn.q(attn_input))
        k = self.self_attn.norm_k(self.self_attn.k(attn_input))
        v = self.self_attn.v(attn_input)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if prefix_k is not None and prefix_v is not None:
            k_all = torch.cat([prefix_k, k], dim=1)
            v_all = torch.cat([prefix_v, v], dim=1)
        else:
            k_all, v_all = k, v
        mixed = attention(q, k_all, v_all, self.num_heads, self_attn_mask)

        x = x + gate_msa * self.self_attn.o(mixed)
        x = x + self.cross_attn(self.norm3(x), context, context_mask.unsqueeze(1))
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, k, v


class VideoHead(nn.Module):
    """Wan2.2 output head. Kept so the checkpoint loads strictly; action-only
    inference never predicts future video, so it is never called."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)


class VideoExpert(nn.Module):
    """Wan2.2-TI2V-5B DiT used as FastWAM's video expert (first-frame prefill only)."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        in_dim: int,
        out_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_dim = freq_dim
        self.patch_size: tuple[int, int, int] = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [FastWAMBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = VideoHead(hidden_dim, out_dim, self.patch_size)
        self.freqs: tuple[torch.Tensor, ...] = precompute_freqs_cis_3d(attn_head_dim)

    def _apply(self, fn, recurse=True):  # type: ignore[override]
        # The RoPE tables are complex128 constants, not buffers: follow device
        # moves but never dtype casts.
        result = super()._apply(fn, recurse)
        device = self.patch_embedding.weight.device
        self.freqs = tuple(f.to(device=device) for f in self.freqs)
        return result

    def get_freqs(self, f: int, h: int, w: int) -> torch.Tensor:
        freq_f, freq_h, freq_w = self.freqs
        return torch.cat(
            [
                freq_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

    def prepare(
        self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Embed first-frame latents [B, C, F, H, W] into tokens; returns
        (tokens, t_mod, context_emb, context_mask, freqs, tokens_per_frame)."""
        batch_size = x.shape[0]
        tokens_per_frame = (x.shape[3] // self.patch_size[1]) * (x.shape[4] // self.patch_size[2])
        # Separated per-token timesteps; the conditioning (first) frame is clean (t=0).
        token_timesteps = torch.ones(
            (batch_size, x.shape[2], tokens_per_frame), dtype=timestep.dtype, device=timestep.device
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        context = self.text_embedding(context)
        context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)
        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x_tokens, t_mod, context, context_mask, self.get_freqs(f, h, w), tokens_per_frame


class ActionExpert(nn.Module):
    """FastWAM action expert (``ActionDiT``): denoises ``action_token_dim``-wide tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.freq_dim = freq_dim
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [FastWAMBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

    def _apply(self, fn, recurse=True):  # type: ignore[override]
        result = super()._apply(fn, recurse)
        self.freqs = self.freqs.to(device=self.head.weight.device)
        return result

    def prepare(
        self, action_tokens: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (tokens, t_mod, context_emb, context_mask, freqs)."""
        seq_len = action_tokens.shape[1]
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1)
        return tokens, t_mod, context_emb, context_mask, freqs


class FastWAMMoT(nn.Module):
    """Mixture of the video and action experts (state-dict prefix ``mixtures.``)."""

    def __init__(self, video: VideoExpert, action: ActionExpert) -> None:
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        if len(video.blocks) != len(action.blocks):
            raise ValueError("Video and action experts must have the same number of layers.")

    @property
    def video(self) -> VideoExpert:
        return self.mixtures["video"]  # type: ignore[return-value]

    @property
    def action(self) -> ActionExpert:
        return self.mixtures["action"]  # type: ignore[return-value]

    def prefill_video_cache(
        self,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the video expert once over the first frame; return per-layer post-RoPE K and V."""
        cache_k: list[torch.Tensor] = []
        cache_v: list[torch.Tensor] = []
        x = tokens
        for block in self.video.blocks:
            x, k, v = block(x, t_mod, freqs, context, context_mask, attention_mask)
            cache_k.append(k)
            cache_v.append(v)
        return cache_k, cache_v

    def denoise_action(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One action-expert evaluation (velocity prediction) against the video cache."""
        expert = self.action
        x, t_mod, context_emb, ctx_mask, freqs = expert.prepare(latents, timestep, context, context_mask)
        for layer_idx, block in enumerate(expert.blocks):
            x, _, _ = block(
                x, t_mod, freqs, context_emb, ctx_mask, attention_mask, cache_k[layer_idx], cache_v[layer_idx]
            )
        return expert.head(x)
