# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp32 RoPE + MLP matching native VoxCPM2 numerics.

Exports: _MiniCPMLongRoPE, _MiniCPMMLP, _apply_rotary_pos_emb
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
#  Primitives
# ===================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings in float32."""
    orig_dtype = q.dtype
    q, k = q.to(torch.float32), k.to(torch.float32)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


# ===================================================================
#  LongRoPE — must match native computation order exactly
# ===================================================================


class _MiniCPMLongRoPE(nn.Module):
    """LongRoPE matching native computation order."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: int | None,
        rope_theta: float,
        max_position_embeddings: int,
        rope_scaling: dict[str, Any],
    ) -> None:
        super().__init__()
        self.dim = kv_channels if kv_channels else hidden_size // num_attention_heads
        self.base = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.short_factor = rope_scaling["short_factor"]
        self.long_factor = rope_scaling["long_factor"]
        self.original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        ext_factors = torch.tensor(
            self.long_factor if seq_len > self.original_max_position_embeddings else self.short_factor,
            dtype=torch.float32,
            device=device,
        )

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype) * self.scaling_factor
        self.sin_cached = emb.sin().to(dtype) * self.scaling_factor

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


# ===================================================================
#  MLP
# ===================================================================


class _MiniCPMMLP(nn.Module):
    """SiLU-gated MLP matching native MiniCPMMLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
