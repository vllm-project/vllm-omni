# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared timestep embedding primitives for diffusion models.

- ``SinusPositionEmbedding``: sin/cos positional encoding (TTS DiT/CFM).
  Used by Qwen3-TTS, Qwen2.5-Omni, CosyVoice3, Ming-Flash-Omni.
- ``DiTTimestepEmbedding``: SinusPosEmb + Linear + SiLU + Linear MLP.
  Used by the same models as above.
- ``timestep_embedding()``: standalone function (GLIDE/DiT convention).
  Used by Bagel, NextStep, Z-Image, HunyuanImage3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (GLIDE/DiT convention).

    Produces cos-then-sin embeddings with log-spaced frequencies.

    Args:
        t: (N,) 1-D tensor of timestep indices (may be fractional).
        dim: Output embedding dimension.
        max_period: Controls the minimum frequency.

    Returns:
        (N, dim) tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for scalar timesteps.

    Maps scalar timestep values to ``dim``-dimensional embeddings using
    the standard log-spaced frequency formula from DDPM/DiT.

    Args:
        dim: Output embedding dimension (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        """
        Args:
            x: (N,) scalar timesteps.
            scale: Frequency scaling factor.

        Returns:
            (N, dim) sinusoidal embeddings, cast to the input dtype.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(x.dtype)


class DiTTimestepEmbedding(nn.Module):
    """Timestep conditioning: SinusPositionEmbedding + Linear + SiLU + Linear.

    Args:
        dim: Hidden dimension (output size).
        freq_embed_dim: Sinusoidal embedding dimension (input to MLP).
    """

    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)
