# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matcha-TTS's U-Net building blocks that S3Gen's causal blocks extend."""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(torch.nn.Module):
    """Sinusoidal embedding of a diffusion time.

    Args:
        dim: Embedding width, even: half sines, half cosines.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        """Embed times ``x``, shape (B,) or a scalar, as shape (B, dim)."""
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    """Convolution, GroupNorm and Mish over the masked frames.

    Args:
        dim: Input channels.
        dim_out: Output channels.
        groups: GroupNorm groups.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        """Shape (B, dim, T) to (B, dim_out, T), zero outside ``mask``."""
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """Two blocks with the time embedding added between them, and a residual projection.

    Args:
        dim: Input channels.
        dim_out: Output channels.
        time_emb_dim: Width of the time embedding.
        groups: GroupNorm groups.
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        """Shape (B, dim, T) and a (B, time_emb_dim) embedding to (B, dim_out, T)."""
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output
