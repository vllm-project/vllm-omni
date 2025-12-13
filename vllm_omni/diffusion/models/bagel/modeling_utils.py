# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Bagel repository to avoid importing external Bagel package.

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class TimestepEmbedder(nn.Module):
    """Match upstream Bagel (DiT-style) timestep embedder: 256-d freq -> hidden -> hidden."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PositionEmbedding(nn.Module):
    """Match upstream Bagel: fixed 2D sin-cos embedding stored in `pos_embed`."""

    def __init__(self, max_size: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = max_size
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(torch.zeros(max_size * max_size, hidden_size), requires_grad=False)
        self._init_weights()

    def _init_weights(self) -> None:
        pos_embed = _get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU() if "gelu" in act else nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
