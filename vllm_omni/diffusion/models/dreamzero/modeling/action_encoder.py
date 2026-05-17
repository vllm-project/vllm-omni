# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Action encoder/decoder for DreamZero."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    """swish activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding: (B, T) timesteps → (B, T, dim)"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """Per-category linear: W[cat_id] @ x + b[cat_id]

    Attributes:
        W: (num_categories, input_dim, hidden_dim)  — note: 0.02 * randn init
        b: (num_categories, hidden_dim)              — zero init
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP: layer1 (relu) → layer2"""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Encode actions with embodiment-specific weights + sinusoidal timestep.

    Flow: actions → W1 → concat(a_emb, pos_enc(timesteps)) → W2 (swish) → W3

    Args:
        action_dim: action vector dimension (e.g. 32)
        hidden_size: output/hidden dimension (e.g. 5120 = model dim)
        num_embodiments: number of robot types (e.g. 32)
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions:   (B, T, action_dim)
            timesteps: (B, T) — per-token timestep
            cat_ids:   (B,)   — embodiment id per sample
        Returns:
            (B, T, hidden_size)
        """
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x
