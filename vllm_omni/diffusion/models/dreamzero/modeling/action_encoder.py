# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Action encoder/decoder for DreamZero.

Adapted from:
- CategorySpecificLinear/MLP/MultiEmbodimentActionEncoder:
    dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py L31-90
- SinusoidalPositionalEncoding/swish:
    dreamzero/groot/vla/model/n1_5/modules/action_encoder.py L1-41
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    """swish activation: x * sigmoid(x)
    Source: action_encoder.py L6-7
    """
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding: (B, T) timesteps → (B, T, dim)

    Source: action_encoder.py L10-40
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Source: action_encoder.py L20-40
        timesteps = timesteps.float()  # L23: ensure float
        half_dim = self.embedding_dim // 2  # L28
        exponent = -torch.arange(  # L30-32
            half_dim, dtype=torch.float, device=timesteps.device
        ) * (torch.log(torch.tensor(10000.0)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # L34: (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # L36-38: (B, T, dim)


class CategorySpecificLinear(nn.Module):
    """Per-category linear: W[cat_id] @ x + b[cat_id]

    Source: wan_video_dit_action_casual_chunk.py L31-42
    Params:
        W: (num_categories, input_dim, hidden_dim)  — note: 0.02 * randn init
        b: (num_categories, hidden_dim)              — zero init
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # Source: wan_video_dit_action_casual_chunk.py L39-42
        selected_W = self.W[cat_ids]  # L40: (B, input_dim, hidden_dim)
        selected_b = self.b[cat_ids]  # L41: (B, hidden_dim)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)  # L42: (B, T, hidden_dim)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP: layer1 (relu) → layer2

    Source: wan_video_dit_action_casual_chunk.py L45-54
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # Source: wan_video_dit_action_casual_chunk.py L52-54
        hidden = F.relu(self.layer1(x, cat_ids))  # L53
        return self.layer2(hidden, cat_ids)  # L54


class MultiEmbodimentActionEncoder(nn.Module):
    """Encode actions with embodiment-specific weights + sinusoidal timestep.

    Source: wan_video_dit_action_casual_chunk.py L57-90
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
        # Source: wan_video_dit_action_casual_chunk.py L69-90
        a_emb = self.W1(actions, cat_ids)  # L79: (B, T, hidden_size)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)  # L82: (B, T, hidden_size)
        x = torch.cat([a_emb, tau_emb], dim=-1)  # L85: (B, T, 2*hidden_size)
        x = swish(self.W2(x, cat_ids))  # L86: (B, T, hidden_size)
        x = self.W3(x, cat_ids)  # L89: (B, T, hidden_size)
        return x
