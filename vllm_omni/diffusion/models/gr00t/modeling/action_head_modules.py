# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Embodiment-conditioned MLP primitives for GR00T-N1.7.

Ported from
``Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py``.  Parameter
names (``W``, ``b``, ``layer1``, ``layer2``, ``W1``, ``W2``, ``W3``,
``pos_encoding.*``) are preserved exactly so upstream checkpoint tensors load
cleanly via ``load_state_dict``.

Training-only ``expand_action_dimension`` helpers are dropped — inference path
only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape ``(B, T, embedding_dim)`` from timesteps
    of shape ``(B, T)``.

    Used inside :class:`MultiEmbodimentActionEncoder` for action-step time
    embedding.  This is a parameter-free module so it has no weights to load.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        _, _ = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """Linear layer with per-category weights and biases.

    Parameter shapes follow upstream exactly:
      ``W`` : ``(num_categories, input_dim, hidden_dim)``
      ``b`` : ``(num_categories, hidden_dim)``
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights per embodiment."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal timestep
    positional encoding.

    Architecture mirrors upstream's ``MultiEmbodimentActionEncoder``:
      W1 : action_dim → hidden_size  (per-embodiment)
      W2 : 2*hidden_size → hidden_size  (per-embodiment, after concat with τ)
      W3 : hidden_size → hidden_size  (per-embodiment)
      pos_encoding : sinusoidal timestep embedding
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = actions.shape

        if timesteps.dim() != 1 or timesteps.shape[0] != B:
            raise ValueError(
                "Expected `timesteps` shape (B,) so it can broadcast across T."
            )
        timesteps = timesteps.unsqueeze(1).expand(-1, T)

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x
