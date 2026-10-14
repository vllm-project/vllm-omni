from __future__ import annotations

import math

import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """Shared Gaussian Fourier features with optional trainable frequencies."""

    def __init__(
        self,
        *,
        in_features: int,
        embedding_size: int,
        scale: float = 1.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.embedding_size = int(embedding_size)
        weight = torch.randn(self.embedding_size, self.in_features) * scale
        self.weight = nn.Parameter(weight, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.in_features)
        x_proj = 2 * math.pi * x @ self.weight.T
        return torch.cat([x_proj.cos(), x_proj.sin()], dim=-1)
