from __future__ import annotations

import torch
from torch import nn


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        if isinstance(hidden_size, tuple):
            if len(hidden_size) != 1:
                raise ValueError(f"Qwen2RMSNorm expects 1D shape, got {hidden_size}")
            hidden_size = int(hidden_size[0])
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).to(dtype)

