# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import torch
from torch import nn


def safe_linalg_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, rhs)
    except RuntimeError as exc:
        if matrix.device.type != "cpu" or "requires compiling PyTorch with LAPACK" not in str(exc):
            raise
        # Some ROCm wheels still fail here at runtime even when the LAPACK probe reports support.
        solved = np.linalg.solve(matrix.detach().cpu().numpy(), rhs.detach().cpu().numpy())
        return torch.from_numpy(solved).to(device=matrix.device, dtype=matrix.dtype)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (B, T, embedding_dim)
        return enc
