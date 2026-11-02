# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time-curve coordinates for compressed ComfyUI-layout H3 AdaLN weights."""

import torch
from torch import nn


class MiniMaxH3AdalnCurve(nn.Module):
    """Interpolate stored coordinates, without reconstructing dense embeddings.

    Matches ComfyUI's H3 curve path: table rows span normalized time [0, 1],
    endpoints clamp, and both interpolation and the following projection use
    FP32. Coordinates already represent the activated embedding, so the
    projection must not apply SiLU again.
    """

    def __init__(self, grid: int, width: int) -> None:
        super().__init__()
        if grid < 2 or width < 1:
            raise ValueError("H3 AdaLN curve requires at least two rows and a positive coordinate width")
        self.table = nn.Parameter(torch.empty(grid, width, dtype=torch.float32), requires_grad=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        positions = timesteps.float().clamp(0.0, 1.0) * (self.table.shape[0] - 1)
        lower = positions.floor().long().clamp(max=self.table.shape[0] - 2)
        return torch.lerp(self.table[lower], self.table[lower + 1], (positions - lower).unsqueeze(-1))
