# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 Fourier embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


def _frequency_bands(
    num_bands: int,
    *,
    temperature: float,
    device: torch.device | str | None,
) -> torch.Tensor:
    exponent = torch.arange(num_bands, dtype=torch.float32, device=device) / num_bands
    return 1.0 / temperature**exponent


class ElementWiseFourierEmbed(nn.Module):
    """Nine-coordinate Fourier embedding used as MAGI's element-wise RoPE."""

    def __init__(
        self,
        dim: int,
        *,
        temperature: float = 10000.0,
        learnable: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        bands = _frequency_bands(dim // 8, temperature=temperature, device=device).to(dtype)
        if learnable:
            self.bands = nn.Parameter(bands)
        else:
            # ``bands`` is part of the released checkpoint. Keep it in the
            # parameter namespace so the generic DLO mmap loader can bind it
            # without first materializing a full CPU transformer.
            self.bands = nn.Parameter(bands, requires_grad=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 9:
            raise ValueError("MAGI coordinates must have shape [tokens,9]")
        xyz = coords[:, :3]
        sizes = coords[:, 3:6]
        references = coords[:, 6:9]
        scales = (references - 1) / (sizes - 1)
        scales = torch.where((references == 1) & (sizes == 1), torch.ones_like(scales), scales)
        if not torch.isfinite(scales).all():
            raise ValueError("invalid MAGI coordinate scale")
        centers = (sizes - 1) / 2
        centers = centers.clone()
        centers[:, 0] = 0
        projection = (xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((projection.sin(), projection.cos()), dim=1).flatten(1)


__all__ = ["ElementWiseFourierEmbed"]
