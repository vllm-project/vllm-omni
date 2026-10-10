# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Coordinate helpers for the LiDAR TransformerVAE.

Only polar range-map coordinates are used (RoPE / neighborhood attention).
Alternate absolute PE modes (spherical harmonics, Fourier features) are not
part of the shipped checkpoint and were removed.
"""

from __future__ import annotations

import torch


def generate_polar_coords(H: int, W: int, device: torch.device | str = "cpu") -> torch.Tensor:  # returns [1,2,H,W]
    """Build spherical angles for a range map.

    phi: elevation in (-pi/2, pi/2], decreasing with row; row 0 is +pi/2.
    theta: azimuth in (-pi, pi], decreasing with column; column 0 is +pi.
    Angles span the model width linearly (padded columns are not wrapped),
    matching the reference buffer stored in the checkpoint.
    """
    phi = (0.5 - torch.arange(H, device=device) / H) * torch.pi  # [H]
    theta = (1 - torch.arange(W, device=device) / W) * 2 * torch.pi - torch.pi  # [W]
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")  # [H,W], [H,W]
    angles = torch.stack([phi, theta])  # [2,H,W]
    return angles[None]  # [1,2,H,W]
