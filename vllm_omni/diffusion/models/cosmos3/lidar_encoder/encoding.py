# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""Coordinate helpers for the LiDAR TransformerVAE.

Only polar range-map coordinates are used (RoPE / neighborhood attention).
Alternate absolute PE modes (spherical harmonics, Fourier features) are not
part of the shipped checkpoint and were removed.
"""

from __future__ import annotations

import torch


def generate_polar_coords(H: int, W: int, device: torch.device | str = "cpu") -> torch.Tensor:  # returns [1,2,H,W]
    """Build polar angles for a range map.

    theta: azimuthal angle in [-pi, pi]
    phi: polar angle in [0, pi]
    """
    phi = (0.5 - torch.arange(H, device=device) / H) * torch.pi  # [H]
    theta = (1 - torch.arange(W, device=device) / W) * 2 * torch.pi - torch.pi  # [W]
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")  # [H,W], [H,W]
    angles = torch.stack([phi, theta])  # [2,H,W]
    return angles[None]  # [1,2,H,W]
