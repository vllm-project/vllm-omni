# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned interval-periodic placement, preserving native Q128/K64 blocks."""

from dataclasses import dataclass
from math import gcd

import torch


@dataclass(frozen=True)
class SparSPVideoLayout:
    grid: tuple[int, int, int]
    degree: int
    permutation: torch.Tensor
    inverse: torch.Tensor


def supports_interval_periodic_layout(grid: tuple[int, int, int], degree: int) -> bool:
    """Whether whole native Q/K blocks can be placed in equal rank shards."""
    frames, height, width = grid
    if min(frames, height, width, degree) < 1:
        raise ValueError("SparSP video geometry and degree must be positive")
    spatial = height * width
    period_frames = 128 // gcd(spatial, 128)
    return frames % period_frames == 0 and frames * spatial % (degree * 128) == 0


def interval_periodic_layout(grid: tuple[int, int, int], degree: int, device: torch.device) -> SparSPVideoLayout:
    frames, height, width = grid
    tokens, spatial = frames * height * width, height * width
    # Group the minimum number of frames that forms whole Q128 units.
    # Equal rank shards must preserve both Q128 and K64 memberships.
    if not supports_interval_periodic_layout(grid, degree):
        raise ValueError(
            "SparSP placement requires complete Q128 frame periods and equally sized Q128-aligned rank shards"
        )
    period_frames = 128 // gcd(spatial, 128)
    units, period = tokens // 128, spatial * period_frames // 128
    block = torch.arange(units, device=device)
    # Same spatial interval across frames, in stable original-time order.
    # Equal rank slices may share one interval when period % degree != 0.
    order = torch.argsort((block % period) * units + block)
    permutation = (order[:, None] * 128 + torch.arange(128, device=device)).flatten()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(tokens, device=device)
    return SparSPVideoLayout(grid, degree, permutation, inverse)


def resolve_video_layout(
    grid: tuple[int, int, int],
    degree: int,
    device: torch.device,
    placement: str,
    cached: SparSPVideoLayout | None,
) -> SparSPVideoLayout | None:
    """Resolve an opt-in model's placement before splitting the video sequence."""
    tokens = grid[0] * grid[1] * grid[2]
    if tokens == 0 or tokens % (64 * degree):
        raise ValueError("SparSP requires complete unpadded K64 blocks on each Ring rank")
    if placement == "none" or (placement == "auto" and not supports_interval_periodic_layout(grid, degree)):
        return None
    if cached is not None and cached.grid == grid and cached.degree == degree and cached.permutation.device == device:
        return cached
    return interval_periodic_layout(grid, degree, device)
