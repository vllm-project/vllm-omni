# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling helpers of AURA's chunk-wise generation: student noise levels and VAE decode spans."""

from __future__ import annotations

import torch


class StudentSigmas:
    """Noise level of each student timestep.

    The student's timesteps (1000, 745, 498, 255) are read on its teacher's flow-matching grid:
    ``steps`` sigmas from 1 down to ``sigma_min``, shifted by ``shift`` (sigma' = shift * sigma /
    (1 + (shift - 1) * sigma)), at timesteps 1000 * sigma'. A timestep takes the sigma of the
    nearest grid point."""

    def __init__(self, steps: int, shift: float, sigma_min: float = 0.003 / 1.002) -> None:
        grid = torch.linspace(1.0, sigma_min, steps, dtype=torch.float32)
        self.grid_sigmas = shift * grid / (1 + (shift - 1) * grid)
        self.grid_timesteps = 1000 * self.grid_sigmas

    def __call__(self, timestep: float) -> float:
        """Sigma of the grid point whose timestep is nearest to ``timestep``."""
        nearest = torch.argmin(torch.abs(self.grid_timesteps - timestep))
        return self.grid_sigmas[nearest].item()


def decode_spans(num_frames: int, group: int, first: bool) -> list[tuple[int, int]]:
    """Latent-frame ranges [start, end) of successive causal-VAE decoder calls for one chunk.

    The first latent frame of a video becomes a single video frame and is decoded on its own; the
    other latent frames are decoded ``group`` at a time."""
    spans: list[tuple[int, int]] = [(0, 1)] if first and num_frames > 0 else []
    start = spans[-1][1] if spans else 0
    while start < num_frames:
        end = min(start + group, num_frames)
        spans.append((start, end))
        start = end
    return spans
