# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rectified-flow Euler matching Diffusers FlowMatchEulerDiscreteScheduler."""

from __future__ import annotations

import torch


class FlowEuler:
    """Faithful replica of Diffusers ``FlowMatchEulerDiscreteScheduler(shift=...)``.

    Diffusers 0.40 builds the shifted grid from an already-shifted ``sigma_min``
    endpoint (``__init__`` reads ``sigma_min`` from the warped schedule), so the
    linspace endpoint is effectively warped twice.
    """

    def __init__(self, steps: int, shift: float = 1.0) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        if shift <= 0:
            raise ValueError("shift must be positive")

        def warp(sigma: torch.Tensor) -> torch.Tensor:
            return shift * sigma / (1 + (shift - 1) * sigma)

        shifted = warp(torch.linspace(1.0, float(warp(torch.tensor(1.0 / 1000))), steps, dtype=torch.float64))
        self.sigmas = [*shifted.float().tolist(), 0.0]
        self.timesteps = [1000.0 * sigma for sigma in self.sigmas[:-1]]

    def advance(self, prediction: torch.Tensor, sample: torch.Tensor, step: int) -> torch.Tensor:
        delta = self.sigmas[step + 1] - self.sigmas[step]
        return (sample.float() + delta * prediction.float()).to(sample.dtype)
