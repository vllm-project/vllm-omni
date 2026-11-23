# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared flow-matching math for Pi-family action models."""

import math

import torch


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Embed a batch of scalar flow timesteps with sine/cosine frequencies.

    Float64 is intentional for the log-linear period sweep and inner products;
    it matches OpenPI's numerical behavior before callers cast the result to
    their model dtype.

    Ref: openpi/models_pytorch/pi0_pytorch.py
    ``create_sinusoidal_pos_embedding``.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be 1-D (batch_size,)")
    if device is None:
        device = time.device

    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_euler_schedule(num_steps: int) -> tuple[tuple[float, float], ...]:
    """Return ``(t, dt)`` pairs for integration from noise at 1 to data at 0.

    The denoiser is evaluated at ``1, 1-dt, ..., 1/num_steps``; the last Euler
    update lands at zero, so the denoiser is never evaluated at ``t=0``.
    """
    dt = -1.0 / num_steps
    return tuple((1.0 + step * dt, dt) for step in range(num_steps))


def euler_step(sample: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
    """Advance one explicit Euler step along the predicted flow velocity."""
    return sample + dt * velocity
