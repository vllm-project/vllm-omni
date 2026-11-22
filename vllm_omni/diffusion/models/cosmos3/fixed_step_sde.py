# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-step SDE/ODE sampler shared by the distilled causal Cosmos3 students.

The imaginaire4 reference runs a few-step student on a flat ``[B, N]`` FP32
state: ``x0 = x - sigma * v`` then, for SDE, ``x = (1 - sigma') * x0 + sigma' *
eps`` where ``eps`` is drawn from a generator reseeded *per step* with
``seed + sample_index * 1_000_003 + (step + 1) * 9_176``. The initial noise uses
``seed + sample_index``. Reproducing those streams exactly is what makes
latent-level parity with the reference possible, so this module owns the seed
arithmetic and the update rule; pipelines supply the velocity closure and the
reshaping around it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import torch

SampleType = Literal["sde", "ode"]
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_STEP_SEED_SAMPLE_STRIDE = 1_000_003
_STEP_SEED_STEP_STRIDE = 9_176


def fixed_step_sigmas(t_list: Sequence[float]) -> tuple[float, ...]:
    """Validate a student schedule and append the terminal ``0.0``."""
    sigmas = [float(value) for value in t_list]
    if not sigmas:
        raise ValueError("fixed_step_sampler_config.t_list must not be empty.")
    if sigmas[-1] != 0.0:
        sigmas.append(0.0)
    if len(sigmas) < 2:
        raise ValueError("fixed_step_sampler_config.t_list must contain a nonzero sigma.")
    if any(not math.isfinite(s) or s < 0.0 or s > 1.0 for s in sigmas):
        raise ValueError(f"Fixed-step sigmas must lie in [0, 1], got {sigmas}.")
    if any(later >= earlier for earlier, later in zip(sigmas[:-1], sigmas[1:])):
        raise ValueError(f"Fixed-step sigmas must be strictly decreasing, got {sigmas}.")
    return tuple(sigmas)


def reference_step_seed(seed: int, sample_index: int, step_index: int) -> int:
    """Seed of the SDE reinjection noise at ``step_index`` (0-based) of sample ``sample_index``.

    Steps start at one so step zero never reuses the initial-noise seed of the
    same sample, and the sample stride keeps (sample, step) pairs from colliding.
    """
    return int(seed) + int(sample_index) * _STEP_SEED_SAMPLE_STRIDE + (int(step_index) + 1) * _STEP_SEED_STEP_STRIDE


def initial_noise(shape: Sequence[int], *, seed: int, sample_index: int, device: torch.device | str) -> torch.Tensor:
    """FP32 standard normal noise from ``Generator(device).manual_seed(seed + sample_index)``."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(sample_index))
    return torch.randn(tuple(shape), device=device, dtype=torch.float32, generator=generator)


def fixed_step_sample(
    velocity_fn: VelocityFn,
    noise: torch.Tensor,
    *,
    t_list: Sequence[float],
    num_train_timesteps: int,
    seed: int,
    sample_index: int,
    sample_type: SampleType = "sde",
) -> torch.Tensor:
    """Run the student schedule from ``noise`` (``[B, N]``) and return the clean state.

    ``velocity_fn(x, timestep)`` receives the FP32 state and a ``[B, 1]`` FP32
    timestep equal to ``sigma * num_train_timesteps``; its output is cast to
    FP32. The update is Euler on ``x0`` with per-step noise reinjection (SDE) or
    the plain velocity step (ODE).
    """
    if noise.ndim != 2:
        raise ValueError(f"Fixed-step sampling expects a flat [B, N] state, got {tuple(noise.shape)}.")
    if sample_type not in ("sde", "ode"):
        raise ValueError(f"Unsupported fixed-step sample_type {sample_type!r}; expected 'sde' or 'ode'.")
    if num_train_timesteps <= 0:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}.")
    sigmas = fixed_step_sigmas(t_list)
    x = noise.float()
    device = x.device
    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32, device=device)
    deltas = torch.tensor(
        [later - earlier for earlier, later in zip(sigmas[:-1], sigmas[1:])], dtype=torch.float32, device=device
    )
    for step_index, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        timestep = torch.full(
            (x.shape[0], 1), sigma_cur * float(num_train_timesteps), dtype=torch.float32, device=device
        )
        velocity = velocity_fn(x, timestep).float()
        if velocity.shape != x.shape:
            raise ValueError(f"velocity_fn returned {tuple(velocity.shape)} for state {tuple(x.shape)}.")
        x0 = x - sigma_tensor[step_index] * velocity
        if sigma_next == 0.0:
            x = x0
            continue
        if sample_type == "ode":
            x = x + deltas[step_index] * velocity
            continue
        generator = torch.Generator(device=device).manual_seed(reference_step_seed(seed, sample_index, step_index))
        eps = torch.empty_like(x).normal_(generator=generator)
        x = (1.0 - sigma_tensor[step_index + 1]) * x0 + sigma_tensor[step_index + 1] * eps
    return x


__all__ = ["fixed_step_sample", "fixed_step_sigmas", "initial_noise", "reference_step_seed"]
