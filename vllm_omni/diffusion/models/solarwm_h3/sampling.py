# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SolarWM, Copyright (c) SolarWM contributors.
# Source: Junchao-cs/SolarWM@ce1da4e7705391eda8eeda6016c0fd3f614b975e
"""Four stochastic SGF evaluations followed by the clean KV commit forward."""

from __future__ import annotations

from collections.abc import Callable

import torch

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
ANCHOR_TIMESTEP = 0.999
DENOISING_STEPS = 4
CHUNK_LATENTS = 5
HISTORY_CHUNKS = 5


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("H3 sigma shift must be positive")
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def stage2_timesteps(device: torch.device | str | None = None) -> torch.Tensor:
    """Native data-ward times, excluding the terminal clean time."""
    base_sigma = torch.linspace(1.0, 0.0, DENOISING_STEPS + 1, dtype=torch.float32, device=device)
    return 1.0 - shift_sigma(base_sigma, VIDEO_SHIFT)[:-1]


def scale_noise(clean: torch.Tensor, noise: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
    """H3 convention: native ``t=1`` is clean and velocity is ``clean-noise``."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise shapes must match")
    time = torch.as_tensor(timestep, device=clean.device, dtype=clean.dtype)
    while time.ndim < clean.ndim:
        time = time.unsqueeze(-1)
    return time * clean + (1.0 - time) * noise


def sample_audio_timestep(
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample once per rollout, then retain this time and noised silence."""
    return (
        1.0 - shift_sigma(torch.rand((1,), device=device, dtype=torch.float32, generator=generator), AUDIO_SHIFT)
    ).reshape(())


@torch.no_grad()
def sample_chunk(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    current_noise: torch.Tensor,
    *,
    commit: Callable[[torch.Tensor, torch.Tensor], object],
    generator: torch.Generator | None = None,
    chunk_index: int = 0,
    noise_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Generate one five-latent chunk and commit clean raw KV separately.

    ``forward`` returns the native data-ward velocity. ``commit`` runs the same
    student at clean time one, staging this chunk's KV transaction. Supplying
    ``noise_fn`` lets sequence-parallel callers broadcast fresh randomness from
    their authoritative rank. It is invoked exactly three times per chunk.
    """
    if current_noise.ndim != 5 or current_noise.shape[2] != CHUNK_LATENTS:
        raise ValueError("SolarWM-H3 SGF requires [B,C,5,H,W] noise chunks")
    if current_noise.dtype != torch.float32:
        raise ValueError("SolarWM-H3 samples its flow trajectory in float32")
    times = stage2_timesteps(current_noise.device)
    current = current_noise
    for step, time in enumerate(times):
        velocity = forward(current, time)
        if velocity.shape != current.shape:
            raise ValueError("H3 student velocity shape differs from its noisy input")
        clean = current + (1.0 - time) * velocity.float()
        if not bool(torch.isfinite(clean).all()):
            raise FloatingPointError(f"non-finite SolarWM-H3 chunk={chunk_index}, denoise={step}")
        if step < DENOISING_STEPS - 1:
            noise = (
                noise_fn(clean)
                if noise_fn is not None
                else torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
            )
            current = scale_noise(clean, noise, times[step + 1])
    commit(clean, current_noise.new_tensor(1.0))
    return clean
