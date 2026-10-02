# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Release sampling paths from Sana e93c883e10730ee5a4a6edf1cbcf501dc4ef753b.

T2V uses order-two multistep DPM-Solver++ on the flow noise schedule.
TI2V uses FlowMatch Euler with per-token timesteps and a fixed first frame.
All solver state is request-local. No upstream runner is required at runtime.
"""

from collections.abc import Callable

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def sample_flow_dpm(
    predict_noise: Callable,
    latents: torch.Tensor,
    steps: int,
    shift: float,
    callback: Callable | None = None,
) -> torch.Tensor:
    """The callback predicts noise, including CFG in noise space, at time t∈[0,1].

    Keep the upstream flow-to-noise-to-data conversion and FP32 coefficients;
    simplifying it algebraically changes rounding near the first timestep.
    """
    if steps < 2:
        raise ValueError("Flow DPM-Solver requires at least two inference steps")
    betas = torch.linspace(1.0, 0.001, steps + 1, device="cpu").to(latents.device)
    sigmas = 1.0 - betas
    times = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(0)
    previous_time = None
    previous_prediction = None
    x = latents
    for index in range(steps):
        s = times[index].reshape((1,) * x.ndim)
        t = times[index + 1].reshape((1,) * x.ndim)
        noise = predict_noise(x, times[index])
        prediction = (x - s * noise) / (1 - s)
        lambda_s = torch.log(1 - s) - torch.log(s)
        lambda_t = torch.log(1 - t) - torch.log(t)
        h = lambda_t - lambda_s
        alpha_t = torch.exp(torch.log(1 - t))
        phi = torch.expm1(-h)
        # Bootstrap and the final step use first order, including for 50 steps.
        if index == 0 or index == steps - 1:
            x = t / s * x - alpha_t * phi * prediction
        else:
            lambda_previous = torch.log(1 - previous_time) - torch.log(previous_time)
            ratio = (lambda_s - lambda_previous) / h
            derivative = (1.0 / ratio) * (prediction - previous_prediction)
            x = (t / s) * x - (alpha_t * phi) * prediction - 0.5 * (alpha_t * phi) * derivative
        previous_time, previous_prediction = s, prediction
        if callback is not None:
            callback(index, times[index + 1], x)
    return x


def sample_ltx_euler(
    predict_flow: Callable,
    latents: torch.Tensor,
    steps: int,
    shift: float,
    callback: Callable | None = None,
) -> torch.Tensor:
    """TI2V release path: frame zero is clean conditioning (noise multiplier 0)."""
    # The engine may establish a CUDA default-device context; Diffusers
    # constructs its schedule through NumPy and must initialize on CPU.
    with torch.device("cpu"):
        scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        scheduler.set_timesteps(steps, device=latents.device)
    condition_mask = torch.zeros_like(latents, dtype=torch.float32)
    condition_mask[:, :, 0] = 1
    for index, time in enumerate(scheduler.timesteps):
        timestep = torch.minimum(time.expand(latents.shape).float(), (1 - condition_mask) * 1000.0)
        prediction = predict_flow(latents, timestep[:, :1, :, :1, :1])
        batch, channels = latents.shape[:2]
        updated = (
            scheduler.step(
                -prediction.reshape(batch, channels, -1).transpose(1, 2),
                time,
                latents.reshape(batch, channels, -1).transpose(1, 2),
                per_token_timesteps=timestep.reshape(batch, channels, -1)[:, 0],
                return_dict=False,
            )[0]
            .transpose(1, 2)
            .reshape(latents.shape)
        )
        denoise_mask = time / 1000 - 1e-6 < (1.0 - condition_mask)
        latents = torch.where(denoise_mask, updated, latents).to(latents.dtype)
        if callback is not None:
            callback(index, time, latents)
    return latents
