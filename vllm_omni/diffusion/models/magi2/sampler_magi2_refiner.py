# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""MAGI-2 refiner hand-off, scheduler and guided sampling.

The ZeroSNR table, the upsample-and-renoise hand-off from the Preview latent,
the refiner scheduler settings and the guided refiner loop are adapted from
SandAI's Apache-2.0 MAGI-2 Preview inference implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler

from .configuration_magi2 import MAGI2_REFINER_GENERATION_CONFIG
from .parallel import gather_sequence, get_magi2_ulysses_group
from .refiner_data_proxy import Magi2RefinerDataProxy
from .sampler_magi2 import build_flow_unipc_scheduler


@cache
def _zero_snr_sigmas(linear_start: float, linear_end: float, num_timesteps: int) -> torch.Tensor:
    # Built on the CPU regardless of the default device: the cumulative product
    # runs in float64 on the host and only the fp32 table leaves this function.
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64, device="cpu") ** 2
    alphas_cumprod = np.cumprod(1.0 - betas.numpy(), axis=0)
    # numpy's fp32 sqrt is correctly rounded; some torch CPU builds are off by one ULP.
    sigmas = torch.from_numpy(np.sqrt(alphas_cumprod.astype(np.float32)))
    first, last = sigmas[0].clone(), sigmas[-1].clone()
    # Rescale to zero terminal SNR; the factor is evaluated before the product.
    return (sigmas - last) * (first / (first - last))


def zero_snr_sigmas() -> torch.Tensor:
    """Return the fp32 ZeroSNR signal-coefficient table on the CPU.

    Index 0 is the cleanest level and the last index is exactly zero.
    """

    gen = MAGI2_REFINER_GENERATION_CONFIG
    return _zero_snr_sigmas(gen.zero_snr_linear_start, gen.zero_snr_linear_end, gen.zero_snr_num_timesteps).clone()


def zero_snr_signal_coefficients(index: int = MAGI2_REFINER_GENERATION_CONFIG.noise_index) -> tuple[float, float]:
    """Return the fp32 ``(signal, noise)`` renoise coefficients at ``index``.

    ``noise = (1 - signal**2) ** 0.5`` is evaluated in fp32.  Both values are
    returned as Python floats that hold the fp32 results exactly.
    """

    gen = MAGI2_REFINER_GENERATION_CONFIG
    sigmas = _zero_snr_sigmas(gen.zero_snr_linear_start, gen.zero_snr_linear_end, gen.zero_snr_num_timesteps)
    if not 0 <= index < sigmas.numel():
        raise ValueError(f"ZeroSNR index must be in [0, {sigmas.numel()}), got {index}")
    signal = sigmas[index]
    noise = (1 - signal**2) ** 0.5
    return signal.item(), noise.item()


def refiner_latent_size(preview_latent_frames: int, height: int, width: int) -> tuple[int, int, int]:
    """Return the refiner latent ``(T, H, W)`` for a Preview latent and output size.

    The refiner doubles the Preview's temporal grid (``2T - 1`` frames) and
    uses a 16x spatial VAE stride.
    """

    _, stride_h, stride_w = MAGI2_REFINER_GENERATION_CONFIG.video_vae_stride
    if preview_latent_frames <= 0:
        raise ValueError(f"preview_latent_frames must be positive, got {preview_latent_frames}")
    if height <= 0 or width <= 0 or height % stride_h or width % stride_w:
        raise ValueError(
            f"MAGI-2 refiner output size must be a positive multiple of ({stride_h}, {stride_w}), "
            f"got height={height}, width={width}"
        )
    return 2 * preview_latent_frames - 1, height // stride_h, width // stride_w


def upsample_and_renoise(
    latent: torch.Tensor,
    size: Sequence[int],
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Upsample a Preview latent to the refiner grid and renoise it.

    ``latent`` is ``[B, C, T, H, W]``; ``size`` is the refiner latent
    ``(T, H, W)``.  The latent is interpolated trilinearly with aligned
    corners in fp32 and mixed with Gaussian noise at the configured ZeroSNR
    level.  Without ``noise``, it is drawn after the interpolation from
    ``generator`` (or the device's default generator) at the upsampled shape.
    """

    if latent.ndim != 5:
        raise ValueError(f"MAGI-2 refiner hand-off expects a [B, C, T, H, W] latent, got {tuple(latent.shape)}")
    size = tuple(int(s) for s in size)
    if len(size) != 3 or any(s <= 0 for s in size):
        raise ValueError(f"MAGI-2 refiner latent size must be three positive (T, H, W) values, got {size}")
    upsampled = F.interpolate(latent.to(torch.float32), size=size, mode="trilinear", align_corners=True)
    if noise is None:
        noise = torch.randn(upsampled.shape, generator=generator, device=upsampled.device, dtype=upsampled.dtype)
    elif noise.shape != upsampled.shape:
        raise ValueError(
            f"MAGI-2 refiner noise shape {tuple(noise.shape)} does not match the upsampled latent "
            f"{tuple(upsampled.shape)}"
        )
    else:
        noise = noise.to(device=upsampled.device, dtype=torch.float32)
    signal, noise_scale = zero_snr_signal_coefficients(MAGI2_REFINER_GENERATION_CONFIG.noise_index)
    return upsampled * signal + noise * noise_scale


def build_magi2_refiner_scheduler(
    num_inference_steps: int = MAGI2_REFINER_GENERATION_CONFIG.num_inference_steps,
    shift: float = MAGI2_REFINER_GENERATION_CONFIG.shift,
    *,
    device: torch.device | str | None = None,
) -> FlowUniPCMultistepScheduler:
    """Create the refiner's video scheduler with the refiner's step count and shift."""

    return build_flow_unipc_scheduler(num_inference_steps, shift=shift, device=device)


class Magi2RefinerSampler(CFGParallelMixin):
    """Run the refiner's guided FlowUniPC loop over a renoised latent.

    Every step evaluates the conditional and then the unconditional text
    branch (or one branch per rank under CFG parallelism), combines them as
    ``u + scale * (c - u)`` with no rescaling, and steps the scheduler in
    fp32.  The model receives no timestep.
    """

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        data_proxy: Magi2RefinerDataProxy | None = None,
        *,
        guidance_scale: float = MAGI2_REFINER_GENERATION_CONFIG.video_guidance_scale,
    ) -> None:
        if not math.isfinite(guidance_scale):
            raise ValueError(f"guidance_scale must be finite, got {guidance_scale}")
        self.model = model
        self.data_proxy = data_proxy or Magi2RefinerDataProxy()
        self.guidance_scale = float(guidance_scale)

    @torch.inference_mode()
    def predict_noise(self, latent: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Return one branch's fp32 ``[1, C, T, H, W]`` velocity for ``latent`` and text features."""

        packed = self.data_proxy.pack(latent, text)
        group = get_magi2_ulysses_group()
        local = packed.shard(group.rank, group.world_size)
        velocity = self.model(**local.model_kwargs())
        velocity = gather_sequence(velocity, local.video_split_sizes, group)
        return self.data_proxy.unpack(velocity, packed.grid)

    def denoise_step(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_null: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
    ) -> torch.Tensor:
        """Run one guided step on the fp32 ``latent`` and return the next latent."""

        velocity = self.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=self.guidance_scale,
            positive_kwargs={"latent": latent, "text": context},
            negative_kwargs={"latent": latent, "text": context_null},
            cfg_normalize=False,
        )
        return scheduler.step(velocity.to(torch.float32), timestep, latent, return_dict=False)[0]

    @torch.inference_mode()
    def sample(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        context_null: torch.Tensor,
        *,
        num_inference_steps: int = MAGI2_REFINER_GENERATION_CONFIG.num_inference_steps,
        shift: float = MAGI2_REFINER_GENERATION_CONFIG.shift,
    ) -> torch.Tensor:
        """Refine a renoised ``[1, C, T, H, W]`` latent and return it in fp32.

        ``context`` and ``context_null`` are the unpadded prompt and negative
        prompt features, ``[L, D]`` or ``[1, L, D]``; their lengths may differ.
        A fresh scheduler is built for every call.
        """

        scheduler = build_magi2_refiner_scheduler(num_inference_steps, shift, device=latent.device)
        latent = latent.to(torch.float32)
        for timestep in scheduler.timesteps:
            latent = self.denoise_step(latent, timestep, context, context_null, scheduler)
        return latent
