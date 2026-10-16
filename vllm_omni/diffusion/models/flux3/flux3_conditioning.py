# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent sampling and conditioning helpers for Flux3.

Provides image noise-latent samplers and conditioning packers that build the
``(tokens, ids, timesteps)`` triples the image DiT consumes.
"""

import torch
from torch import Tensor

from vllm_omni.diffusion.models.flux3.flux3_positional import (
    batched_prc_img,
    prc_img,
)

CONDITIONING_NOISE_SEED_OFFSET = 10_000_003
I2I_COND_T = 0
I2I_COND_L = -1


def sample_image_latents(
    *,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    image_rope_scale: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample image noise latents + 4-axis ids.

    Returns ``(x, x_ids, orig_ids)`` where ``x`` is ``(1, h*w, 128)``, ``x_ids`` is
    ``(1, h*w, 4)`` (h/w axes scaled by ``image_rope_scale``), and ``orig_ids`` is the
    ``(1, h*w, 4)`` integer ids used to scatter the decoded latent. Noise is sampled
    directly in token order.
    """
    h, w = max(1, height // 16), max(1, width // 16)
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn((h * w, 128), generator=rng).to(device=device, dtype=dtype).unsqueeze(0)
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    orig_ids = torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(1, h * w, 4)
    x_ids = orig_ids.to(dtype=torch.float32)
    x_ids[..., 1:3] = x_ids[..., 1:3] * image_rope_scale
    return x, x_ids, orig_ids


def _conditioning_noise_like(x: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed + CONDITIONING_NOISE_SEED_OFFSET)
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def pack_image_condition(
    x_cond: Tensor,
    *,
    device: torch.device,
    conditioning_noise: float = 0.0,
    seed: int | None = None,
    image_rope_scale: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack an image-conditioning latent ``(1, 128, h, w)`` into
    ``(x_cond, x_cond_ids, x_cond_timesteps)`` for the DiT ``x_image_cond`` stream."""
    x_cond, x_cond_ids = batched_prc_img(x_cond)
    x_cond_ids = x_cond_ids.to(dtype=torch.float32)
    x_cond_ids[..., 1:3] = x_cond_ids[..., 1:3] * image_rope_scale
    x_cond_timesteps = conditioning_noise * torch.ones(x_cond.shape[:2], device=x_cond.device, dtype=x_cond.dtype)
    if conditioning_noise > 0:
        if seed is None:
            raise ValueError("seed is required when conditioning_noise > 0")
        x_cond = conditioning_noise * _conditioning_noise_like(x_cond, seed) + (1 - conditioning_noise) * x_cond
    x_cond = x_cond.to(device=device, dtype=torch.bfloat16)
    x_cond_ids = x_cond_ids.to(device=device)
    x_cond_timesteps = x_cond_timesteps.to(device=device, dtype=torch.bfloat16)
    return x_cond, x_cond_ids, x_cond_timesteps


def pack_i2i_image_condition(
    latents: list[Tensor],
    *,
    device: torch.device,
    conditioning_noise: float = 0.0,
    seed: int | None = None,
    l_start: int = I2I_COND_L,
    image_rope_scale: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack one or more reference-image latents ``(128, h, w)`` for the DiT
    ``x_image_cond`` stream, each placed at a distinct ``l`` coordinate."""
    x_cond_parts = []
    x_cond_id_parts = []
    for i_ref, latent in enumerate(latents):
        x_ref, x_ref_ids = prc_img(
            latent,
            t_coord=torch.tensor([I2I_COND_T], device=latent.device, dtype=torch.long),
            l_coord=torch.tensor([l_start - i_ref], device=latent.device, dtype=torch.long),
        )
        x_cond_parts.append(x_ref)
        x_cond_id_parts.append(x_ref_ids)

    x_cond = torch.cat(x_cond_parts, dim=0).unsqueeze(0)
    x_cond_ids = torch.cat(x_cond_id_parts, dim=0).unsqueeze(0)
    x_cond_ids = x_cond_ids.to(dtype=torch.float32)
    x_cond_ids[..., 1:3] = x_cond_ids[..., 1:3] * image_rope_scale
    x_cond_timesteps = conditioning_noise * torch.ones(x_cond.shape[:2], device=x_cond.device, dtype=x_cond.dtype)
    if conditioning_noise > 0:
        if seed is None:
            raise ValueError("seed is required when conditioning_noise > 0")
        x_cond = conditioning_noise * _conditioning_noise_like(x_cond, seed) + (1 - conditioning_noise) * x_cond
    x_cond = x_cond.to(device=device, dtype=torch.bfloat16)
    x_cond_ids = x_cond_ids.to(device=device)
    x_cond_timesteps = x_cond_timesteps.to(device=device, dtype=torch.bfloat16)
    return x_cond, x_cond_ids, x_cond_timesteps
