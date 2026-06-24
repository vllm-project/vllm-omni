# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Krea 2 pre/post-processing utilities for vLLM-Omni."""

from __future__ import annotations

import torch


def pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """Pack latents into patch tokens: (B,C,H,W) -> (B, H/p*W/p, C*p*p)."""
    p = patch_size
    latents = latents.view(batch_size, num_channels, height // p, p, width // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // p) * (width // p), num_channels * p * p)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """Unpack patch tokens back to spatial: (B, S, C*p*p) -> (B, C, 1, H, W)."""
    if latents.dim() == 4:
        return latents
    batch_size, _, channels = latents.shape
    p = patch_size

    lat_h = p * (int(height) // (vae_scale_factor * p))
    lat_w = p * (int(width) // (vae_scale_factor * p))

    latents = latents.view(batch_size, lat_h // p, lat_w // p, channels // (p * p), p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (p * p), 1, lat_h, lat_w)
    return latents


def prepare_position_ids(
    text_seq_len: int,
    grid_height: int,
    grid_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Build (text_seq + image_seq, 3) rotary coordinates.

    Text tokens sit at the origin; image tokens carry (0, h, w) grid coords.
    """
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)


def denormalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    """Denormalize VAE latents using per-channel mean/std from VAE config."""
    z_dim = latents.shape[1]
    mean = latents_mean.view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    std = (1.0 / latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * std + mean
