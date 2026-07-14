# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared latent layout and normalization primitives for LTX pipelines."""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor


def pack_latents(
    latents: torch.Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size, _, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size,
        num_frames,
        height,
        width,
        -1,
        patch_size_t,
        patch_size,
        patch_size,
    )
    return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


def normalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) * scaling_factor / latents_std


def normalize_audio_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def denormalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * latents_std / scaling_factor + latents_mean


def denormalize_audio_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return latents * latents_std + latents_mean


def create_noised_state(
    latents: torch.Tensor,
    noise_scale: float | torch.Tensor,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> torch.Tensor:
    noise = randn_tensor(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    return noise_scale * noise + (1 - noise_scale) * latents


def pack_audio_latents(
    latents: torch.Tensor,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size, _, latent_length, latent_mel_bins = latents.shape
        post_patch_latent_length = latent_length / patch_size_t
        post_patch_mel_bins = latent_mel_bins / patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_latent_length,
            patch_size_t,
            post_patch_mel_bins,
            patch_size,
        )
        return latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_audio_latents(
    latents: torch.Tensor,
    latent_length: int,
    num_mel_bins: int,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size,
            latent_length,
            num_mel_bins,
            -1,
            patch_size_t,
            patch_size,
        )
        return latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


def unpad_audio_latents(latents: torch.Tensor, num_frames: int) -> torch.Tensor:
    return latents[:, :num_frames]


def get_sp_padded_audio_latent_length(audio_latent_length: int, sp_size: int) -> int:
    if sp_size > 1:
        audio_latent_length += (sp_size - (audio_latent_length % sp_size)) % sp_size
    return audio_latent_length


def resolve_video_latent_shape(
    height: int,
    width: int,
    num_frames: int,
    *,
    vae_spatial_compression_ratio: int,
    vae_temporal_compression_ratio: int,
) -> tuple[int, int, int]:
    return (
        (num_frames - 1) // vae_temporal_compression_ratio + 1,
        height // vae_spatial_compression_ratio,
        width // vae_spatial_compression_ratio,
    )
