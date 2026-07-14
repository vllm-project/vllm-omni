# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared latent layout and normalization primitives for LTX pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class LTXAVState:
    """Packed video and audio latents carried between denoise steps."""

    video: torch.Tensor
    audio: torch.Tensor


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


def prepare_video_latents(
    pipeline: Any,
    batch_size: int = 1,
    num_channels_latents: int = 128,
    height: int = 512,
    width: int = 768,
    num_frames: int = 121,
    noise_scale: float = 0.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    if latents is not None:
        if latents.ndim == 5:
            latents = normalize_latents(
                latents,
                pipeline.vae.latents_mean,
                pipeline.vae.latents_std,
                pipeline.vae.config.scaling_factor,
            )
            latents = pack_latents(
                latents,
                pipeline.transformer_spatial_patch_size,
                pipeline.transformer_temporal_patch_size,
            )
        if latents.ndim != 3:
            raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")
        return create_noised_state(latents, noise_scale, generator).to(device=device, dtype=dtype)

    num_frames, height, width = resolve_video_latent_shape(
        height,
        width,
        num_frames,
        vae_spatial_compression_ratio=pipeline.vae_spatial_compression_ratio,
        vae_temporal_compression_ratio=pipeline.vae_temporal_compression_ratio,
    )
    shape = (batch_size, num_channels_latents, num_frames, height, width)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return pack_latents(
        latents,
        pipeline.transformer_spatial_patch_size,
        pipeline.transformer_temporal_patch_size,
    )


def prepare_audio_latents(
    pipeline: Any,
    batch_size: int = 1,
    num_channels_latents: int = 8,
    audio_latent_length: int = 1,
    num_mel_bins: int = 64,
    noise_scale: float = 0.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    original_latent_length = audio_latent_length
    latent_mel_bins = num_mel_bins // pipeline.audio_vae_mel_compression_ratio
    sp_size = getattr(pipeline.od_config.parallel_config, "sequence_parallel_size", 1) or 1
    padded_latent_length = get_sp_padded_audio_latent_length(original_latent_length, int(sp_size))

    if latents is not None:
        if latents.ndim == 4:
            latents = pack_audio_latents(latents)
        if latents.ndim != 3:
            raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")
        latents = normalize_audio_latents(
            latents,
            pipeline.audio_vae.latents_mean,
            pipeline.audio_vae.latents_std,
        )
        latents = create_noised_state(latents, noise_scale, generator)

        if latents.shape[1] not in {original_latent_length, padded_latent_length}:
            raise ValueError(
                "Provided `audio_latents` has incompatible audio frame count "
                f"{latents.shape[1]}; expected {original_latent_length} or {padded_latent_length}."
            )
        if latents.shape[1] == original_latent_length and padded_latent_length > original_latent_length:
            padding = torch.zeros(
                latents.shape[0],
                padded_latent_length - original_latent_length,
                latents.shape[2],
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, padding], dim=1)
        return latents.to(device=device, dtype=dtype), original_latent_length, padded_latent_length

    shape = (batch_size, num_channels_latents, padded_latent_length, latent_mel_bins)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return pack_audio_latents(latents), original_latent_length, padded_latent_length
