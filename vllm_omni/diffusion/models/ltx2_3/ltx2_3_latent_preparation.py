# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 latent preparation and decode conditioning helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.utils.torch_utils import randn_tensor


@dataclass(frozen=True)
class VideoLatentShape:
    """Unpacked and packed layout for one LTX-2.3 video latent state."""

    batch_size: int
    num_channels: int
    num_frames: int
    height: int
    width: int
    patch_size: int
    patch_size_t: int

    @property
    def conditioning_mask_shape(self) -> tuple[int, int, int, int, int]:
        return (self.batch_size, 1, self.num_frames, self.height, self.width)

    @property
    def packed_shape(self) -> tuple[int, int, int]:
        return (
            self.batch_size,
            (self.num_frames // self.patch_size_t) * (self.height // self.patch_size) * (self.width // self.patch_size),
            self.num_channels * self.patch_size_t * self.patch_size * self.patch_size,
        )


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


def create_noised_state(
    latents: torch.Tensor,
    noise_scale: float | torch.Tensor,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> torch.Tensor:
    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    return noise_scale * noise + (1 - noise_scale) * latents


def prepare_video_latent_state(
    latents: torch.Tensor | None,
    *,
    shape: VideoLatentShape,
    generator: torch.Generator | list[torch.Generator] | None,
    dtype: torch.dtype | None,
    device: torch.device | None,
    latents_mean: torch.Tensor | None,
    latents_std: torch.Tensor | None,
    scaling_factor: float,
    noise_scale: float | torch.Tensor = 1.0,
    conditioning_mask: torch.Tensor | None = None,
    noise_packed_latents: bool = True,
    latents_are_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize, noise, and pack a video state for T2V or I2V.

    T2V passes no conditioning mask and uses ordinary packed-latent noising.
    I2V supplies an unpacked first-frame mask and disables noising for caller-
    provided packed latents, which already encode the full conditioned state.
    Image-encoded I2V latents can be normalized before temporal repetition and
    marked as such to avoid repeating the normalization work for every frame.
    """
    packed_conditioning_mask = None
    if conditioning_mask is not None:
        packed_conditioning_mask = pack_latents(
            conditioning_mask,
            patch_size=shape.patch_size,
            patch_size_t=shape.patch_size_t,
        ).squeeze(-1)

    if latents is None:
        if conditioning_mask is not None:
            raise ValueError("Conditioning mask requires an initial video latent state.")
        return randn_tensor(shape.packed_shape, generator=generator, device=device, dtype=dtype), None

    if latents.ndim == 5:
        if not latents_are_normalized:
            if latents_mean is None or latents_std is None:
                raise ValueError("5D video latents require VAE normalization statistics.")
            latents = normalize_latents(latents, latents_mean, latents_std, scaling_factor)
        latents = create_noised_state(latents, noise_scale, generator)
        latents = pack_latents(latents, patch_size=shape.patch_size, patch_size_t=shape.patch_size_t)
    elif latents.ndim == 3:
        if tuple(latents.shape) != shape.packed_shape:
            raise ValueError(
                f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is {shape.packed_shape}."
            )
        if noise_packed_latents:
            latents = create_noised_state(latents, noise_scale, generator)
    else:
        raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")

    return latents.to(device=device, dtype=dtype), packed_conditioning_mask


def repeat_prompt_tensor_for_outputs(tensor: torch.Tensor, num_videos_per_prompt: int) -> torch.Tensor:
    if num_videos_per_prompt == 1:
        return tensor
    return tensor.repeat_interleave(num_videos_per_prompt, dim=0)


def pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
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
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def normalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


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
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def denormalize_audio_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


def pack_audio_latents(
    latents: torch.Tensor,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
        post_patch_latent_length = latent_length // patch_size_t
        post_patch_mel_bins = latent_mel_bins // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_latent_length,
            patch_size_t,
            post_patch_mel_bins,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
    else:
        latents = latents.transpose(1, 2).flatten(2, 3)
    return latents


def unpack_audio_latents(
    latents: torch.Tensor,
    latent_length: int,
    num_mel_bins: int,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size = latents.size(0)
        post_patch_latent_length = latent_length // patch_size_t
        post_patch_mel_bins = num_mel_bins // patch_size
        latents = latents.reshape(
            batch_size,
            post_patch_latent_length,
            post_patch_mel_bins,
            -1,
            patch_size_t,
            patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
    else:
        latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
    return latents


def unpad_audio_latents(latents: torch.Tensor, num_frames: int) -> torch.Tensor:
    return latents[:, :num_frames]


def get_sp_padded_audio_latent_length(audio_latent_length: int, sp_size: int) -> int:
    if sp_size > 1:
        audio_latent_length += (sp_size - (audio_latent_length % sp_size)) % sp_size
    return audio_latent_length


def expand_per_prompt_decode_value(
    value: float | list[float],
    *,
    prompt_batch_size: int,
    effective_batch_size: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list):
        return [value] * effective_batch_size
    if len(value) == 1:
        return value * effective_batch_size
    if len(value) == effective_batch_size:
        return value
    if prompt_batch_size > 0 and len(value) == prompt_batch_size and effective_batch_size % prompt_batch_size == 0:
        repeats = effective_batch_size // prompt_batch_size
        return [item for item in value for _ in range(repeats)]
    raise ValueError(
        f"`{field_name}` must have length 1, prompt batch size ({prompt_batch_size}), or effective batch size"
        f" ({effective_batch_size}); got {len(value)}."
    )


def prepare_decode_timestep_conditioning(
    *,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    prompt_batch_size: int,
    effective_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_timestep_values = expand_per_prompt_decode_value(
        decode_timestep,
        prompt_batch_size=prompt_batch_size,
        effective_batch_size=effective_batch_size,
        field_name="decode_timestep",
    )
    if decode_noise_scale is None:
        decode_noise_scale_values = decode_timestep_values
    else:
        decode_noise_scale_values = expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )
