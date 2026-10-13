# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SolarWM, Copyright (c) SolarWM contributors.
# Source: Junchao-cs/SolarWM@ce1da4e7705391eda8eeda6016c0fd3f614b975e
"""Native H3 geometry and the packed document used by the SolarWM student."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FullLengthGeometry:
    source_frames: int
    decoded_frames: int
    decode_latents: int
    rollout_latents: int
    camera_indices: tuple[int, ...]
    tail_camera_rows: int


def full_length_geometry(source_frames: int) -> FullLengthGeometry:
    """Align a physical video to H3's 17-pixel / five-latent cadence.

    The image anchor is encoded separately. It does not consume target latent
    zero or either of the two extra latents required by the temporal decoder.
    """
    if isinstance(source_frames, bool) or not isinstance(source_frames, int) or source_frames < 22:
        raise ValueError("SolarWM-H3 requires at least 22 consecutive frames")
    periods = (source_frames - 5 + 16) // 17
    decode_latents = 5 * periods + 2
    rollout_latents = (decode_latents + 4) // 5 * 5
    indices, source_index, tail = [0], 0, 0
    for index in range(rollout_latents - 1):
        source_index += (1, 4, 4, 4, 4)[index % 5]
        if source_index < source_frames:
            indices.append(source_index)
        else:
            indices.append(indices[-1])
            tail += 1
    return FullLengthGeometry(source_frames, 17 * periods + 5, decode_latents, rollout_latents, tuple(indices), tail)


def native_video_position_grid(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    origin: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return native, nonuniform H3 ``(t,h,w)`` positions in frame-major order."""
    pt, ph, pw = patch_size
    if pt != 1 or ph <= 0 or pw <= 0 or latent_height <= 0 or latent_width <= 0:
        raise ValueError("H3 requires temporal patch size one and positive spatial dimensions")
    if latent_height % ph or latent_width % pw or num_latent_frames < 0:
        raise ValueError("H3 latent grid must be divisible by its patch size")
    area = math.sqrt(latent_height * latent_width)

    def axis(size: int, patch: int) -> torch.Tensor:
        ratio = size / area
        return (
            torch.arange(size // patch, device=device, dtype=torch.float64) / (size // patch) * ratio
            + (1.0 - ratio) / 2.0
        ) * 32.0

    h, w = torch.meshgrid(axis(latent_height, ph), axis(latent_width, pw), indexing="ij")
    spatial = torch.stack((h.flatten(), w.flatten()), -1)
    times = torch.empty(num_latent_frames, dtype=torch.float64, device=device)
    if num_latent_frames:
        times[0] = origin
        cadence = torch.tensor((1, 4, 4, 4, 4), dtype=torch.float64, device=device)
        offsets = cadence[torch.arange(num_latent_frames - 1, device=device) % 5] * (5.0 / 3.0)
        times[1:] = origin + offsets.cumsum(0)
    return torch.cat(
        (times[:, None, None].expand(-1, spatial.shape[0], 1), spatial[None].expand(num_latent_frames, -1, -1)), -1
    ).reshape(-1, 3)


@dataclass(frozen=True)
class H3PackedLayout:
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    condition_indices: torch.Tensor
    condition_video_indices: torch.Tensor
    noisy_video_indices: torch.Tensor
    rows_per_video_frame: int
    latent_height: int
    latent_width: int
    patch_size: tuple[int, int, int] = (1, 2, 2)

    @property
    def sequence_length(self) -> int:
        return self.position_ids.shape[0]

    @property
    def num_condition_video_rows(self) -> int:
        return self.condition_video_indices.numel()

    @property
    def noisy_video_output_slice(self) -> slice:
        return slice(self.num_condition_video_rows, self.video_indices.numel())

    @property
    def target_video_output_slice(self) -> slice:
        return self.noisy_video_output_slice

    def transformer_kwargs(self) -> dict[str, torch.Tensor]:
        """The original Diffusers structural names, also useful to native callers."""
        return {
            name: getattr(self, name)
            for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
        }


def build_layout(
    text_token_tags: torch.Tensor,
    num_video_latents: int = 5,
    latent_height: int = 48,
    latent_width: int = 84,
    num_audio_latents: int = 263,
    *,
    device: torch.device | str | None = None,
) -> H3PackedLayout:
    """Pack joint Qwen rows, separate visual anchor, stereo audio and video.

    Qwen image rows retain modality tag zero while remaining in the text input
    segment. Do not infer segment boundaries or camera rows from modality tags.
    """
    tags = torch.as_tensor(text_token_tags, device=device)
    if tags.ndim != 1 or tags.numel() == 0 or not bool(((tags == 0) | (tags == 1)).all()):
        raise ValueError("Qwen text_token_tags must be a nonempty vector containing zero or one")
    if tags.dtype.is_floating_point or tags.dtype == torch.bool:
        raise ValueError("Qwen text_token_tags must have integer dtype")
    tags = tags.long()
    if num_video_latents <= 0 or num_audio_latents < 0:
        raise ValueError("H3 video length must be positive and audio length nonnegative")
    video_grid = native_video_position_grid(
        num_video_latents, latent_height, latent_width, origin=float(tags.numel()), device=tags.device
    )
    rows = latent_height // 2 * (latent_width // 2)
    num_text, num_audio = tags.numel(), num_audio_latents * 2
    audio_start, video_start = num_text + rows, num_text + rows + num_audio
    total = video_start + num_video_latents * rows

    def indices(start: int, stop: int) -> torch.Tensor:
        return torch.arange(start, stop, dtype=torch.long, device=tags.device)

    positions = torch.zeros((total, 3), dtype=torch.float64, device=tags.device)
    positions[:num_text, 0] = indices(0, num_text)
    positions[num_text:audio_start] = video_grid[:rows]
    if num_audio_latents:
        positions[audio_start:video_start, 0] = (
            torch.arange(num_audio_latents, dtype=torch.float64, device=tags.device) + num_text
        ).repeat(2)
        width_grid = video_grid[:rows, 2]
        positions[audio_start : audio_start + num_audio_latents, 2] = width_grid[0]
        positions[audio_start + num_audio_latents : video_start, 2] = width_grid[-1]
    positions[video_start:] = video_grid
    anchor = indices(num_text, audio_start)
    current = indices(video_start, total)
    return H3PackedLayout(
        position_ids=positions,
        token_tags=torch.cat(
            (tags, tags.new_zeros(rows), tags.new_full((num_audio,), 2), tags.new_zeros(num_video_latents * rows))
        ),
        video_indices=torch.cat((anchor, current)),
        audio_indices=indices(audio_start, video_start),
        text_indices=indices(0, num_text),
        condition_indices=indices(0, audio_start),
        condition_video_indices=anchor,
        noisy_video_indices=current,
        rows_per_video_frame=rows,
        latent_height=latent_height,
        latent_width=latent_width,
    )


def build_row_timesteps(
    layout: H3PackedLayout,
    video_timestep: float | torch.Tensor,
    audio_timestep: float | torch.Tensor,
    *,
    text_timestep: float = 1.0,
    condition_video_timestep: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group per-row native data-ward times without changing modality tags."""
    device = layout.position_ids.device

    def scalar(value: float | torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, device=device, dtype=torch.float32)
        if (
            result.numel() != 1
            or not bool(torch.isfinite(result).all())
            or not bool(((result >= 0) & (result <= 1)).all())
        ):
            raise ValueError("H3 timesteps must be finite scalars in [0,1]")
        return result.reshape(())

    times = torch.empty(layout.sequence_length, dtype=torch.float32, device=device)
    times[layout.text_indices] = scalar(text_timestep)
    times[layout.condition_video_indices] = scalar(condition_video_timestep)
    times[layout.audio_indices] = scalar(audio_timestep)
    times[layout.noisy_video_indices] = scalar(video_timestep)
    return torch.unique(times, sorted=True, return_inverse=True)


def patchify_video(latents: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """Pack BCTHW latents into batch-major video patch rows."""
    if latents.ndim != 5:
        raise ValueError("video latents must have shape [B,C,T,H,W]")
    batch, channels, frames, height, width = latents.shape
    pt, ph, pw = patch_size
    if min(pt, ph, pw) <= 0 or frames % pt or height % ph or width % pw:
        raise ValueError("video latent shape must be divisible by patch size")
    return (
        latents.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(batch, -1, channels * pt * ph * pw)
    )


def unpatchify_video(
    rows: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    pt, ph, pw = patch_size
    if rows.ndim != 3 or min(pt, ph, pw) <= 0 or frames % pt or height % ph or width % pw:
        raise ValueError("invalid H3 patch rows or latent dimensions")
    if rows.shape[1] != frames // pt * (height // ph) * (width // pw) or rows.shape[2] % (pt * ph * pw):
        raise ValueError("H3 patch rows do not match the requested latent grid")
    channels = rows.shape[2] // (pt * ph * pw)
    return (
        rows.reshape(rows.shape[0], frames // pt, height // ph, width // pw, channels, pt, ph, pw)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(rows.shape[0], channels, frames, height, width)
    )
