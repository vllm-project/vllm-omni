# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared sensor clock, chunk schedule and token geometry for Cosmos3-Nano-Sim-Auto.

The causal multiview student generates one *chunk* at a time. A chunk is a
fixed wall-clock span (``frames_per_chunk`` RGB latents, 0.4 s at 30 fps with
the 4x Wan temporal compression) that also owns the LiDAR sweeps falling into
the same span. Chunk 0 is a singleton (latent 0 and sweep 0 only) and the last
chunk may be partial. Every helper here is pure and host-side; the pipeline
derives the paged-KV frame unit and the per-chunk packed layout from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Paged attention kernels in ``experimental.ar_diffusion`` page in multiples of 16 tokens.
PAGED_TOKEN_ALIGNMENT = 16


def latent_frames_from_pixel_frames(num_frames: int, temporal_compression: int = 4) -> int:
    """Wan ``4k+1`` pixel frames -> ``k+1`` latent frames."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if temporal_compression < 1:
        raise ValueError(f"temporal_compression must be positive, got {temporal_compression}.")
    if (num_frames - 1) % temporal_compression:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto requires num_frames = {temporal_compression}k + 1 pixel frames, got {num_frames}."
        )
    return 1 + (num_frames - 1) // temporal_compression


def lidar_num_sweeps(num_frames: int, *, camera_fps: float, lidar_fps: float) -> int:
    """Camera frames on the camera clock -> LiDAR sweeps covering the same span (at least one)."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if not (math.isfinite(camera_fps) and camera_fps > 0 and math.isfinite(lidar_fps) and lidar_fps > 0):
        raise ValueError(f"camera_fps and lidar_fps must be finite and positive, got {camera_fps}, {lidar_fps}.")
    return max(1, round(num_frames * lidar_fps / camera_fps))


def lidar_sweep_ids(num_sweeps: int, *, camera_fps: float, lidar_fps: float) -> tuple[int, ...]:
    """Camera-clock source frame id of every sweep (``round(i * camera_fps / lidar_fps)``)."""
    if num_sweeps < 1:
        raise ValueError(f"num_sweeps must be positive, got {num_sweeps}.")
    stride = camera_fps / lidar_fps
    return tuple(round(index * stride) for index in range(num_sweeps))


@dataclass(frozen=True)
class Chunk:
    """One causal step on the shared RGB/LiDAR clock.

    ``rgb_frames`` and ``lidar_sweeps`` are absolute latent/sweep indexes (per
    camera for RGB). ``*_prefix_end`` is the exclusive end of everything up to
    and including this chunk, i.e. how many latents/sweeps exist once it is done.
    """

    step: int
    rgb_frames: tuple[int, ...]
    lidar_sweeps: tuple[int, ...]
    rgb_prefix_end: int
    lidar_prefix_end: int

    @property
    def is_first(self) -> bool:
        return self.step == 0


def build_chunk_schedule(
    *,
    rgb_latents: int,
    lidar_sweeps: int | None,
    frames_per_chunk: int,
    rgb_seconds_per_frame: float,
    lidar_seconds_per_frame: float | None = None,
) -> tuple[Chunk, ...]:
    """Assign every RGB latent and LiDAR sweep to a causal step.

    RGB latent ``f`` belongs to step ``ceil(f / frames_per_chunk - 1e-5)`` and
    LiDAR sweep ``s`` to ``ceil(s * lidar_seconds / chunk_seconds - 1e-5)``,
    which puts latent 0 and sweep 0 alone in step 0 and otherwise fills each
    step with ``frames_per_chunk`` latents and the sweeps of the same span.
    Joint schedules require both sensors in every step.
    """
    if rgb_latents < 1 or frames_per_chunk < 1:
        raise ValueError("The chunk schedule requires a positive latent count and chunk size.")
    if not (math.isfinite(rgb_seconds_per_frame) and rgb_seconds_per_frame > 0):
        raise ValueError(f"rgb_seconds_per_frame must be finite and positive, got {rgb_seconds_per_frame}.")
    chunk_seconds = frames_per_chunk * rgb_seconds_per_frame
    rgb_steps = [math.ceil(frame / frames_per_chunk - 1e-5) for frame in range(rgb_latents)]
    if lidar_sweeps is None:
        lidar_steps: list[int] = []
    else:
        if lidar_sweeps < 1:
            raise ValueError(f"lidar_sweeps must be positive, got {lidar_sweeps}.")
        if lidar_seconds_per_frame is None or not (
            math.isfinite(lidar_seconds_per_frame) and lidar_seconds_per_frame > 0
        ):
            raise ValueError("Joint schedules require a finite positive lidar_seconds_per_frame.")
        lidar_steps = [
            math.ceil(sweep * lidar_seconds_per_frame / chunk_seconds - 1e-5) for sweep in range(lidar_sweeps)
        ]
    chunks: list[Chunk] = []
    for step in sorted(set(rgb_steps) | set(lidar_steps)):
        chunk = Chunk(
            step=step,
            rgb_frames=tuple(frame for frame, value in enumerate(rgb_steps) if value == step),
            lidar_sweeps=tuple(sweep for sweep, value in enumerate(lidar_steps) if value == step),
            rgb_prefix_end=sum(value <= step for value in rgb_steps),
            lidar_prefix_end=sum(value <= step for value in lidar_steps),
        )
        if lidar_sweeps is not None and (not chunk.rgb_frames or not chunk.lidar_sweeps):
            raise ValueError(f"Joint chunk {step} must contain both RGB latents and LiDAR sweeps, got {chunk}.")
        chunks.append(chunk)
    return tuple(chunks)


@dataclass(frozen=True)
class SensorGrid:
    """Transformer token grid of one latent frame (or sweep) after spatial patching."""

    latent_height: int
    latent_width: int
    patch_height: int
    patch_width: int

    def __post_init__(self) -> None:
        for name in ("latent_height", "latent_width", "patch_height", "patch_width"):
            if getattr(self, name) < 1:
                raise ValueError(f"SensorGrid.{name} must be positive, got {getattr(self, name)}.")

    @property
    def grid_height(self) -> int:
        return -(-self.latent_height // self.patch_height)

    @property
    def grid_width(self) -> int:
        return -(-self.latent_width // self.patch_width)

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_height * self.grid_width


def rgb_grid(height: int, width: int, *, vae_spatial_compression: int = 16, patch_size: int = 2) -> SensorGrid:
    """480x832 -> latent 30x52 -> 15x26 = 390 tokens per camera per latent frame."""
    if height % vae_spatial_compression or width % vae_spatial_compression:
        raise ValueError(f"Resolution {height}x{width} must be a multiple of {vae_spatial_compression}.")
    return SensorGrid(height // vae_spatial_compression, width // vae_spatial_compression, patch_size, patch_size)


def lidar_grid(
    *,
    model_height: int = 128,
    model_width: int = 1808,
    spatial_compression: tuple[int, int] = (16, 16),
    patch_size: tuple[int, int] = (2, 2),
) -> SensorGrid:
    """V1.2 LiDAR: 128x1808 -> latent 8x113 -> 4x57 = 228 tokens per sweep."""
    if model_height % spatial_compression[0] or model_width % spatial_compression[1]:
        raise ValueError(f"LiDAR canvas {model_height}x{model_width} must be a multiple of {spatial_compression}.")
    return SensorGrid(
        model_height // spatial_compression[0], model_width // spatial_compression[1], patch_size[0], patch_size[1]
    )


def chunk_token_count(
    chunk: Chunk,
    *,
    num_views: int,
    rgb_tokens_per_frame: int,
    lidar_tokens_per_sweep: int | None,
    with_controls: bool = True,
) -> int:
    """Packed GEN tokens of one chunk: ``[RGB control, RGB target, LiDAR control, LiDAR target]``."""
    if num_views < 1:
        raise ValueError(f"num_views must be positive, got {num_views}.")
    items = 2 if with_controls else 1
    tokens = items * num_views * len(chunk.rgb_frames) * rgb_tokens_per_frame
    if chunk.lidar_sweeps:
        if lidar_tokens_per_sweep is None:
            raise ValueError("A joint chunk needs lidar_tokens_per_sweep.")
        tokens += items * len(chunk.lidar_sweeps) * lidar_tokens_per_sweep
    return tokens


def max_chunk_tokens(
    schedule: tuple[Chunk, ...],
    *,
    num_views: int,
    rgb_tokens_per_frame: int,
    lidar_tokens_per_sweep: int | None,
    with_controls: bool = True,
) -> int:
    return max(
        chunk_token_count(
            chunk,
            num_views=num_views,
            rgb_tokens_per_frame=rgb_tokens_per_frame,
            lidar_tokens_per_sweep=lidar_tokens_per_sweep,
            with_controls=with_controls,
        )
        for chunk in schedule
    )


def paged_frame_unit(tokens: int, alignment: int = PAGED_TOKEN_ALIGNMENT) -> int:
    """Smallest multiple of the kernel page alignment that holds one chunk.

    The AR-Diffusion pool stores one chunk per frame unit; keeping the unit a
    multiple of the page size makes block == frame unit, so the sink/window
    eviction never shares a page between two chunks and nothing leaks.
    """
    if tokens < 1 or alignment < 1:
        raise ValueError("paged_frame_unit needs positive token count and alignment.")
    return -(-tokens // alignment) * alignment
