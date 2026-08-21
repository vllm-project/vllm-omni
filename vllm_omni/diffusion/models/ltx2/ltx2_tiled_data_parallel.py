# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Spatial tiled data parallelism for the high-resolution LTX denoise phase."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.forward_context import set_sequence_parallel_enabled


@dataclass(frozen=True)
class LTXTileInterval:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LTXSpatialTile:
    index: int
    row: int
    column: int
    height: LTXTileInterval
    width: LTXTileInterval
    token_indices: torch.Tensor
    blend_mask: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.token_indices.numel())


@dataclass(frozen=True)
class LTXTiledDataParallelPlan:
    num_frames: int
    height: int
    width: int
    grid_rows: int
    grid_columns: int
    overlap: int
    rank: int
    world_size: int
    tiles: tuple[LTXSpatialTile, ...]

    @property
    def local_tiles(self) -> tuple[LTXSpatialTile, ...]:
        return tuple(self.tiles[index] for index in range(self.rank, len(self.tiles), self.world_size))

    @property
    def token_count(self) -> int:
        return self.num_frames * self.height * self.width


def _factor_grid(world_size: int, height: int, width: int) -> tuple[int, int]:
    if world_size < 1:
        raise ValueError(f"LTX tiled data parallel world size must be positive, got {world_size}.")
    del height, width
    rows = math.isqrt(world_size)
    while world_size % rows:
        rows -= 1
    return rows, world_size // rows


def _split_dimension(size: int, tile_count: int, overlap: int) -> tuple[LTXTileInterval, ...]:
    if tile_count < 1:
        raise ValueError(f"Tile count must be positive, got {tile_count}.")
    if overlap < 0:
        raise ValueError(f"Tile overlap must be non-negative, got {overlap}.")
    if tile_count == 1:
        return (LTXTileInterval(0, size),)

    total = size + overlap * (tile_count - 1)
    tile_size, remainder = divmod(total, tile_count)
    if tile_size <= overlap:
        raise ValueError(
            f"Cannot split dimension {size} into {tile_count} tiles with overlap {overlap}; "
            "each tile must contain non-overlap tokens."
        )

    intervals: list[LTXTileInterval] = []
    start = 0
    for index in range(tile_count):
        current_size = tile_size + int(index < remainder)
        end = start + current_size
        intervals.append(LTXTileInterval(start, end))
        start = end - overlap
    if intervals[-1].end != size:
        raise RuntimeError(f"Internal LTX tile split error: final end {intervals[-1].end} != dimension {size}.")
    return tuple(intervals)


def _trapezoid_mask(
    interval: LTXTileInterval,
    full_size: int,
    overlap: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(interval.size, dtype=torch.float32, device=device)
    if overlap == 0:
        return mask
    ramp = torch.linspace(0.0, 1.0, overlap + 2, dtype=torch.float32, device=device)[1:-1]
    if interval.start > 0:
        mask[:overlap] = ramp
    if interval.end < full_size:
        mask[-overlap:] = ramp.flip(0)
    return mask


def build_spatial_tiling_plan(
    *,
    num_frames: int,
    height: int,
    width: int,
    world_size: int,
    rank: int,
    overlap: int = 5,
    device: torch.device | str = "cpu",
) -> LTXTiledDataParallelPlan:
    if not 0 <= rank < world_size:
        raise ValueError(f"LTX tiled data parallel rank {rank} is outside world size {world_size}.")
    rows, columns = _factor_grid(world_size, height, width)
    height_intervals = _split_dimension(height, rows, overlap)
    width_intervals = _split_dimension(width, columns, overlap)
    device = torch.device(device)

    tiles: list[LTXSpatialTile] = []
    for row, height_interval in enumerate(height_intervals):
        height_mask = _trapezoid_mask(height_interval, height, overlap, device=device)
        for column, width_interval in enumerate(width_intervals):
            width_mask = _trapezoid_mask(width_interval, width, overlap, device=device)
            frame_indices = torch.arange(num_frames, device=device)[:, None, None]
            height_indices = torch.arange(height_interval.start, height_interval.end, device=device)[None, :, None]
            width_indices = torch.arange(width_interval.start, width_interval.end, device=device)[None, None, :]
            token_indices = (frame_indices * height * width + height_indices * width + width_indices).reshape(-1)
            spatial_mask = height_mask[:, None] * width_mask[None, :]
            blend_mask = spatial_mask.unsqueeze(0).expand(num_frames, -1, -1).reshape(-1)
            tiles.append(
                LTXSpatialTile(
                    index=len(tiles),
                    row=row,
                    column=column,
                    height=height_interval,
                    width=width_interval,
                    token_indices=token_indices,
                    blend_mask=blend_mask,
                )
            )

    return LTXTiledDataParallelPlan(
        num_frames=num_frames,
        height=height,
        width=width,
        grid_rows=rows,
        grid_columns=columns,
        overlap=overlap,
        rank=rank,
        world_size=world_size,
        tiles=tuple(tiles),
    )


def _slice_video_coords(video_coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    coords = video_coords.index_select(2, indices).clone()
    origins = coords[..., 0].amin(dim=2, keepdim=True)
    return coords - origins.unsqueeze(-1)


def _blend_tile_output(
    full_output: torch.Tensor,
    tile: LTXSpatialTile,
    tile_output: torch.Tensor,
) -> None:
    weights = tile.blend_mask.to(device=tile_output.device, dtype=tile_output.dtype).view(1, -1, 1)
    full_output.index_add_(1, tile.token_indices, tile_output * weights)


def forward_tiled_data_parallel(
    transformer: Any,
    kwargs: dict[str, Any],
    plan: LTXTiledDataParallelPlan,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run local spatial tiles, blend them, then reduce AV predictions."""
    hidden_states = kwargs["hidden_states"]
    if hidden_states.shape[1] != plan.token_count:
        raise ValueError(f"LTX TDP expected {plan.token_count} packed video tokens, got {hidden_states.shape[1]}.")

    video_output = None
    audio_output = None
    for tile in plan.local_tiles:
        indices = tile.token_indices
        tile_kwargs = dict(kwargs)
        tile_kwargs["hidden_states"] = hidden_states.index_select(1, indices)
        tile_kwargs["video_coords"] = _slice_video_coords(kwargs["video_coords"], indices)
        tile_kwargs["height"] = tile.height.size
        tile_kwargs["width"] = tile.width.size

        keyframes_mask = kwargs.get("keyframes_mask")
        if keyframes_mask is not None:
            tile_kwargs["keyframes_mask"] = keyframes_mask.index_select(1, indices)
        timestep = kwargs.get("timestep")
        if isinstance(timestep, torch.Tensor) and timestep.ndim > 1 and timestep.shape[1] == plan.token_count:
            tile_kwargs["timestep"] = timestep.index_select(1, indices)

        # The tile inputs are sliced from the full Stage-2 latent. LTX's
        # multi-branch SP plan can retain a non-zero activity depth after the
        # completed Stage-1 forward, so explicitly switch the reused SP group to
        # local tile execution here.
        with set_sequence_parallel_enabled(False, allow_active_shard_depth=True):
            tile_video, tile_audio = transformer(**tile_kwargs)
        if video_output is None:
            video_output = tile_video.new_zeros(
                tile_video.shape[0],
                plan.token_count,
                tile_video.shape[-1],
            )
        if audio_output is None:
            audio_output = torch.zeros_like(tile_audio)
        _blend_tile_output(video_output, tile, tile_video)
        audio_output.add_(tile_audio)

    if video_output is None or audio_output is None:
        raise RuntimeError(f"LTX TDP rank {plan.rank} was not assigned a tile.")
    video_output = group.all_reduce(video_output.contiguous())
    audio_output = group.all_reduce(audio_output.contiguous())
    audio_output.div_(len(plan.tiles))
    return video_output, audio_output
