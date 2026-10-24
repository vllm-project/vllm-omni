# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded cross-clip scheduling for the checkpoint's independent VAE tiles."""

from collections import defaultdict
from collections.abc import Callable
from typing import Protocol

import torch
import torch.distributed as dist


class TiledVideoDecoder(Protocol):
    """Checkpoint primitives needed to schedule independent spatial tiles."""

    training: bool
    decoder_parallel: bool
    decoder_tiling: bool
    parallel_tiling: bool
    stack_tiling: bool
    vae_ratio: int

    def split_tiles(self, length: int, decode: bool) -> tuple[list[int], list[int], list[int]]: ...
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
    def _adaptive_decode(self, z: torch.Tensor) -> torch.Tensor: ...
    def blend(self, a: torch.Tensor, b: torch.Tensor, overlap: int, dim: int) -> torch.Tensor: ...
    def _run_tile_tasks(
        self,
        tiles: list[torch.Tensor],
        indices: list[int],
        forward: Callable[[torch.Tensor], torch.Tensor],
        stack: bool,
    ) -> list[torch.Tensor]: ...
    def _all_gather_tiled_results(self, tasks: list[torch.Tensor], count: int) -> list[torch.Tensor]: ...


def validate_temporal_window(model: object) -> None:
    """Fail at initialization when the checkpoint cannot implement the opt-in."""
    methods = ("split_tiles", "_run_tile_tasks", "_all_gather_tiled_results", "decode", "blend")
    missing = [name for name in methods if not callable(getattr(model, name, None))]
    if missing:
        raise ValueError(f"VAE temporal windows require checkpoint tile methods: {missing}")


def _validate_decode_state(model: TiledVideoDecoder, rank: int, world_size: int) -> None:
    if model.training or getattr(model, "decoder_parallel", False):
        raise ValueError("VAE temporal windows require inference with decoder_parallel disabled")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid VAE tile rank {rank} for group size {world_size}")


def temporal_window_enabled(model: TiledVideoDecoder, z: torch.Tensor, *, rank: int, world_size: int) -> bool:
    """Select per request, before buffering clips, using replicated metadata.

    Uniform unstacked tiles evenly divided across ranks have no critical-path
    work reduction from a window. Keep native streaming and its memory bound.
    Stacked tiles can change forward batch sizes, so retain the window there.
    """
    _validate_decode_state(model, rank, world_size)
    if not model.decoder_tiling or not model.parallel_tiling or world_size == 1:
        return False
    if model.stack_tiling:
        return True
    _, heights, _ = model.split_tiles(z.shape[-2] * model.vae_ratio, True)
    _, widths, _ = model.split_tiles(z.shape[-1] * model.vae_ratio, True)
    balanced = len(set(heights)) == len(set(widths)) == 1 and len(heights) * len(widths) % world_size == 0
    return not balanced


def decode_temporal_window(
    model: TiledVideoDecoder,
    clips: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    process_group: dist.ProcessGroup | None,
) -> list[torch.Tensor]:
    """Decode independent tiles, then restore native spatial/temporal order.

    Only equal-shaped tiles share a collective or a native stacked batch.
    A bucket smaller than the process group is decoded on every rank: the
    checkpoint gather rejects empty local task lists. The choice depends only
    on replicated input metadata, so all ranks follow the same collectives.
    """
    _validate_decode_state(model, rank, world_size)
    if not model.decoder_tiling or not model.parallel_tiling or world_size == 1:
        return [model._adaptive_decode(clip) for clip in clips]

    # Each entry keeps its clip and row-major spatial tile identity. Never
    # batch a short tail with a full clip or an edge tile with a different shape.
    buckets: dict[tuple, list[tuple[int, int, torch.Tensor]]] = defaultdict(list)
    layouts = []
    results: list[dict[int, torch.Tensor]] = []
    for clip_index, clip in enumerate(clips):
        ratio = model.vae_ratio
        y_idx, y_len, y_overlap = model.split_tiles(clip.shape[-2] * ratio, True)
        x_idx, x_len, x_overlap = model.split_tiles(clip.shape[-1] * ratio, True)
        layouts.append((len(y_idx), len(x_idx), y_overlap, x_overlap))
        results.append({})
        for i, (y, height) in enumerate(zip(y_idx, y_len)):
            for j, (x, width) in enumerate(zip(x_idx, x_len)):
                tile = clip[..., y // ratio : y // ratio + height // ratio, x // ratio : x // ratio + width // ratio]
                key = (tuple(tile.shape), tile.dtype, tile.device)
                buckets[key].append((clip_index, i * len(x_idx) + j, tile))

    for entries in buckets.values():
        tiles = [entry[2] for entry in entries]
        distributed = len(tiles) >= world_size
        indices = list(range(rank, len(tiles), world_size)) if distributed else list(range(len(tiles)))
        decoded = model._run_tile_tasks(tiles, indices, model.decode, model.stack_tiling)
        if distributed:
            dist.barrier(group=process_group)
            decoded = model._all_gather_tiled_results(decoded, len(tiles))
        if len(decoded) != len(entries):
            raise RuntimeError("VAE tile gather returned the wrong number of tiles")
        for (clip_index, tile_index, _), output in zip(entries, decoded):
            results[clip_index][tile_index] = output.to(clips[clip_index].device)

    outputs = []
    for tiles, (height, width, y_overlap, x_overlap) in zip(results, layouts):
        rows = [[tiles[i * width + j] for j in range(width)] for i in range(height)]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # Match native tiled_decode: blend against the raw neighbors,
                # above before left, then crop overlap. Do not mutate rows.
                if i > 0:
                    tile = model.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = model.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < height - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < width - 1:
                    tile = tile[..., :, : -x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        outputs.append(torch.cat(result_rows, dim=-2))
    return outputs
