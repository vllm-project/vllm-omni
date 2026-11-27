# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

encode_tile_batch_size: ContextVar[int] = ContextVar("minimax_h3_encode_tile_batch_size", default=1)
_encode_tile_group: ContextVar[dist.ProcessGroup | None] = ContextVar("minimax_h3_encode_tile_group", default=None)
_clip_count: ContextVar[int] = ContextVar("minimax_h3_encode_clip_count", default=1)
_TileLayout = tuple[torch.Size, torch.dtype]


@contextmanager
def reference_video_tile_batching(batch_size: int, group: dist.ProcessGroup | None = None) -> Iterator[None]:
    batch_token = encode_tile_batch_size.set(batch_size)
    group_token = _encode_tile_group.set(group)
    try:
        yield
    finally:
        _encode_tile_group.reset(group_token)
        encode_tile_batch_size.reset(batch_token)


@contextmanager
def reference_video_tile_geometry(model: nn.Module, geometry: tuple[int, int] | None) -> Iterator[None]:
    """Apply the configured encoder tile size and overlap to reference video encoding only."""
    if geometry is None:
        yield
        return
    checkpoint_geometry = model.tile_size, model.tile_overlap_min
    model.tile_size, model.tile_overlap_min = geometry
    try:
        yield
    finally:
        model.tile_size, model.tile_overlap_min = checkpoint_geometry


def _gather_clip_tiles(
    local: list[torch.Tensor],
    layout: _TileLayout,
    tile_count: int,
    clip_count: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> list[torch.Tensor]:
    """Gather clip-major tasks and restore sample-major batches per spatial tile."""
    world_size = dist.get_world_size(group)
    task_count = tile_count * clip_count
    rank_capacity = (task_count + world_size - 1) // world_size
    shape, dtype = layout
    send = torch.empty((rank_capacity, *shape), dtype=dtype, device=device)
    if local:
        torch.stack(local, out=send[: len(local)])
    send[len(local) :].zero_()
    received = send.new_empty((world_size * rank_capacity, *shape))
    dist.all_gather_into_tensor(received, send, group=group)
    received = received.unflatten(0, (world_size, rank_capacity))
    return [
        torch.stack(
            [received[task % world_size, task // world_size] for task in range(tile, task_count, tile_count)], dim=1
        ).flatten(0, 1)
        for tile in range(tile_count)
    ]


def _gather_clip_results(
    moments: list[torch.Tensor],
    cls_tokens: list[torch.Tensor],
    tile_count: int,
    clip_count: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Gather moments and CLS tokens with a layout every rank takes from rank 0."""
    layouts = [(values[0].shape, values[0].dtype) if values else None for values in (moments, cls_tokens)]
    header = [tuple(layouts)]
    dist.broadcast_object_list(header, group=group, group_src=0, device=device)
    for values, layout in zip((moments, cls_tokens), header[0]):
        if values and (values[0].shape, values[0].dtype) != layout:
            raise ValueError("MiniMax H3 encoder tile ranks returned different per-task tensor layouts")
    moment_layout, cls_layout = header[0]
    return (
        _gather_clip_tiles(moments, moment_layout, tile_count, clip_count, group, device),
        _gather_clip_tiles(cls_tokens, cls_layout, tile_count, clip_count, group, device) if cls_layout else [],
    )


def encode_clip_group(model: nn.Module, clips: torch.Tensor, clip_count: int) -> torch.Tensor:
    """Encode independent clips as one checkpoint tile pass, grouped when a tile group is set."""
    if _encode_tile_group.get() is None:
        return model._adaptive_encode(clips)
    parallel_tiling = model.parallel_tiling
    token = _clip_count.set(clip_count)
    model.parallel_tiling = False
    try:
        return model._adaptive_encode(clips)
    finally:
        model.parallel_tiling = parallel_tiling
        _clip_count.reset(token)


def _run_batched_tiles(run_tile_tasks, tiles, indices, forward_fn, cls_agg, sample_limit):
    tiles_per_forward = max(1, sample_limit // tiles[0].shape[0])
    results = []
    for start in range(0, len(indices), tiles_per_forward):
        batch = indices[start : start + tiles_per_forward]
        results.extend(run_tile_tasks(tiles, batch, forward_fn, len(batch) > 1, cls_agg))
    return results


def _run_encode_tile_tasks(model, run_tile_tasks, tiles, indices, forward_fn, stack_tiling, cls_agg=None):
    sample_limit = encode_tile_batch_size.get()
    if sample_limit == 1 or forward_fn != model.encode:
        return run_tile_tasks(tiles, indices, forward_fn, stack_tiling, cls_agg)
    group = _encode_tile_group.get()
    clip_count = _clip_count.get()
    if group is None or clip_count == 1:
        return _run_batched_tiles(run_tile_tasks, tiles, indices, forward_fn, cls_agg, sample_limit)

    tile_count = len(tiles)
    by_clip = [tile.unflatten(0, (-1, clip_count)) for tile in tiles]
    local_tiles = []
    for task in range(dist.get_rank(group), clip_count * tile_count, dist.get_world_size(group)):
        clip, tile = divmod(task, tile_count)
        local_tiles.append(by_clip[tile][:, clip])
    local = (
        _run_batched_tiles(run_tile_tasks, local_tiles, range(len(local_tiles)), forward_fn, cls_agg, sample_limit)
        if local_tiles
        else []
    )
    encoded, cls_agg.cls_tokens = _gather_clip_results(
        local, cls_agg.cls_tokens, tile_count, clip_count, group, tiles[0].device
    )
    return encoded


def configure_video_encode_tiling(
    model: nn.Module, *, batch_size: int, tile_size: int | None, tile_overlap: int | None
) -> tuple[int, int] | None:
    """Validate video-only geometry and install the bounded encode task runner."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("video_vae_encode_tile_batch_size must be a positive integer")
    geometry = None
    if tile_size is not None or tile_overlap is not None:
        size = model.tile_size if tile_size is None else tile_size
        overlap = model.tile_overlap_min if tile_overlap is None else tile_overlap
        ratio = model.vae_ratio
        if (
            type(size) is not int
            or type(overlap) is not int
            or not 0 < overlap < size
            or size % ratio
            or overlap % ratio
        ):
            raise ValueError(f"Video VAE encoder needs integer 0 < overlap < tile size, both divisible by {ratio}")
        geometry = size, overlap
    if batch_size > 1 and not model.encoder_parallel:
        model._run_tile_tasks = partial(_run_encode_tile_tasks, model, model._run_tile_tasks)
    return geometry
