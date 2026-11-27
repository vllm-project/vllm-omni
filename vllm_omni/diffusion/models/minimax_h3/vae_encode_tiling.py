# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.distributed as dist
from torch import nn

encode_tile_batch_size: ContextVar[int] = ContextVar("minimax_h3_encode_tile_batch_size", default=1)
_encode_tile_group: ContextVar[dist.ProcessGroup | None] = ContextVar("minimax_h3_encode_tile_group", default=None)


@contextmanager
def reference_video_tile_batching(batch_size: int, group: dist.ProcessGroup | None = None) -> Iterator[None]:
    batch_token = encode_tile_batch_size.set(batch_size)
    group_token = _encode_tile_group.set(group)
    try:
        yield
    finally:
        _encode_tile_group.reset(group_token)
        encode_tile_batch_size.reset(batch_token)


def _gather_clip_tiles(
    local: list[torch.Tensor],
    tile_count: int,
    clip_count: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> list[torch.Tensor]:
    """Gather round-robin tasks and restore the checkpoint's spatial-tile order."""
    world_size = dist.get_world_size(group)
    task_count = tile_count * clip_count
    rank_capacity = (task_count + world_size - 1) // world_size
    metadata = (local[0].shape, local[0].dtype) if local else None
    if task_count < world_size:
        # Rank 0 owns task 0. Empty ranks learn the format without encoding a
        # dummy tile; all ranks still use the same non-empty HCCL payload.
        header = [metadata]
        dist.broadcast_object_list(header, group=group, group_src=0, device=device)
        metadata = header[0]
    if metadata is None:  # The checkpoint emitted no CLS tokens.
        return []

    shape, dtype = metadata
    send = torch.empty((rank_capacity, *shape), dtype=dtype, device=device)
    if local:
        torch.stack(local, out=send[: len(local)])
    send[len(local) :].zero_()
    received = send.new_empty((world_size * rank_capacity, *shape))
    dist.all_gather_into_tensor(received, send, group=group)
    received = received.unflatten(0, (world_size, rank_capacity))

    # Native blending and CLS aggregation expect one tensor per spatial tile
    # with the (sample, clip) axes flattened together, not clip-major tasks.
    return [
        torch.stack(
            [received[task % world_size, task // world_size] for task in range(tile, task_count, tile_count)], dim=1
        ).flatten(0, 1)
        for tile in range(tile_count)
    ]


def encode_clip_group(model: nn.Module, clips: torch.Tensor, clip_count: int) -> torch.Tensor:
    group = _encode_tile_group.get()
    if group is None:
        return model._adaptive_encode(clips)

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    run_tile_tasks = model._run_tile_tasks

    def run_clip_tiles(tiles, _tile_indices, forward_fn, stack_tiling, cls_agg):
        tile_count = len(tiles)
        by_clip = [tile.unflatten(0, (-1, clip_count)) for tile in tiles]
        local_tiles = []
        for task in range(rank, clip_count * tile_count, world_size):
            clip, tile = divmod(task, tile_count)
            local_tiles.append(by_clip[tile][:, clip])
        local = (
            run_tile_tasks(local_tiles, range(len(local_tiles)), forward_fn, stack_tiling, cls_agg)
            if local_tiles
            else []
        )
        encoded = _gather_clip_tiles(local, tile_count, clip_count, group, clips.device)
        cls_agg.cls_tokens = _gather_clip_tiles(cls_agg.cls_tokens, tile_count, clip_count, group, clips.device)
        return encoded

    # Let the checkpoint build and blend its spatial grid, but replace its
    # spatial-only dispatch/gather for this window. Don't change group state.
    parallel_tiling = model.parallel_tiling
    model.parallel_tiling = False
    model._run_tile_tasks = run_clip_tiles
    try:
        return model._adaptive_encode(clips)
    finally:
        model._run_tile_tasks = run_tile_tasks
        model.parallel_tiling = parallel_tiling


def configure_video_encode_tiling(
    model: nn.Module, *, batch_size: int, tile_size: int | None, tile_overlap: int | None
) -> None:
    """Tune the encoder grid and bound the checkpoint's rank-local tile batch."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("video_vae_encode_tile_batch_size must be a positive integer")
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
        model.tile_size, model.tile_overlap_min = size, overlap
    if batch_size == 1 or model.encoder_parallel:
        return

    run_tile_tasks = model._run_tile_tasks

    def run_encode_tile_tasks(tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        sample_limit = encode_tile_batch_size.get()
        if sample_limit == 1 or forward_fn != model.encode:
            return run_tile_tasks(tiles, tile_indices, forward_fn, stack_tiling, cls_agg)
        # Each spatial tile may already contain several temporal clips on its
        # sample axis. Count those samples against the same forward budget.
        tiles_per_forward = max(1, sample_limit // tiles[0].shape[0])
        results = []
        for start in range(0, len(tile_indices), tiles_per_forward):
            indices = tile_indices[start : start + tiles_per_forward]
            results.extend(run_tile_tasks(tiles, indices, forward_fn, len(indices) > 1, cls_agg))
        return results

    model._run_tile_tasks = run_encode_tile_tasks
