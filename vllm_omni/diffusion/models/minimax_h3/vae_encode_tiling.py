# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from torch import nn

encode_tile_batch_size: ContextVar[int] = ContextVar("minimax_h3_encode_tile_batch_size", default=1)


@contextmanager
def reference_video_tile_batching(batch_size: int) -> Iterator[None]:
    token = encode_tile_batch_size.set(batch_size)
    try:
        yield
    finally:
        encode_tile_batch_size.reset(token)


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
