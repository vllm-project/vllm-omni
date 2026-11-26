# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from torch import nn


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
        if forward_fn != model.encode:
            return run_tile_tasks(tiles, tile_indices, forward_fn, stack_tiling, cls_agg)
        results = []
        for start in range(0, len(tile_indices), batch_size):
            indices = tile_indices[start : start + batch_size]
            results.extend(run_tile_tasks(tiles, indices, forward_fn, len(indices) > 1, cls_agg))
        return results

    model._run_tile_tasks = run_encode_tile_tasks
