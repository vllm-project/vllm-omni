# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 temporal/spatial tile scheduling, decoder batching, and ordered assembly."""

from __future__ import annotations

import types
from contextlib import contextmanager
from functools import lru_cache

import torch


@lru_cache(None)
def jobs(windows=21, tiles=28, world=8):
    assert (windows, tiles, world) == (21, 28, 8)
    rounds = []
    for first in range(0, windows, 2):
        count = min(2, windows - first) * tiles
        rounds.append(
            tuple(tuple((first + i // tiles, i % tiles) for i in range(r, count, world)) for r in range(world))
        )
    counts = tuple(sum(len(row[r]) for row in rounds) for r in range(world))
    assert counts == (74, 74, 74, 74, 73, 73, 73, 73)
    flat = [job for row in rounds for rank in row for job in rank]
    assert len(flat) == len(set(flat)) == windows * tiles
    assert set(flat) == {(w, t) for w in range(windows) for t in range(tiles)}
    return tuple(rounds)


def geometry(model, latent):
    assert tuple(latent.shape) == (1, 24, 107, 44, 80) and latent.dtype == torch.float32
    assert model.vae_ratio == 16 and model.vae_ratio_t == 4
    assert model.tokens_chunk_size == 5 and model.token_overlap == 2
    assert model.frame_pre_padding == 3 and model.frame_overlap == 5 and model.token_drop == 3
    assert not model.training and not model.stack_tiling and not model.decoder_parallel
    assert model.decoder_tiling and model.decoder_tile_size == 256
    yi, yl, yo = model.split_tiles(704, True)
    xi, xl, xo = model.split_tiles(1280, True)
    assert len(yi) == 4 and len(xi) == 7 and set(yl + xl) == {256}
    slices = tuple(
        (slice(y // 16, (y + length) // 16), slice(x // 16, (x + k) // 16))
        for y, length in zip(yi, yl)
        for x, k in zip(xi, xl)
    )
    return slices, tuple(yo), tuple(xo)


def assemble_spatial(model, tiles, y_overlap, x_overlap):
    # Same sequence as checkpoint AutoencoderKL.tiled_decode after gathering:
    # vertical blend, horizontal blend, trims, horizontal cats, vertical cat.
    rows = [list(tiles[i * 7 : (i + 1) * 7]) for i in range(4)]
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = model.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = model.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def decode_pairs(model, latent, group, callback, *, temporal_cat_dtype=None, gather_stream=None):
    """Group wrapper requires world_size/rank_in_group/ranks/device_group/all_reduce."""
    from vllm_omni.diffusion.models.minimax_h3.vae_parallel import (
        _agree_on_failure,
        _gather_stack_to_rank_zero,
        _LeaderAssembler,
    )

    assert group.world_size == 8 and (callback is not None) == (group.rank_in_group == 0)
    spatial, y_overlap, x_overlap = geometry(model, latent)
    schedule = jobs()
    rank = group.rank_in_group
    assembler = (
        _LeaderAssembler(model, callback, main_frames=17, overlap_frames=5, output_frames=362, device=latent.device)
        if callback is not None
        else None
    )
    local_error = None
    tile_shape = (1, 3, 22, 256, 256)
    tile_dtype = torch.float32
    tile_count = 0
    try:
        for round_id, round_jobs in enumerate(schedule):
            width = max(map(len, round_jobs))
            # Uniform collective shape, including four padding tiles in the
            # final round. Padding is never decoded or consumed by assembly.
            packed = torch.empty((width, *tile_shape), device=latent.device, dtype=tile_dtype)
            for slot, (window, tile) in enumerate(round_jobs[rank]):
                if local_error is not None:
                    packed[slot].zero_()
                    continue
                try:
                    sy, sx = spatial[tile]
                    clip = latent[:, :, window * 5 : window * 5 + 7, sy, sx]
                    with torch.cuda.nvtx.range(f"h3.vae.pair.tile.{window}.{tile}"):
                        decoded = model.decode(clip)
                    assert decoded.shape == (1, 3, 28, 256, 256) and decoded.dtype == tile_dtype
                    # Blends are per frame. Gather only the original live
                    # ranges, preserving FP32 until spatial blending finishes.
                    packed[slot, :, :, :17].copy_(decoded[:, :, 3:20])
                    packed[slot, :, :, 17:].copy_(decoded[:, :, 23:28])
                    tile_count += 1
                except BaseException as e:
                    e.__traceback__ = None
                    local_error = e
                    packed[slot].zero_()
            if len(round_jobs[rank]) < width:
                packed[len(round_jobs[rank]) :].zero_()
            if assembler is not None:
                assembler.drain_previous_round()
            with torch.cuda.nvtx.range(f"h3.vae.pair.gather.{round_id}"):
                if gather_stream is None:
                    gathered = _gather_stack_to_rank_zero(packed, group)
                else:
                    from .vae_parallel import gather_on_stream

                    gathered = gather_on_stream(packed, group, gather_stream)
            if assembler is not None and gathered is not None:
                # Assemble on the dedicated output stream, allowing the next
                # pair's decoder to run concurrently. The original temporal
                # assembler preserves callback order and the five-frame tail.
                stream = assembler._stream
                stream.wait_stream(
                    gather_stream
                    if gather_stream is not None
                    else torch.get_device_module().current_stream(latent.device)
                )
                gathered.record_stream(stream)
                try:
                    with torch.get_device_module().stream(stream):
                        for window in range(round_id * 2, min(round_id * 2 + 2, 21)):
                            parts = [None] * 28
                            for owner, owner_jobs in enumerate(round_jobs):
                                for slot, (w, t) in enumerate(owner_jobs):
                                    if w == window:
                                        parts[t] = gathered[owner, slot]
                            assert all(t is not None for t in parts)
                            segment = assemble_spatial(model, parts, y_overlap, x_overlap)
                            if temporal_cat_dtype is not None:
                                segment = segment.to(temporal_cat_dtype)
                            assembler.push(segment)
                        assembler.finish_round()
                except BaseException as e:
                    assembler._remember_error(e)
            del gathered, packed
        if assembler is not None:
            assembler.finalize()
    finally:
        # Include non-output ranks and exceptional exits. No gather may still
        # access a send/receive allocation when this request is released or
        # the all-rank failure agreement enters the original caller stream.
        try:
            if gather_stream is not None:
                gather_stream.synchronize()
        finally:
            if assembler is not None:
                assembler.synchronize()
    callback_error = assembler.error if assembler is not None else None
    if _agree_on_failure(group, latent.device, local_error is not None or callback_error is not None):
        raise RuntimeError("Paired VAE failed after draining collectives") from (local_error or callback_error)
    assert tile_count == sum(len(row[rank]) for row in schedule)
    if gather_stream is not None:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "H3_VAE_GATHER_OVERLAP rank=%d gathers=11 separate_stream=1 "
            "send_record_stream=1 decoder_calls=42 tiles=%d full_vae=1 "
            "mixed=1 gather_drained=1 frames=362 complete=1",
            rank,
            tile_count,
        )
    return dict(rank=rank, tile_calls=tile_count, gathers=11, windows=21, padding_tiles=4, frames=362)


@lru_cache(None)
def batch_plan(rank):
    return tuple(
        tuple(round_jobs[rank][i : i + 4]) for round_jobs in jobs() for i in range(0, len(round_jobs[rank]), 4)
    )


@contextmanager
def mixed_decode(model, latent, rank):
    """Yield measured calls, without changing existing exact-op counters.

    Each original paired decode call still consumes its original tile. At the
    first call in a B4 group, compute the group together and retain its outputs
    until their original consumers run. Groups never cross a gather boundary. The only linear
    override executes attention to_out at its original B1 shape and stride.
    """
    spatial, _, _ = geometry(model, latent)
    plan = batch_plan(rank)
    ordered = tuple(job for batch in plan for job in batch)
    original_decode = model.decode
    originals = []
    stats = dict(tile_outputs=0, decoder_calls=0, batch_sizes=[], to_out_calls=0)
    pending = []
    batch_index = 0

    def tile_view(job):
        window, tile = job
        sy, sx = spatial[tile]
        return latent[:, :, window * 5 : window * 5 + 7, sy, sx]

    def decode(self, clip):
        nonlocal batch_index
        expected = tile_view(ordered[stats["tile_outputs"]])
        assert (clip.data_ptr(), clip.shape, clip.stride(), clip.dtype) == (
            expected.data_ptr(),
            expected.shape,
            expected.stride(),
            expected.dtype,
        )
        if not pending:
            batch = plan[batch_index]
            batch_index += 1
            x = clip if len(batch) == 1 else torch.cat([tile_view(job) for job in batch], dim=0)
            decoded = original_decode(x)
            assert decoded.shape == (len(batch), 3, 28, 256, 256)
            assert decoded.dtype == torch.float32
            pending.extend(decoded[i : i + 1] for i in range(len(batch)))
            stats["decoder_calls"] += 1
            stats["batch_sizes"].append(len(batch))
        stats["tile_outputs"] += 1
        return pending.pop(0)

    try:
        for block in model.decoder.transformer_blocks:
            module = block.attn.to_out
            original = module.forward
            originals.append((module, original))

            def forward(self, x, _original=original):
                assert x.ndim == 3 and x.shape[1:] == (1797, 2048)
                assert x.dtype == torch.float16 and x.stride() == (1797 * 2048, 2048, 1)
                stats["to_out_calls"] += x.shape[0]
                if x.shape[0] == 1:
                    return _original(x)
                return torch.cat([_original(x[i : i + 1]) for i in range(x.shape[0])], dim=0)

            module.forward = types.MethodType(forward, module)
        model.decode = types.MethodType(decode, model)
        yield stats
        assert not pending and batch_index == len(plan)
        assert stats["tile_outputs"] == len(ordered)
        assert stats["decoder_calls"] == 21
        assert stats["to_out_calls"] == 36 * len(ordered)
    finally:
        model.decode = original_decode
        for module, original in originals:
            module.forward = original
