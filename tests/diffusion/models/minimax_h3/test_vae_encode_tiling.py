# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.vae_encode_tiling import (
    _gather_clip_results,
    configure_video_encode_tiling,
    reference_video_tile_batching,
    reference_video_tile_geometry,
)
from vllm_omni.diffusion.models.minimax_h3.vae_temporal import _encode_temporal_tail_pad

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_reference_encode_batches_across_clips_without_changing_tile_order():
    class Model(nn.Module):
        tile_size = decoder_tile_size = 256
        tile_overlap_min = decoder_tile_overlap_min = 64
        vae_ratio = 16
        encoder_tiling = True
        encoder_parallel = False
        isolated_first_frame = isolated_last_frame = isolated_key_frame = False
        frame_pre_padding = token_drop = 0
        clip_length = 2

        def __init__(self):
            super().__init__()
            self.encoder = nn.Identity()

        def encode(self, tile_batch):
            calls.append(tile_batch[:, 0, 0, 0, 0].tolist())
            return tile_batch + 1

        def decode(self, tile_batch):
            return tile_batch - 1

        def _run_tile_tasks(self, tiles, indices, forward_fn, stack_tiling, cls_agg=None):
            local_tiles = [tiles[index] for index in indices]
            if stack_tiling:
                return list(forward_fn(torch.cat(local_tiles)).split(tiles[0].shape[0]))
            return [forward_fn(tile) for tile in local_tiles]

        def _adaptive_encode(self, frames):
            tiles = frames.split(1, dim=-1)
            return torch.cat(self._run_tile_tasks(tiles, [0, 1], self.encode, False), dim=-1)

    model = Model()
    calls = []
    video = torch.arange(10).reshape(1, 1, 5, 1, 2)
    baseline = _encode_temporal_tail_pad(model, video)
    geometry = configure_video_encode_tiling(model, batch_size=4, tile_size=288, tile_overlap=16)
    calls.clear()
    with reference_video_tile_geometry(model, geometry), reference_video_tile_batching(4):
        encoded = _encode_temporal_tail_pad(model, video)
        assert calls == [[0, 4, 1, 5], [8, 9]]
        tile = video[..., :1]
        torch.testing.assert_close(model._run_tile_tasks([tile], [0], model.decode, False), [tile - 1])
        assert (model.tile_size, model.tile_overlap_min) == (288, 16)
    torch.testing.assert_close(encoded, baseline, rtol=0, atol=0)
    calls.clear()
    torch.testing.assert_close(_encode_temporal_tail_pad(model, video), baseline, rtol=0, atol=0)
    assert calls == [[0], [1], [4], [5], [8], [9]]
    assert (model.tile_size, model.tile_overlap_min) == (256, 64)
    assert (model.decoder_tile_size, model.decoder_tile_overlap_min) == (256, 64)


def _check_clip_tile_gather(rank, rendezvous):
    world_size = 3
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        for tile_count in (1, 2):
            clip_count = 2
            values = torch.arange(2 * clip_count * tile_count).reshape(clip_count, tile_count, 2)
            tasks = values.flatten(0, 1).unbind(0)
            local = list(tasks[rank::world_size])
            spatial, cls = _gather_clip_results(
                local, [tile + 10 for tile in local], tile_count, clip_count, dist.group.WORLD, torch.device("cpu")
            )
            expected = values.permute(1, 2, 0).flatten(1)
            torch.testing.assert_close(torch.stack(spatial), expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.stack(cls), expected + 10, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_clip_tile_gather_keeps_sample_order_with_idle_ranks(tmp_path):
    mp.spawn(_check_clip_tile_gather, args=((tmp_path / "tile-group").as_uri(),), nprocs=3)


@pytest.mark.parametrize(
    ("batch", "size", "overlap"),
    [(0, None, None), (True, None, None), (1, 0, 16), (1, 256, 0), (1, 256, 256), (1, 256, 17), (1, 0.5, 16)],
)
def test_invalid_encode_tile_config(batch, size, overlap):
    model = nn.Identity()
    model.vae_ratio = 16
    with pytest.raises(ValueError):
        configure_video_encode_tiling(model, batch_size=batch, tile_size=size, tile_overlap=overlap)


def test_reference_encode_rejects_zero_aligned_tile_extent():
    model = nn.Identity()
    model.isolated_first_frame = False
    with pytest.raises(ValueError, match="non-zero spatial dimensions after alignment"):
        _encode_temporal_tail_pad(model, torch.empty(1, 3, 2, 0, 16))
