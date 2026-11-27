# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.vae_encode_tiling import (
    configure_video_encode_tiling,
    reference_video_tile_batching,
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
    configure_video_encode_tiling(model, batch_size=4, tile_size=288, tile_overlap=16)
    calls.clear()
    with reference_video_tile_batching(4):
        encoded = _encode_temporal_tail_pad(model, video)
        assert calls == [[0, 4, 1, 5], [8, 9]]
        tile = video[..., :1]
        torch.testing.assert_close(model._run_tile_tasks([tile], [0], model.decode, False), [tile - 1])
    torch.testing.assert_close(encoded, baseline, rtol=0, atol=0)
    calls.clear()
    torch.testing.assert_close(_encode_temporal_tail_pad(model, video), baseline, rtol=0, atol=0)
    assert calls == [[0], [1], [4], [5], [8], [9]]
    assert (model.tile_size, model.tile_overlap_min) == (288, 16)
    assert (model.decoder_tile_size, model.decoder_tile_overlap_min) == (256, 64)
