# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.vae_encode_tiling import configure_video_encode_tiling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_encode_tile_batches_preserve_rank_order_and_leave_decoder_unchanged():
    class Model(nn.Module):
        tile_size = decoder_tile_size = 256
        tile_overlap_min = decoder_tile_overlap_min = 64
        vae_ratio = 16
        encoder_parallel = False

        def encode(self, tensor):
            return tensor + 1

        def decode(self, tensor):
            return tensor - 1

        def _run_tile_tasks(self, tiles, indices, forward_fn, stack_tiling, cls_agg):
            calls.append((list(indices), stack_tiling, cls_agg))
            return [forward_fn(tiles[index]) for index in indices]

    model = Model()
    calls = []
    tiles = [torch.arange(2) + i for i in range(5)]
    indices = [0, 2, 4]
    aggregator = object()
    configure_video_encode_tiling(model, batch_size=2, tile_size=288, tile_overlap=16)
    encoded = model._run_tile_tasks(tiles, indices, model.encode, False, aggregator)
    assert calls == [([0, 2], True, aggregator), ([4], False, aggregator)]
    torch.testing.assert_close(encoded, [tiles[i] + 1 for i in indices])
    model._run_tile_tasks(tiles, indices, model.decode, True, None)
    assert calls[-1] == (indices, True, None)
    assert (model.tile_size, model.tile_overlap_min) == (288, 16)
    assert (model.decoder_tile_size, model.decoder_tile_overlap_min) == (256, 64)
