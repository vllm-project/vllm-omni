# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the bounded tile stacking in the MiniMax-H3 video VAE decode."""

from contextlib import nullcontext

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeCheckpointModel:
    """The checkpoint's ``stack_tiling`` flag and ``_run_tile_tasks`` loop."""

    def __init__(self):
        self.stack_tiling = False
        self.batches = []

    def _run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        if stack_tiling and tile_indices:
            batch = torch.cat([tiles[idx] for idx in tile_indices], dim=0)
            self.batches.append(len(tile_indices))
            return list(forward_fn(batch).unflatten(0, (len(tile_indices), tiles[0].shape[0])))
        self.batches.extend(1 for _ in tile_indices)
        return [forward_fn(tiles[idx]) for idx in tile_indices]


def _vae(stack=True, cap=4):
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = _FakeCheckpointModel()
    vae.stack_decode_tiles = stack
    vae.max_stacked_decode_tiles = cap
    return vae


def _decode_tiles(vae, num_tiles):
    tiles = [torch.full((1, 2, 3), float(i)) for i in range(num_tiles)]
    with vae._stacked_decode_tiles(nullcontext()):
        model = vae.model
        return model._run_tile_tasks(tiles, list(range(num_tiles)), lambda x: x * 2, model.stack_tiling)


@pytest.mark.parametrize(
    ("stack", "cap", "num_tiles", "batches"),
    [
        (True, 4, 28, [4] * 7),  # 720p on one VAE rank
        (True, 4, 10, [4, 4, 2]),
        (True, 4, 3, [3]),  # e.g. one rank's share of an 8-rank grid
        (True, None, 28, [28]),
        (False, 4, 5, [1] * 5),
    ],
)
def test_stacked_tiles_are_bounded_and_ordered(stack, cap, num_tiles, batches):
    vae = _vae(stack, cap)
    outputs = _decode_tiles(vae, num_tiles)
    assert vae.model.batches == batches
    assert [int(out[0, 0, 0]) for out in outputs] == [2 * i for i in range(num_tiles)]
    assert all(out.shape == (1, 2, 3) for out in outputs)


def test_stacked_tiles_restore_the_checkpoint_loop():
    vae = _vae()
    _decode_tiles(vae, 9)
    assert vae.model.stack_tiling is False
    assert "_run_tile_tasks" not in vae.model.__dict__
    with pytest.raises(RuntimeError):
        with vae._stacked_decode_tiles(nullcontext()):
            raise RuntimeError
    assert vae.model.stack_tiling is False
    assert "_run_tile_tasks" not in vae.model.__dict__


def test_encoder_aggregation_calls_are_not_split():
    vae = _vae(cap=2)
    calls = []
    original = vae.model._run_tile_tasks
    vae.model._run_tile_tasks = lambda *args: calls.append(args) or original(*args[:4])
    tiles = [torch.zeros(1, 2) for _ in range(5)]
    with vae._stacked_decode_tiles(nullcontext()):
        vae.model._run_tile_tasks(tiles, list(range(5)), lambda x: x, True, object())
    assert len(calls) == 1
