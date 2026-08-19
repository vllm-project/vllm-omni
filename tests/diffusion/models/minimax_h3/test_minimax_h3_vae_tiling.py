# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the decoder tile-shortage guard in the MiniMax-H3 video VAE."""

import sys
import types

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TILE_SIZE, OVERLAP, RATIO = 256, 64, 16


class _FakeCheckpointModel:
    """The parts of the checkpoint's AutoencoderKL the guard touches."""

    def __init__(self):
        self.vae_ratio = RATIO
        self.parallel_tiling = True
        self.decoded_with = []

    def split_tiles(self, input_len, is_decoder=False):
        """Same grid arithmetic as the checkpoint, on pixel dimensions."""
        if TILE_SIZE >= input_len:
            return [0], [input_len], []
        n = -(-input_len // TILE_SIZE)
        while TILE_SIZE * n - OVERLAP * (n - 1) - input_len < 0:
            n += 1
        return list(range(n)), [TILE_SIZE] * n, [OVERLAP] * (n - 1)


def _vae(parallel_size, state):
    """A stand-in instance: the guard only reads .model/.remote/.parallel_size."""
    module = types.ModuleType("fake_ckpt.parallel")
    module.get_parallel_state = lambda: state
    sys.modules.setdefault("fake_ckpt", types.ModuleType("fake_ckpt"))
    sys.modules["fake_ckpt.parallel"] = module

    remote = type("Remote", (), {"__module__": "fake_ckpt.klvae"})()

    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = _FakeCheckpointModel()
    vae.remote = remote
    vae.parallel_size = parallel_size
    return vae


def _latent(h, w):
    return torch.zeros(1, 24, 9, h, w)


@pytest.mark.parametrize(
    ("h", "w", "expected"),
    [
        (16, 16, 1),  # 256x256 px -> single tile
        (24, 16, 2),  # 384x256 px
        (24, 24, 4),  # 384x384 px
        (96, 168, 112),  # 1536x2688 px, the shipped H3 resolution
    ],
)
def test_tile_count_matches_the_checkpoint_grid(h, w, expected):
    assert MiniMaxH3VideoVAE._decoder_tile_count(_vae(1, {}), _latent(h, w)) == expected


def test_rank_local_tiling_restores_the_group_state():
    state = {"sp_size": 4, "sp_rank": 2, "sp_enabled": True, "sp_process_group": "pg"}
    vae = _vae(4, state)

    with vae._rank_local_tiling():
        assert state["sp_size"] == 1
        assert state["sp_rank"] == 0
        assert state["sp_enabled"] is False
        assert state["sp_process_group"] is None
        assert vae.model.parallel_tiling is False

    assert state == {"sp_size": 4, "sp_rank": 2, "sp_enabled": True, "sp_process_group": "pg"}
    assert vae.model.parallel_tiling is True


def test_rank_local_tiling_restores_after_an_exception():
    state = {"sp_size": 4, "sp_rank": 1, "sp_enabled": True, "sp_process_group": "pg"}
    vae = _vae(4, state)

    with pytest.raises(RuntimeError, match="decode failed"):
        with vae._rank_local_tiling():
            raise RuntimeError("decode failed")

    assert state["sp_size"] == 4
    assert vae.model.parallel_tiling is True


@pytest.mark.parametrize(
    ("parallel_size", "h", "w", "falls_back"),
    [
        (4, 24, 16, True),  # 2 tiles, ranks 2 and 3 would get none
        (4, 16, 16, True),  # 1 tile
        (4, 24, 24, False),  # 4 tiles, exactly enough
        (4, 96, 168, False),  # 112 tiles
        (2, 24, 16, False),  # 2 tiles across 2 ranks
        (1, 16, 16, False),  # never parallel, nothing to guard
    ],
)
def test_guard_fires_exactly_when_tiles_are_short(parallel_size, h, w, falls_back):
    """Fewer tiles than ranks is the hang condition; equal or more is fine."""
    vae = _vae(parallel_size, {"sp_size": parallel_size, "sp_rank": 0})
    num_tiles = MiniMaxH3VideoVAE._decoder_tile_count(vae, _latent(h, w))
    assert (vae.parallel_size > 1 and num_tiles < vae.parallel_size) is falls_back
