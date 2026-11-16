# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the tile-shortage guards in the MiniMax-H3 video VAE."""

import sys
import types

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

RATIO = 16
# The checkpoint keeps two independent grids -- ``split_tiles`` selects
# ``decoder_tile_size``/``decoder_tile_overlap_min`` when ``is_decoder`` is set
# and ``tile_size``/``tile_overlap_min`` otherwise. The shipped H3 config leaves
# both at 256/64, so the two happen to coincide there, but that is a config
# coincidence. The double deliberately gives them different values so that
# passing the wrong flag changes the tile count and fails a test.
TILE_SIZE, OVERLAP = 256, 64  # decoder, matching the shipped config
ENCODER_TILE_SIZE, ENCODER_OVERLAP = 128, 32


class _FakeCheckpointModel:
    """The parts of the checkpoint's AutoencoderKL the guard touches."""

    def __init__(self):
        self.vae_ratio = RATIO
        self.parallel_tiling = True
        self.decoded_with = []

    def split_tiles(self, input_len, is_decoder=False):
        """Same grid arithmetic as the checkpoint, on pixel dimensions."""
        tile = TILE_SIZE if is_decoder else ENCODER_TILE_SIZE
        overlap = OVERLAP if is_decoder else ENCODER_OVERLAP
        if tile >= input_len:
            return [0], [input_len], []
        n = -(-input_len // tile)
        while tile * n - overlap * (n - 1) - input_len < 0:
            n += 1
        return list(range(n)), [tile] * n, [overlap] * (n - 1)


def _vae(parallel_size, state):
    """A stand-in instance: the guard only reads .model/.remote/.parallel_size."""
    module = types.ModuleType("fake_ckpt.parallel")
    module.__dict__["get_parallel_state"] = lambda: state
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
        (48, 84, 28),  # 768x1344 px, the shipped H3 request
        (96, 168, 112),  # 1536x2688 px, a larger canvas the request path also allows
    ],
)
def test_tile_count_matches_the_checkpoint_grid(h, w, expected):
    assert MiniMaxH3VideoVAE._decoder_tile_count(_vae(1, {}), _latent(h, w)) == expected


@pytest.mark.parametrize(("h", "w", "decoder_tiles", "encoder_tiles"), [(16, 16, 1, 9), (24, 24, 4, 16)])
def test_tile_count_uses_the_decoder_grid_not_the_encoder_one(h, w, decoder_tiles, encoder_tiles):
    """Passing ``is_decoder=False`` would silently change the count, so pin it."""
    vae = _vae(1, {})
    assert MiniMaxH3VideoVAE._decoder_tile_count(vae, _latent(h, w)) == decoder_tiles

    px_h, px_w = h * RATIO, w * RATIO
    as_encoder = len(vae.model.split_tiles(px_h, False)[0]) * len(vae.model.split_tiles(px_w, False)[0])
    assert as_encoder == encoder_tiles != decoder_tiles


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


# ---------------------------------------------------------------------------
# Dispatch: every one of these calls ``decode_latent``, so deleting the guard
# turns the whole matrix red rather than leaving it green.
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    @staticmethod
    def revert_tensor(decoded):
        return decoded


def _dispatch_vae(parallel_size):
    """A VAE whose ``decode_base`` records the parallel state it ran under."""
    state = {
        "group_size": parallel_size,
        "group_rank": 0,
        "local_process_group": object(),
        "sp_size": parallel_size,
        "sp_rank": 0,
        "sp_enabled": True,
        "sp_process_group": object(),
        "tp_size": 1,
        "tp_rank": 0,
    }
    vae = _vae(parallel_size, state)
    vae.config_dict = {
        "latent_channels": 24,
        "latents_mean": [0.0] * 24,
        "latents_std": [1.0] * 24,
    }
    vae.model.processor = _RecordingProcessor()

    seen = {}

    def decode_base(sample):
        seen["sp_size"] = state["sp_size"]
        seen["sp_rank"] = state["sp_rank"]
        seen["sp_enabled"] = state["sp_enabled"]
        seen["sp_process_group"] = state["sp_process_group"]
        seen["parallel_tiling"] = vae.model.parallel_tiling
        # the guard never inspects the decoded shape, so keep it small
        return torch.zeros(1, 3, 2, 4, 4)

    vae.model.decode_base = decode_base
    return vae, state, seen


@pytest.mark.parametrize(
    ("parallel_size", "h", "w", "rank_local", "sp_size"),
    [
        (4, 24, 16, False, 2),  # 2 tiles: one per rank, ranks 2 and 3 repeat tiles 0 and 1
        (8, 24, 40, False, 6),  # 6 tiles across 8 ranks
        (4, 16, 16, True, 1),  # 1 tile: nothing to share out
        (4, 24, 24, False, 4),  # 4 tiles, exactly enough
        (4, 48, 84, False, 4),  # 28 tiles, the shipped 768x1344 request
        (4, 96, 168, False, 4),  # 112 tiles, 1536x2688
        (2, 24, 16, False, 2),  # 2 tiles across 2 ranks
        (1, 16, 16, False, 1),  # never parallel, nothing to guard
    ],
)
def test_decode_latent_never_leaves_a_rank_without_a_tile(parallel_size, h, w, rank_local, sp_size):
    """Fewer tiles than ranks is the hang condition; equal or more is fine."""
    vae, _, seen = _dispatch_vae(parallel_size)

    MiniMaxH3VideoVAE.decode_latent(vae, _latent(h, w))

    # only the rank-local guard clears ``parallel_tiling``, so it is the signal that it fired
    assert (seen["parallel_tiling"] is False) is rank_local
    assert seen["sp_size"] == sp_size


def _checkpoint_tile_shares(state_by_rank, num_tiles):
    """The checkpoint's ownership and gather reconstruction, for every rank of the group.

    Mirrors ``_local_tile_indices`` and ``_all_gather_tiled_results`` in the
    checkpoint's ``klvae.py``: a rank decodes ``range(sp_rank, num_tiles,
    sp_size)``, and the grid is rebuilt from the shares of every group rank.
    """
    shares = [list(range(state["sp_rank"], num_tiles, state["sp_size"])) for state in state_by_rank]
    if any(not share for share in shares):
        raise ValueError("Found empty tasks")
    sp_size = state_by_rank[0]["sp_size"]
    results = [None] * num_tiles
    for rank, share in enumerate(shares):
        for k, tile in enumerate(share):
            global_idx = k * sp_size + rank
            if global_idx >= num_tiles:
                break
            results[global_idx] = tile
    return shares, results


@pytest.mark.parametrize(("parallel_size", "num_tiles"), [(8, 6), (8, 2), (4, 3), (8, 7)])
def test_wrapped_ownership_gives_each_rank_one_tile_and_rebuilds_the_whole_grid(parallel_size, num_tiles):
    states = []
    for rank in range(parallel_size):
        state = {"sp_size": parallel_size, "sp_rank": rank, "sp_enabled": True, "sp_process_group": "pg"}
        vae = _vae(parallel_size, state)
        with vae._wrapped_tile_ownership(num_tiles):
            states.append(dict(state))
        # the group state is restored once the decode leaves the context
        assert state == {"sp_size": parallel_size, "sp_rank": rank, "sp_enabled": True, "sp_process_group": "pg"}

    shares, results = _checkpoint_tile_shares(states, num_tiles)

    assert all(len(share) == 1 for share in shares), "every rank decodes exactly one tile"
    assert results == list(range(num_tiles)), "the gather rebuilds every tile in order"
    assert all(state["sp_process_group"] == "pg" and state["sp_enabled"] for state in states)

    # Without it the surplus ranks hold an empty share, which is the hang the guard exists for.
    unwrapped = [{"sp_size": parallel_size, "sp_rank": rank} for rank in range(parallel_size)]
    with pytest.raises(ValueError, match="Found empty tasks"):
        _checkpoint_tile_shares(unwrapped, num_tiles)


def test_wrapped_ownership_restores_the_group_state_after_an_exception():
    state = {"sp_size": 8, "sp_rank": 7, "sp_enabled": True, "sp_process_group": "pg"}
    vae = _vae(8, state)

    with pytest.raises(RuntimeError, match="decode failed"):
        with vae._wrapped_tile_ownership(6):
            assert (state["sp_size"], state["sp_rank"]) == (6, 1)
            raise RuntimeError("decode failed")

    assert (state["sp_size"], state["sp_rank"]) == (8, 7)


def test_short_grid_keeps_the_rank_local_fallback_under_the_equal_shape_gather():
    """The replacement gather sizes its buffer for one share per group rank."""
    vae, _, seen = _dispatch_vae(4)
    vae._checkpoint_tile_gather = lambda tasks, num_tiles: tasks
    vae._decoder_tile_grid_hw = lambda latent: None

    MiniMaxH3VideoVAE.decode_latent(vae, _latent(24, 16))

    assert seen["parallel_tiling"] is False and seen["sp_size"] == 1


def test_decode_latent_isolates_and_restores_the_whole_group_state():
    """1 tile, 4 ranks: ``decode_base`` must observe rank-local tiling."""
    vae, state, seen = _dispatch_vae(4)

    frames = MiniMaxH3VideoVAE.decode_latent(vae, _latent(16, 16))

    assert seen["sp_size"] == 1
    assert seen["sp_enabled"] is False
    assert seen["sp_process_group"] is None
    assert seen["parallel_tiling"] is False
    assert frames.ndim == 5 and frames.dtype is torch.float32
    # the group state is restored once decode returns
    assert state["sp_size"] == 4 and state["sp_enabled"] is True
    assert vae.model.parallel_tiling is True


def test_decode_latent_leaves_the_group_state_alone_when_the_grid_is_large_enough():
    """The shipped 768x1344 request is 28 tiles: the guard must not fire."""
    vae, state, seen = _dispatch_vae(4)

    MiniMaxH3VideoVAE.decode_latent(vae, _latent(48, 84))

    assert seen["sp_size"] == 4
    assert seen["sp_enabled"] is True
    assert seen["sp_process_group"] is state["sp_process_group"]
    assert seen["parallel_tiling"] is True


def _control_vae(parallel_size):
    vae, state, seen = _dispatch_vae(parallel_size)
    vae.parameters = lambda: iter([torch.zeros(1)])
    # Control requests use the checkpoint's shipped 256/64 encoder grid.
    vae.model.split_tiles = lambda length, is_decoder=False: _FakeCheckpointModel.split_tiles(vae.model, length, True)

    def encode_temporal(pixels):
        seen.update(state)
        seen["parallel_tiling"] = vae.model.parallel_tiling
        return torch.zeros(1, 48, 1, 2, 2)

    vae.model.encode_temporal = encode_temporal
    return vae, state, seen


@pytest.mark.parametrize(
    ("parallel_size", "height", "width", "expected_size", "expected_tiling"),
    [
        (4, 384, 256, 1, False),  # two tiles: two ranks would get no work
        (4, 256, 256, 1, False),  # one tile: three ranks would get no work
        (4, 384, 384, 4, True),  # four tiles: keep normal parallel encoding
        (1, 256, 256, 1, False),  # clear a stale parallel_tiling flag
    ],
)
def test_control_encode_uses_safe_tiling_and_restores_state(
    parallel_size, height, width, expected_size, expected_tiling
):
    vae, state, seen = _control_vae(parallel_size)
    saved_state = dict(state)

    latent = vae.encode_control_latents(torch.zeros(1, 3, 1, height, width))

    assert seen["sp_size"] == expected_size
    assert seen["parallel_tiling"] is expected_tiling
    if parallel_size > 1 and not expected_tiling:
        assert seen["sp_enabled"] is False
        assert seen["sp_process_group"] is None
    assert latent.shape == (1, 24, 1, 2, 2)
    assert state == saved_state
    assert vae.model.parallel_tiling is True


@pytest.mark.parametrize("parallel_size", [1, 4])
def test_control_encode_restores_state_after_error(parallel_size):
    vae, state, seen = _control_vae(parallel_size)
    saved_state = dict(state)
    record_encode = vae.model.encode_temporal

    def fail_encode(pixels):
        record_encode(pixels)
        raise RuntimeError("encode failed")

    vae.model.encode_temporal = fail_encode
    with pytest.raises(RuntimeError, match="encode failed"):
        vae.encode_control_latents(torch.zeros(1, 3, 1, 256, 256))

    assert seen["sp_size"] == 1
    assert seen["parallel_tiling"] is False
    assert state == saved_state
    assert vae.model.parallel_tiling is True
