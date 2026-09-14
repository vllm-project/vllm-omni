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

    def __init__(
        self,
        *,
        encoder_tile_size=ENCODER_TILE_SIZE,
        encoder_overlap=ENCODER_OVERLAP,
    ):
        self.vae_ratio = RATIO
        self.parallel_tiling = True
        self.decoded_with = []
        self.processor = _RecordingProcessor()
        self.encoder_tile_size = encoder_tile_size
        self.encoder_overlap = encoder_overlap

    def split_tiles(self, input_len, is_decoder=False):
        """Same grid arithmetic as the checkpoint, on pixel dimensions."""
        tile = TILE_SIZE if is_decoder else self.encoder_tile_size
        overlap = OVERLAP if is_decoder else self.encoder_overlap
        if tile >= input_len:
            return [0], [input_len], []
        n = -(-input_len // tile)
        while tile * n - overlap * (n - 1) - input_len < 0:
            n += 1
        return list(range(n)), [tile] * n, [overlap] * (n - 1)


def _vae(
    parallel_size,
    state,
    *,
    encoder_tile_size=ENCODER_TILE_SIZE,
    encoder_overlap=ENCODER_OVERLAP,
):
    """A stand-in instance: the guard only reads .model/.remote/.parallel_size."""
    module = types.ModuleType("fake_ckpt.parallel")
    module.get_parallel_state = lambda: state
    sys.modules.setdefault("fake_ckpt", types.ModuleType("fake_ckpt"))
    sys.modules["fake_ckpt.parallel"] = module

    remote = type("Remote", (), {"__module__": "fake_ckpt.klvae"})()

    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = _FakeCheckpointModel(
        encoder_tile_size=encoder_tile_size,
        encoder_overlap=encoder_overlap,
    )
    vae.remote = remote
    vae.parallel_size = parallel_size
    return vae


def _latent(h, w):
    return torch.zeros(1, 24, 9, h, w)


def _frames(h, w, *, tensor=False):
    if tensor:
        return torch.zeros(3, 9, h, w)
    return type("NumpyLike", (), {"shape": (9, h, w, 3)})()


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


@pytest.mark.parametrize(
    ("h", "w", "expected"),
    [
        (128, 128, 1),
        (256, 128, 3),
        (256, 256, 9),
        (256, 448, 15),
    ],
)
@pytest.mark.parametrize("tensor", [False, True])
def test_encoder_tile_count_matches_checkpoint_grid(h, w, expected, tensor):
    vae = _vae(1, {})
    assert (
        MiniMaxH3VideoVAE._encoder_tile_count(
            vae,
            _frames(h, w, tensor=tensor),
        )
        == expected
    )


def test_encoder_tile_count_uses_aligned_crop_dimensions():
    vae = _vae(1, {})
    # The processor crops 159x255 to 128x224 before tiled_encode sees it.
    assert MiniMaxH3VideoVAE._encoder_tile_count(vae, _frames(159, 255)) == 2


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
        (4, 128, 128, True),
        (4, 256, 128, True),
        (4, 224, 224, False),
        (4, 256, 448, False),
        (2, 224, 128, False),
        (1, 128, 128, False),
    ],
)
def test_encode_tiling_context_falls_back_exactly_when_tiles_are_short(
    parallel_size,
    h,
    w,
    falls_back,
):
    state = {
        "sp_size": parallel_size,
        "sp_rank": 0,
        "sp_enabled": parallel_size > 1,
        "sp_process_group": object() if parallel_size > 1 else None,
    }
    vae = _vae(parallel_size, state)

    with vae._encode_tiling_context(_frames(h, w)):
        assert (vae.model.parallel_tiling is False) is falls_back
        assert state["sp_size"] == (1 if falls_back else parallel_size)

    assert state["sp_size"] == parallel_size
    assert vae.model.parallel_tiling is True


@pytest.mark.parametrize("rank", range(4))
def test_shipped_encoder_grid_falls_back_on_every_rank(rank):
    """The real 448x256 edit input has two shipped-config tiles over four ranks."""
    state = {
        "group_size": 4,
        "group_rank": rank,
        "local_process_group": "pg",
        "sp_size": 4,
        "sp_rank": rank,
        "sp_enabled": True,
        "sp_process_group": "pg",
        "tp_size": 1,
        "tp_rank": 0,
    }
    original_state = dict(state)
    vae = _vae(
        4,
        state,
        encoder_tile_size=256,
        encoder_overlap=64,
    )
    frames = type("NumpyLike", (), {"shape": (107, 256, 448, 3)})()

    assert vae._encoder_tile_count(frames) == 2
    with vae._encode_tiling_context(frames):
        assert state["sp_size"] == 1
        assert state["sp_rank"] == 0
        assert vae.model.parallel_tiling is False

    assert state == original_state
    assert vae.model.parallel_tiling is True


def test_encode_video_dispatches_with_rank_local_tiling_for_short_grid():
    state = {
        "sp_size": 4,
        "sp_rank": 0,
        "sp_enabled": True,
        "sp_process_group": object(),
    }
    vae = _vae(4, state)
    vae.config_dict = {
        "latent_channels": 24,
        "latents_mean": [0.0] * 24,
        "latents_std": [1.0] * 24,
    }
    vae.device_module = torch
    parameter = torch.nn.Parameter(torch.zeros(()))
    vae.parameters = lambda: iter([parameter])
    seen = {}

    def encode_videos(frames, *, use_fp16_latent):
        assert use_fp16_latent is True
        seen["sp_size"] = state["sp_size"]
        seen["sp_enabled"] = state["sp_enabled"]
        seen["parallel_tiling"] = vae.model.parallel_tiling
        return [torch.zeros(24, 2, 4, 4)]

    vae.model.encode_videos = encode_videos

    rows, shape = MiniMaxH3VideoVAE.encode_video(
        vae,
        torch.zeros(3, 9, 128, 128),
    )

    assert seen == {
        "sp_size": 1,
        "sp_enabled": False,
        "parallel_tiling": False,
    }
    assert rows.shape == (8, 96)
    assert shape == (2, 4, 4)
    assert state["sp_size"] == 4 and state["sp_enabled"] is True
    assert vae.model.parallel_tiling is True


# ---------------------------------------------------------------------------
# Dispatch: every one of these calls ``decode_latent``, so deleting the guard
# turns the whole matrix red rather than leaving it green.
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    @staticmethod
    def _align_to_total_patch_size(h, w):
        return (h // 32) * 32, (w // 32) * 32

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
        seen["sp_enabled"] = state["sp_enabled"]
        seen["sp_process_group"] = state["sp_process_group"]
        seen["parallel_tiling"] = vae.model.parallel_tiling
        # the guard never inspects the decoded shape, so keep it small
        return torch.zeros(1, 3, 2, 4, 4)

    vae.model.decode_base = decode_base
    return vae, state, seen


@pytest.mark.parametrize(
    ("parallel_size", "h", "w", "falls_back"),
    [
        (4, 24, 16, True),  # 2 tiles, ranks 2 and 3 would get none
        (4, 16, 16, True),  # 1 tile
        (4, 24, 24, False),  # 4 tiles, exactly enough
        (4, 48, 84, False),  # 28 tiles, the shipped 768x1344 request
        (4, 96, 168, False),  # 112 tiles, 1536x2688
        (2, 24, 16, False),  # 2 tiles across 2 ranks
        (1, 16, 16, False),  # never parallel, nothing to guard
    ],
)
def test_decode_latent_falls_back_exactly_when_tiles_are_short(parallel_size, h, w, falls_back):
    """Fewer tiles than ranks is the hang condition; equal or more is fine."""
    vae, _, seen = _dispatch_vae(parallel_size)

    MiniMaxH3VideoVAE.decode_latent(vae, _latent(h, w))

    # only the guard clears ``parallel_tiling``, so it is the signal that it fired
    assert (seen["parallel_tiling"] is False) is falls_back
    assert seen["sp_size"] == (1 if falls_back else parallel_size)


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
