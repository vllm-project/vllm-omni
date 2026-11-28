# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Configuration and mainline stacking contracts for native H3 video decode."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import vae_reshape

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def checkpoint():
    """Provide separate encoder and decoder geometries from a native checkpoint."""
    return SimpleNamespace(
        vae_ratio=16,
        tile_size=256,
        tile_overlap_min=64,
        decoder_tile_size=256,
        decoder_tile_overlap_min=64,
    )


@pytest.mark.parametrize(
    ("config", "decoder_geometry", "batch_size"),
    [
        (None, (256, 64), None),
        ({}, (256, 64), None),
        ({"tile_batch_size": 3}, (256, 64), 3),
        ({"tile_size": 288}, (288, 64), None),
        ({"tile_min_overlap": 16}, (256, 16), None),
        ({"tile_size": 32, "tile_min_overlap": 16, "tile_batch_size": 1}, (32, 16), 1),
    ],
)
def test_config_updates_only_decoder_geometry(checkpoint, config, decoder_geometry, batch_size):
    """Keep absent values and encoder settings while applying independent decoder overrides."""
    assert vae_reshape.configure_video_decode_tiles(checkpoint, config) == batch_size
    assert (checkpoint.decoder_tile_size, checkpoint.decoder_tile_overlap_min) == decoder_geometry
    assert (checkpoint.tile_size, checkpoint.tile_overlap_min) == (256, 64)


@pytest.mark.parametrize(
    ("config", "field"),
    [
        ([], "video_vae_decode"),
        ({"tile_overlap": 16}, "tile_overlap"),
        ({"tile_batch_size": True}, "tile_batch_size"),
        ({"tile_size": "288"}, "tile_size"),
        ({"tile_min_overlap": 0}, "tile_min_overlap"),
        ({"tile_min_overlap": 17, "tile_size": 288}, "tile_min_overlap"),
        ({"tile_size": 64}, "tile_min_overlap"),
    ],
)
def test_invalid_config_does_not_apply_partial_overrides(checkpoint, config, field):
    """Reject invalid JSON and invalid resolved pairs before changing the checkpoint."""
    with pytest.raises(ValueError, match=field):
        vae_reshape.configure_video_decode_tiles(checkpoint, config)
    assert (checkpoint.decoder_tile_size, checkpoint.decoder_tile_overlap_min) == (256, 64)
    assert (checkpoint.tile_size, checkpoint.tile_overlap_min) == (256, 64)


def test_json_batch_size_overrides_legacy_and_respects_native_cap(monkeypatch):
    """Keep packed tasks within the mainline runner's limit without changing the native flag."""
    model = SimpleNamespace(
        tokens_chunk_size=2,
        token_overlap=0,
        token_drop=0,
        vae_ratio=1,
        stack_tiling=False,
        split_tiles=lambda length, is_decoder: ([0], [length], []),
        tiled_decode=lambda latent: latent,
    )
    adapter = SimpleNamespace(model=model, stack_decode_tiles=True, max_stacked_decode_tiles=7)
    monkeypatch.setenv("H3_VIDEO_DECODER_CROSS_CHUNK", "1")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCHING", "0")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCH_SIZE", "2")
    vae_reshape.install_cross_chunk_tiling(adapter, tile_batch_size=9)
    latent = torch.zeros(1, 1, 4, 4, 4)
    pool = vae_reshape._TilePool(adapter, latent, all_outputs=False)
    assert pool.batch == 7
    assert pool.batching is True
    assert model.stack_tiling is False


def test_nonuniform_grid_uses_native_decoder(monkeypatch):
    """Do not wrap a decode whose effective geometry cannot use an equal-shape gather."""
    model = SimpleNamespace()
    latent = torch.ones(1, 1, 4, 4, 4)
    adapter = SimpleNamespace(
        model=model,
        decode_latent=lambda value: value + 1,
        _decoder_tile_grid_uniform=lambda value: False,
    )
    monkeypatch.setenv("H3_VIDEO_DECODER_CROSS_CHUNK", "1")
    vae_reshape.install_cross_chunk_tiling(adapter, tile_batch_size=3)
    torch.testing.assert_close(adapter.decode_latent(latent), latent + 1)
    assert model._h3_vae_reshape_last_stats["active"] is False
