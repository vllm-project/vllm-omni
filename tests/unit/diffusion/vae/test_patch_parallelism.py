# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.vae import patch_parallelism as pp


class _DummyConfig:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class _DummyVae:
    def __init__(self, *, config=None, **attrs):
        self.config = config
        for k, v in attrs.items():
            setattr(self, k, v)


def test_get_vae_spatial_scale_factor_uses_block_out_channels_len_minus_1():
    vae = _DummyVae(config=_DummyConfig(block_out_channels=[128, 256, 512, 512]))
    assert pp._get_vae_spatial_scale_factor(vae) == 8

    vae = _DummyVae(config=_DummyConfig(block_out_channels=[1, 2, 3, 4, 5]))
    assert pp._get_vae_spatial_scale_factor(vae) == 16


def test_get_vae_spatial_scale_factor_defaults_to_8_on_missing_or_empty():
    assert pp._get_vae_spatial_scale_factor(_DummyVae(config=_DummyConfig())) == 8
    assert pp._get_vae_spatial_scale_factor(_DummyVae(config=_DummyConfig(block_out_channels=[]))) == 8
    assert pp._get_vae_spatial_scale_factor(_DummyVae(config=None)) == 8


def test_get_vae_spatial_scale_factor_defaults_to_8_on_exception():
    class _BrokenConfig:
        @property
        def block_out_channels(self):
            raise RuntimeError("boom")

    assert pp._get_vae_spatial_scale_factor(_DummyVae(config=_BrokenConfig())) == 8


@pytest.mark.parametrize(
    ("pp_size", "expected"),
    [
        (0, (1, 1)),
        (1, (1, 1)),
        (2, (1, 2)),
        (3, (1, 3)),
        (4, (2, 2)),
        (6, (2, 3)),
        (8, (2, 4)),
        (12, (3, 4)),
        (16, (4, 4)),
    ],
)
def test_factor_pp_grid(pp_size: int, expected: tuple[int, int]):
    assert pp._factor_pp_grid(pp_size) == expected


def test_get_world_rank_pp_size(monkeypatch):
    monkeypatch.setattr(pp.dist, "get_world_size", lambda _: 8)
    monkeypatch.setattr(pp.dist, "get_rank", lambda _: 3)

    world_size, rank, pp_size = pp._get_world_rank_pp_size(object(), 4)
    assert (world_size, rank, pp_size) == (8, 3, 4)

    world_size, rank, pp_size = pp._get_world_rank_pp_size(object(), 16)
    assert (world_size, rank, pp_size) == (8, 3, 8)


def test_get_vae_out_channels_defaults_to_3():
    assert pp._get_vae_out_channels(_DummyVae(config=None)) == 3
    assert pp._get_vae_out_channels(_DummyVae(config=_DummyConfig())) == 3


def test_get_vae_out_channels_reads_config():
    assert pp._get_vae_out_channels(_DummyVae(config=_DummyConfig(out_channels=4))) == 4
    assert pp._get_vae_out_channels(_DummyVae(config=_DummyConfig(out_channels="5"))) == 5


def test_get_vae_tile_params_returns_none_if_missing():
    assert pp._get_vae_tile_params(_DummyVae(tile_latent_min_size=None, tile_overlap_factor=0.25)) is None
    assert pp._get_vae_tile_params(_DummyVae(tile_latent_min_size=128, tile_overlap_factor=None)) is None


def test_get_vae_tile_params_parses_types():
    vae = _DummyVae(tile_latent_min_size="128", tile_overlap_factor="0.25")
    assert pp._get_vae_tile_params(vae) == (128, 0.25)


def test_get_vae_tiling_params_returns_none_if_missing():
    vae = _DummyVae(tile_latent_min_size=128, tile_overlap_factor=0.25, tile_sample_min_size=None)
    assert pp._get_vae_tiling_params(vae) is None

    vae = _DummyVae(tile_latent_min_size=None, tile_overlap_factor=0.25, tile_sample_min_size=1024)
    assert pp._get_vae_tiling_params(vae) is None


def test_get_vae_tiling_params_parses_types():
    vae = _DummyVae(tile_latent_min_size="128", tile_overlap_factor="0.25", tile_sample_min_size="1024")
    assert pp._get_vae_tiling_params(vae) == (128, 0.25, 1024)

