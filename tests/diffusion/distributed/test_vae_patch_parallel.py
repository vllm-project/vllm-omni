# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for VAE patch/tile parallelism helpers (CPU-only)."""

import pytest
import torch

from vllm_omni.diffusion.distributed import vae_patch_parallel as vae_patch_parallel


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
    assert vae_patch_parallel._get_vae_spatial_scale_factor(vae) == 8

    vae = _DummyVae(config=_DummyConfig(block_out_channels=[1, 2, 3, 4, 5]))
    assert vae_patch_parallel._get_vae_spatial_scale_factor(vae) == 16


def test_get_vae_spatial_scale_factor_defaults_to_8_on_missing_or_empty():
    assert vae_patch_parallel._get_vae_spatial_scale_factor(_DummyVae(config=_DummyConfig())) == 8
    assert vae_patch_parallel._get_vae_spatial_scale_factor(_DummyVae(config=_DummyConfig(block_out_channels=[]))) == 8
    assert vae_patch_parallel._get_vae_spatial_scale_factor(_DummyVae(config=None)) == 8


def test_get_vae_spatial_scale_factor_defaults_to_8_on_exception():
    class _BrokenConfig:
        @property
        def block_out_channels(self):
            raise RuntimeError("boom")

    assert vae_patch_parallel._get_vae_spatial_scale_factor(_DummyVae(config=_BrokenConfig())) == 8


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
    assert vae_patch_parallel._factor_pp_grid(pp_size) == expected


def test_get_world_rank_pp_size(monkeypatch):
    monkeypatch.setattr(vae_patch_parallel.dist, "get_world_size", lambda _: 8)
    monkeypatch.setattr(vae_patch_parallel.dist, "get_rank", lambda _: 3)

    world_size, rank, pp_size = vae_patch_parallel._get_world_rank_pp_size(object(), 4)
    assert (world_size, rank, pp_size) == (8, 3, 4)

    world_size, rank, pp_size = vae_patch_parallel._get_world_rank_pp_size(object(), 16)
    assert (world_size, rank, pp_size) == (8, 3, 8)


def test_get_vae_out_channels_defaults_to_3():
    assert vae_patch_parallel._get_vae_out_channels(_DummyVae(config=None)) == 3
    assert vae_patch_parallel._get_vae_out_channels(_DummyVae(config=_DummyConfig())) == 3


def test_get_vae_out_channels_reads_config():
    assert vae_patch_parallel._get_vae_out_channels(_DummyVae(config=_DummyConfig(out_channels=4))) == 4
    assert vae_patch_parallel._get_vae_out_channels(_DummyVae(config=_DummyConfig(out_channels="5"))) == 5


def test_get_vae_tile_params_returns_none_if_missing():
    assert (
        vae_patch_parallel._get_vae_tile_params(_DummyVae(tile_latent_min_size=None, tile_overlap_factor=0.25)) is None
    )
    assert (
        vae_patch_parallel._get_vae_tile_params(_DummyVae(tile_latent_min_size=128, tile_overlap_factor=None)) is None
    )


def test_get_vae_tile_params_parses_types():
    vae = _DummyVae(tile_latent_min_size="128", tile_overlap_factor="0.25")
    assert vae_patch_parallel._get_vae_tile_params(vae) == (128, 0.25)


def test_get_vae_tiling_params_returns_none_if_missing():
    vae = _DummyVae(tile_latent_min_size=128, tile_overlap_factor=0.25, tile_sample_min_size=None)
    assert vae_patch_parallel._get_vae_tiling_params(vae) is None

    vae = _DummyVae(tile_latent_min_size=None, tile_overlap_factor=0.25, tile_sample_min_size=1024)
    assert vae_patch_parallel._get_vae_tiling_params(vae) is None


def test_get_vae_tiling_params_parses_types():
    vae = _DummyVae(tile_latent_min_size="128", tile_overlap_factor="0.25", tile_sample_min_size="1024")
    assert vae_patch_parallel._get_vae_tiling_params(vae) == (128, 0.25, 1024)


def test_distributed_tiled_decode_stitches_tiles(monkeypatch):
    class _TinyConfig:
        def __init__(self):
            self.out_channels = 1
            self.use_post_quant_conv = False

    class _TinyVae:
        def __init__(self):
            self.config = _TinyConfig()
            self.tile_latent_min_size = 2
            self.tile_overlap_factor = 0.0
            self.tile_sample_min_size = 2

        def decoder(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def blend_v(self, _a: torch.Tensor, b: torch.Tensor, _blend_extent: int) -> torch.Tensor:
            return b

        def blend_h(self, _a: torch.Tensor, b: torch.Tensor, _blend_extent: int) -> torch.Tensor:
            return b

    def _collect_local_tiles(
        *,
        vae: _TinyVae,
        z: torch.Tensor,
        rank: int,
        pp_size: int,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int, int]]]:
        tile_latent_min_size = vae.tile_latent_min_size
        overlap_size = int(tile_latent_min_size * (1 - vae.tile_overlap_factor))
        h_starts = list(range(0, z.shape[2], overlap_size))
        w_starts = list(range(0, z.shape[3], overlap_size))

        local_tiles: list[torch.Tensor] = []
        local_meta: list[tuple[int, int, int]] = []
        tile_id = 0
        for i in h_starts:
            for j in w_starts:
                tile_rank = (tile_id + 1) % pp_size
                if tile_rank == rank:
                    tile = z[:, :, i : i + tile_latent_min_size, j : j + tile_latent_min_size]
                    decoded = vae.decoder(tile)
                    local_tiles.append(decoded)
                    local_meta.append((tile_id, int(decoded.shape[-2]), int(decoded.shape[-1])))
                tile_id += 1
        return local_tiles, local_meta

    vae = _TinyVae()
    z = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    rank0_tiles, rank0_meta = _collect_local_tiles(vae=vae, z=z, rank=0, pp_size=2)
    rank1_tiles, rank1_meta = _collect_local_tiles(vae=vae, z=z, rank=1, pp_size=2)
    max_count = max(len(rank0_tiles), len(rank1_tiles))

    def _pack_meta_and_tiles(
        tiles: list[torch.Tensor],
        meta: list[tuple[int, int, int]],
        max_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        meta_tensor = torch.full((max_count, 3), -1, dtype=torch.int64)
        tile_tensor = torch.zeros(
            (max_count, z.shape[0], vae.config.out_channels, vae.tile_sample_min_size, vae.tile_sample_min_size),
            dtype=z.dtype,
        )
        for idx, (tile_id, h, w) in enumerate(meta):
            meta_tensor[idx, 0] = tile_id
            meta_tensor[idx, 1] = h
            meta_tensor[idx, 2] = w
            tile_tensor[idx, :, :, :h, :w] = tiles[idx]
        return meta_tensor, tile_tensor

    rank1_meta_tensor, rank1_tile_tensor = _pack_meta_and_tiles(rank1_tiles, rank1_meta, max_count)
    rank1_count_tensor = torch.tensor([len(rank1_tiles)], dtype=torch.int64)

    def _fake_gather(tensor, gather_list=None, dst=0, group=None):
        if gather_list is None:
            return
        if tensor.ndim == 1 and tensor.numel() == 1:
            gather_list[0].copy_(tensor)
            gather_list[1].copy_(rank1_count_tensor)
            return
        if tensor.ndim == 2 and tensor.shape[1] == 3:
            gather_list[0].copy_(tensor)
            gather_list[1].copy_(rank1_meta_tensor)
            return
        if tensor.ndim == 5:
            gather_list[0].copy_(tensor)
            gather_list[1].copy_(rank1_tile_tensor)
            return
        raise AssertionError("Unexpected gather payload for test.")

    def _fake_broadcast(_tensor, src=0, group=None):
        return

    monkeypatch.setattr(vae_patch_parallel.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(vae_patch_parallel.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(vae_patch_parallel.dist, "gather", _fake_gather)
    monkeypatch.setattr(vae_patch_parallel.dist, "broadcast", _fake_broadcast)

    output = vae_patch_parallel._distributed_tiled_decode(
        vae=vae,
        orig_decode=lambda z, return_dict=False: (z,),
        z=z,
        group=object(),
        vae_patch_parallel_size=2,
    )

    assert torch.equal(output, z)
