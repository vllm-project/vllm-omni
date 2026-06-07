# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import (
    DistributedAutoencoderKLHunyuan,
)
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan_video_15 import (
    DistributedAutoencoderKLHunyuanVideo15,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyDistributedAutoencoderKLHunyuan(DistributedAutoencoderKLHunyuan):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.tile_latent_min_size = 8
        self.tile_sample_min_size = 8
        self.tile_overlap_factor = 0.25
        self.use_spatial_tiling = False

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        return x + 10

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return x + 20

    def blend_v(self, _a: torch.Tensor, b: torch.Tensor, _blend_extent: int) -> torch.Tensor:
        return b

    def blend_h(self, _a: torch.Tensor, b: torch.Tensor, _blend_extent: int) -> torch.Tensor:
        return b


class _DummyDistributedAutoencoderKLHunyuanVideo15(DistributedAutoencoderKLHunyuanVideo15):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.tile_sample_min_height = 8
        self.tile_sample_min_width = 12
        self.tile_latent_min_height = 8
        self.tile_latent_min_width = 12
        self.tile_overlap_factor = 0.25
        self.use_tiling = False

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        return x + 10

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return x + 20

    def blend_v(self, _a: torch.Tensor, b: torch.Tensor, _blend_extent: int) -> torch.Tensor:
        return b

    def blend_h(self, _a: torch.Tensor, b: torch.Tensor, _blend_extent: int) -> torch.Tensor:
        return b


def test_hunyuan_vae_use_tiling_aliases_spatial_tiling():
    # Verify use_tiling property maps to use_spatial_tiling.
    vae = _DummyDistributedAutoencoderKLHunyuan()

    assert not vae.use_tiling

    vae.use_tiling = True

    assert vae.use_spatial_tiling


def test_hunyuan_vae_decode_tiles_round_trip():
    # Validate decode tile split/exec/merge returns expected reconstructed tensor.
    vae = _DummyDistributedAutoencoderKLHunyuan()
    z = torch.arange(144, dtype=torch.float32).reshape(1, 1, 1, 12, 12)

    tile_tasks, grid_spec = vae.tile_split(z)
    decoded_tiles = {task.grid_coord: vae.tile_exec(task) for task in tile_tasks}
    output = vae.tile_merge(decoded_tiles, grid_spec)

    assert grid_spec.split_dims == (3, 4)
    assert grid_spec.grid_shape == (2, 2)
    assert grid_spec.tile_spec == {"blend_extent": 2, "row_limit": 6}
    assert len(tile_tasks) == 4
    assert torch.equal(output, z + 10)


def test_hunyuan_vae_encode_tiles_round_trip():
    # Validate encode tile split/exec/merge returns expected latent tensor.
    vae = _DummyDistributedAutoencoderKLHunyuan()
    x = torch.arange(144, dtype=torch.float32).reshape(1, 1, 1, 12, 12)

    tile_tasks, grid_spec = vae.encode_tile_split(x)
    encoded_tiles = {task.grid_coord: vae.encode_tile_exec(task) for task in tile_tasks}
    output = vae.encode_tile_merge(encoded_tiles, grid_spec)

    assert grid_spec.split_dims == (3, 4)
    assert grid_spec.grid_shape == (2, 2)
    assert grid_spec.tile_spec == {"blend_extent": 2, "row_limit": 6}
    assert len(tile_tasks) == 4
    assert torch.equal(output, x + 20)


def test_hunyuan_video15_vae_decode_tiles_round_trip():
    vae = _DummyDistributedAutoencoderKLHunyuanVideo15()
    z = torch.arange(192, dtype=torch.float32).reshape(1, 1, 1, 12, 16)

    tile_tasks, grid_spec = vae.tile_split(z)
    decoded_tiles = {task.grid_coord: vae.tile_exec(task) for task in tile_tasks}
    output = vae.tile_merge(decoded_tiles, grid_spec)

    assert grid_spec.split_dims == (3, 4)
    assert grid_spec.grid_shape == (2, 2)
    assert grid_spec.tile_spec == {
        "blend_height": 2,
        "blend_width": 3,
        "row_limit_height": 6,
        "row_limit_width": 9,
    }
    assert len(tile_tasks) == 4
    assert torch.equal(output, z + 10)


def test_hunyuan_video15_vae_encode_tiles_round_trip():
    vae = _DummyDistributedAutoencoderKLHunyuanVideo15()
    x = torch.arange(192, dtype=torch.float32).reshape(1, 1, 1, 12, 16)

    tile_tasks, grid_spec = vae.encode_tile_split(x)
    encoded_tiles = {task.grid_coord: vae.encode_tile_exec(task) for task in tile_tasks}
    output = vae.tile_merge(encoded_tiles, grid_spec)

    assert grid_spec.split_dims == (3, 4)
    assert grid_spec.grid_shape == (2, 2)
    assert grid_spec.tile_spec == {
        "blend_height": 2,
        "blend_width": 3,
        "row_limit_height": 6,
        "row_limit_width": 9,
    }
    assert len(tile_tasks) == 4
    assert torch.equal(output, x + 20)
