# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DistributedAutoencoderKLWan encode parallel (CPU-only)."""

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _DummyConfig:
    def __init__(self, patch_size=None, scale_factor_temporal=4):
        self.patch_size = patch_size
        self.scale_factor_temporal = scale_factor_temporal


class _DummyWanVae:
    """Minimal mock of DistributedAutoencoderKLWan for testing encode_tile_split."""

    def __init__(
        self,
        config=None,
        spatial_compression_ratio=8,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
    ):
        self.config = config or _DummyConfig()
        self.spatial_compression_ratio = spatial_compression_ratio
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width
        self.dtype = torch.float32

        # Mock caches
        self._enc_feat_map = None
        self._enc_conv_idx = [0]

    def clear_cache(self):
        self._enc_feat_map = None
        self._enc_conv_idx = [0]

    def encoder(self, x, feat_cache=None, feat_idx=None):  # noqa: ARG002
        # Simple mock: just return the input
        return x

    def quant_conv(self, x):
        return x

    def blend_v(self, _a, b, _blend_extent):
        return b

    def blend_h(self, _a, b, _blend_extent):
        return b


def _import_encode_tile_split():
    """Import the encode_tile_split method from the module."""
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    return DistributedAutoencoderKLWan.encode_tile_split


def _import_encode_tile_exec():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    return DistributedAutoencoderKLWan.encode_tile_exec


def _import_encode_tile_merge():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    return DistributedAutoencoderKLWan.encode_tile_merge


class TestEncodeTileSplit:
    """Tests for encode_tile_split method."""

    def test_basic_split_without_patch_size(self):
        """Test basic tile splitting without patch_size."""
        encode_tile_split = _import_encode_tile_split()

        vae = _DummyWanVae(
            config=_DummyConfig(patch_size=None, scale_factor_temporal=4),
            spatial_compression_ratio=8,
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=192,
            tile_sample_stride_width=192,
        )

        # Input: (B, C, T, H, W) = (1, 3, 5, 256, 256)
        x = torch.randn(1, 3, 5, 256, 256)

        tiletask_list, grid_spec = encode_tile_split(vae, x)

        # With stride 192 and input size 256, we should get:
        # Height: ceil(256/192) = 2 positions (0, 192) but 192+256 > 256, so only 1
        # Actually for i in range(0, 256, 192): i = 0, 192 but 192 is out of bounds
        # So we get 1x1 grid
        assert len(tiletask_list) >= 1
        assert grid_spec.grid_shape[0] >= 1
        assert grid_spec.grid_shape[1] >= 1

        # Check temporal chunking: 5 frames -> 1 + (5-1)//4 = 2 chunks
        first_task = tiletask_list[0]
        assert len(first_task.tensor) == 2  # 2 temporal chunks

    def test_split_with_patch_size_scales_coordinates(self):
        """Test that patch_size properly scales tile coordinates."""
        encode_tile_split = _import_encode_tile_split()

        # Without patch_size
        vae_no_patch = _DummyWanVae(
            config=_DummyConfig(patch_size=None, scale_factor_temporal=4),
            spatial_compression_ratio=8,
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=128,
            tile_sample_stride_width=128,
        )

        # With patch_size=2 (simulating patchified input)
        vae_with_patch = _DummyWanVae(
            config=_DummyConfig(patch_size=2, scale_factor_temporal=4),
            spatial_compression_ratio=8,
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=128,
            tile_sample_stride_width=128,
        )

        # Same patchified input size
        x = torch.randn(1, 3, 5, 256, 256)

        tasks_no_patch, _ = encode_tile_split(vae_no_patch, x)
        tasks_with_patch, _ = encode_tile_split(vae_with_patch, x)

        # With patch_size=2, stride becomes 128//2=64, so more tiles
        assert len(tasks_with_patch) >= len(tasks_no_patch)

    def test_temporal_compression_from_config(self):
        """Test that temporal compression ratio is read from config."""
        encode_tile_split = _import_encode_tile_split()

        # temporal_compression=4 (default)
        vae_4x = _DummyWanVae(
            config=_DummyConfig(scale_factor_temporal=4),
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=512,
            tile_sample_stride_width=512,
        )

        # temporal_compression=2
        vae_2x = _DummyWanVae(
            config=_DummyConfig(scale_factor_temporal=2),
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=512,
            tile_sample_stride_width=512,
        )

        # 9 frames input
        x = torch.randn(1, 3, 9, 512, 512)

        tasks_4x, _ = encode_tile_split(vae_4x, x)
        tasks_2x, _ = encode_tile_split(vae_2x, x)

        # With 4x compression: 1 + (9-1)//4 = 3 chunks
        assert len(tasks_4x[0].tensor) == 3

        # With 2x compression: 1 + (9-1)//2 = 5 chunks
        assert len(tasks_2x[0].tensor) == 5

    def test_grid_spec_latent_dimensions(self):
        """Test that grid_spec contains correct latent dimensions."""
        encode_tile_split = _import_encode_tile_split()

        vae = _DummyWanVae(
            config=_DummyConfig(patch_size=None),
            spatial_compression_ratio=8,
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=512,
            tile_sample_stride_width=512,
        )

        # Input: 512x512 with compression 8 -> 64x64 latent
        x = torch.randn(1, 3, 5, 512, 512)

        _, grid_spec = encode_tile_split(vae, x)

        assert grid_spec.tile_spec["latent_height"] == 64
        assert grid_spec.tile_spec["latent_width"] == 64


class TestEncodeTileExec:
    """Tests for encode_tile_exec method."""

    def test_basic_exec(self):
        """Test basic tile execution."""
        encode_tile_exec = _import_encode_tile_exec()

        vae = _DummyWanVae()

        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
            TileTask,
        )

        # Create a simple task with 2 temporal chunks
        tile1 = torch.randn(1, 3, 1, 32, 32)
        tile2 = torch.randn(1, 3, 4, 32, 32)
        task = TileTask(tile_id=0, grid_coord=(0, 0), tensor=[tile1, tile2])

        result = encode_tile_exec(vae, task)

        # Result should concatenate temporal dimension
        assert result.shape[2] == 5  # 1 + 4 frames


class TestEncodeTileMerge:
    """Tests for encode_tile_merge method."""

    def test_basic_merge(self):
        """Test basic tile merging."""
        encode_tile_merge = _import_encode_tile_merge()

        vae = _DummyWanVae()

        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
            GridSpec,
        )

        # Create 2x2 grid of tiles
        tile_00 = torch.ones(1, 16, 2, 32, 32) * 0
        tile_01 = torch.ones(1, 16, 2, 32, 32) * 1
        tile_10 = torch.ones(1, 16, 2, 32, 32) * 2
        tile_11 = torch.ones(1, 16, 2, 32, 32) * 3

        coord_tensor_map = {
            (0, 0): tile_00,
            (0, 1): tile_01,
            (1, 0): tile_10,
            (1, 1): tile_11,
        }

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "latent_height": 48,
                "latent_width": 48,
                "blend_height": 8,
                "blend_width": 8,
                "tile_latent_stride_height": 24,
                "tile_latent_stride_width": 24,
            },
        )

        result = encode_tile_merge(vae, coord_tensor_map, grid_spec)

        # Output should be (1, 16, 2, 48, 48)
        assert result.shape == (1, 16, 2, 48, 48)
