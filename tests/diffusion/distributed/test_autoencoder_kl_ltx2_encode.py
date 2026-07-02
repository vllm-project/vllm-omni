# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DistributedAutoencoderKLLTX2Video encode parallel (CPU-only)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _DummyConfig:
    def __init__(self, latent_channels=2):
        self.latent_channels = latent_channels


class _DummyLTX2Vae:
    """Minimal mock of DistributedAutoencoderKLLTX2Video for testing the encode tile ops.

    The mock encoder is deterministic and content-dependent so that merge
    equivalence against the sequential (diffusers-style) tiled encode is
    meaningful, and its output shape follows the real encoder contract:
    (B, 2 * latent_channels, (F - 1) // tcr + 1, H // scr, W // scr).
    """

    def __init__(
        self,
        config=None,
        spatial_compression_ratio=4,
        temporal_compression_ratio=2,
        tile_sample_min_height=16,
        tile_sample_min_width=16,
        tile_sample_stride_height=12,
        tile_sample_stride_width=12,
    ):
        self.config = config or _DummyConfig()
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width
        self.dtype = torch.float32
        self.encoder_calls = []

    def encoder(self, x, causal=None):
        self.encoder_calls.append({"shape": tuple(x.shape), "causal": causal})
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_frames = (num_frames - 1) // self.temporal_compression_ratio + 1
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio
        pooled = F.adaptive_avg_pool3d(x, (latent_frames, latent_height, latent_width))

        out_channels = 2 * self.config.latent_channels
        repeats = (out_channels + num_channels - 1) // num_channels
        out = pooled.repeat(1, repeats, 1, 1, 1)[:, :out_channels]
        # Scale each channel differently so channel mixups would be caught.
        scale = torch.arange(1, out_channels + 1, dtype=out.dtype).view(1, -1, 1, 1, 1)
        return out * scale

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b


def _import_encode_tile_split():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )

    return DistributedAutoencoderKLLTX2Video.encode_tile_split


def _import_encode_tile_exec():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )

    return DistributedAutoencoderKLLTX2Video.encode_tile_exec


def _import_encode_tile_merge():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )

    return DistributedAutoencoderKLLTX2Video.encode_tile_merge


def _reference_tiled_encode(vae, x):
    """Sequential tiled encode mirroring diffusers AutoencoderKLLTX2Video.tiled_encode."""
    _, _, _, height, width = x.shape
    latent_height = height // vae.spatial_compression_ratio
    latent_width = width // vae.spatial_compression_ratio

    tile_latent_min_height = vae.tile_sample_min_height // vae.spatial_compression_ratio
    tile_latent_min_width = vae.tile_sample_min_width // vae.spatial_compression_ratio
    tile_latent_stride_height = vae.tile_sample_stride_height // vae.spatial_compression_ratio
    tile_latent_stride_width = vae.tile_sample_stride_width // vae.spatial_compression_ratio

    blend_height = tile_latent_min_height - tile_latent_stride_height
    blend_width = tile_latent_min_width - tile_latent_stride_width

    rows = []
    for i in range(0, height, vae.tile_sample_stride_height):
        row = []
        for j in range(0, width, vae.tile_sample_stride_width):
            time = vae.encoder(
                x[:, :, :, i : i + vae.tile_sample_min_height, j : j + vae.tile_sample_min_width],
                causal=None,
            )
            row.append(time)
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
        result_rows.append(torch.cat(result_row, dim=4))

    return torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]


class TestEncodeTileSplit:
    """Tests for encode_tile_split."""

    def test_grid_shape_and_tile_count(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyLTX2Vae()

        # Height/width 28 with stride 12 -> positions 0, 12, 24 -> 3x3 grid
        x = torch.randn(1, 3, 5, 28, 28)
        tiletask_list, grid_spec = encode_tile_split(vae, x)

        assert grid_spec.grid_shape == (3, 3)
        assert len(tiletask_list) == 9
        assert grid_spec.split_dims == (3, 4)
        assert grid_spec.output_dtype == vae.dtype

    def test_tiles_cover_input_with_overlap(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyLTX2Vae()

        x = torch.randn(1, 3, 5, 28, 28)
        tiletask_list, _ = encode_tile_split(vae, x)

        # Interior tiles are min-sized; edge tiles are cropped to the input.
        assert tiletask_list[0].tensor.shape == (1, 3, 5, 16, 16)
        assert tiletask_list[-1].tensor.shape == (1, 3, 5, 4, 4)
        for task in tiletask_list:
            assert task.workload == task.tensor.shape[2] * task.tensor.shape[3] * task.tensor.shape[4]

    def test_tile_spec_latent_dimensions(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyLTX2Vae()

        x = torch.randn(1, 3, 5, 28, 28)
        _, grid_spec = encode_tile_split(vae, x)

        tile_spec = grid_spec.tile_spec
        assert tile_spec["latent_height"] == 7  # 28 // 4
        assert tile_spec["latent_width"] == 7
        assert tile_spec["tile_latent_stride_height"] == 3  # 12 // 4
        assert tile_spec["tile_latent_stride_width"] == 3
        assert tile_spec["blend_height"] == 1  # 16 // 4 - 12 // 4
        assert tile_spec["blend_width"] == 1

    def test_output_shape_metadata_matches_encoder_outputs(self):
        """tile_output_shapes must match encoder outputs exactly (known-metadata gather)."""
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        vae = _DummyLTX2Vae()

        x = torch.randn(1, 3, 5, 28, 28)
        tiletask_list, grid_spec = encode_tile_split(vae, x)

        tile_output_shapes = grid_spec.tile_spec["tile_output_shapes"]
        max_tile_output_shape = grid_spec.tile_spec["max_tile_output_shape"]
        for task in tiletask_list:
            encoded = encode_tile_exec(vae, task)
            assert tuple(encoded.shape) == tile_output_shapes[task.tile_id]
            for actual, upper in zip(encoded.shape, max_tile_output_shape):
                assert actual <= upper

    def test_output_shape_metadata_omitted_for_non_divisible_tiles(self):
        """Non-divisible edge tiles must fall back to the metadata-gather path."""
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyLTX2Vae()

        # Width 26 -> edge tile is 2 wide, not divisible by scr=4.
        x = torch.randn(1, 3, 5, 28, 26)
        _, grid_spec = encode_tile_split(vae, x)

        assert "max_tile_output_shape" not in grid_spec.tile_spec
        assert "tile_output_shapes" not in grid_spec.tile_spec

    def test_latent_num_frames_formula(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyLTX2Vae(temporal_compression_ratio=2)

        x = torch.randn(1, 3, 9, 16, 16)
        _, grid_spec = encode_tile_split(vae, x)

        # (9 - 1) // 2 + 1 = 5 latent frames
        assert grid_spec.tile_spec["tile_output_shapes"][0][2] == 5


class TestEncodeTileExec:
    """Tests for encode_tile_exec."""

    def test_causal_passthrough(self):
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        vae = _DummyLTX2Vae()

        x = torch.randn(1, 3, 5, 16, 16)
        tiletask_list, _ = encode_tile_split(vae, x)

        encode_tile_exec(vae, tiletask_list[0], causal=True)
        assert vae.encoder_calls[-1]["causal"] is True

        encode_tile_exec(vae, tiletask_list[0], causal=None)
        assert vae.encoder_calls[-1]["causal"] is None


class TestTiledEncodeDispatch:
    """Tests that tiled_encode wires the encode operator into the executor correctly."""

    def test_tiled_encode_dispatches_encode_operator_with_broadcast(self):
        from types import SimpleNamespace

        from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_ltx2

        x = torch.zeros(1, 3, 1, 32, 48)
        expected = torch.ones(1, 8, 1, 4, 6)
        seen = {}

        class FakeExecutor:
            def execute(self, tensor, operator, broadcast_result=True):
                seen["tensor"] = tensor
                seen["operator"] = operator
                seen["broadcast_result"] = broadcast_result
                return expected

        cls = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video
        vae = SimpleNamespace(distributed_executor=FakeExecutor(), is_distributed_enabled=lambda: True)
        vae.encode_tile_split = cls.encode_tile_split.__get__(vae)
        vae.encode_tile_exec = cls.encode_tile_exec.__get__(vae)
        vae.encode_tile_merge = cls.encode_tile_merge.__get__(vae)

        def fake_encoder(tensor, causal=None):
            seen["exec_causal"] = causal
            return tensor

        vae.encoder = fake_encoder

        output = cls.tiled_encode(vae, x, causal=True)

        assert output is expected
        assert seen["tensor"] is x
        # Encode latents are consumed by the denoiser on every rank, so the
        # merged result must be broadcast (decode only needs rank 0).
        assert seen["broadcast_result"] is True
        assert seen["operator"].split.__name__ == "encode_tile_split"
        assert seen["operator"].merge.__name__ == "encode_tile_merge"

        # The exec closure must forward the causal flag to the encoder.
        seen["operator"].exec(SimpleNamespace(tensor=x))
        assert seen["exec_causal"] is True


class TestEncodeTileMerge:
    """Tests for encode_tile_merge and end-to-end split/exec/merge equivalence."""

    def test_merged_latent_shape(self):
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        encode_tile_merge = _import_encode_tile_merge()
        vae = _DummyLTX2Vae()

        x = torch.randn(2, 3, 5, 28, 28)
        tiletask_list, grid_spec = encode_tile_split(vae, x)
        coord_tensor_map = {task.grid_coord: encode_tile_exec(vae, task) for task in tiletask_list}
        enc = encode_tile_merge(vae, coord_tensor_map, grid_spec)

        assert enc.shape == (2, 2 * vae.config.latent_channels, 3, 7, 7)

    @pytest.mark.parametrize(
        "height,width",
        [
            (12, 12),  # single tile
            (28, 28),  # 3x3 grid with cropped edge tiles
            (24, 36),  # non-square grid
        ],
    )
    def test_split_exec_merge_matches_sequential_tiled_encode(self, height, width):
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        encode_tile_merge = _import_encode_tile_merge()

        torch.manual_seed(0)
        x = torch.randn(1, 3, 5, height, width)

        vae = _DummyLTX2Vae()
        tiletask_list, grid_spec = encode_tile_split(vae, x)
        coord_tensor_map = {task.grid_coord: encode_tile_exec(vae, task) for task in tiletask_list}
        distributed_enc = encode_tile_merge(vae, coord_tensor_map, grid_spec)

        reference_enc = _reference_tiled_encode(_DummyLTX2Vae(), x)

        assert distributed_enc.shape == reference_enc.shape
        torch.testing.assert_close(distributed_enc, reference_enc)
