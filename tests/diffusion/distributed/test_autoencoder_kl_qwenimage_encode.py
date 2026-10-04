# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DistributedAutoencoderKLQwenImage encode parallel (CPU-only)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _DummyConfig:
    def __init__(self, temperal_downsample=(False, True)):
        self.temperal_downsample = list(temperal_downsample)


class _DummyQwenImageVae:
    """Minimal mock of DistributedAutoencoderKLQwenImage for the encode tile ops.

    Follows the Wan-style causal encoder contract used by the QwenImage VAE:
    the encoder consumes temporal chunks (1 frame, then tcr frames per chunk)
    with an instance-level feature cache, emits one latent frame per chunk,
    and `quant_conv` runs on each chunk's output. The mock is deterministic
    and content-dependent so merge equivalence against the sequential
    (diffusers-style) tiled encode is meaningful.
    """

    def __init__(
        self,
        config=None,
        spatial_compression_ratio=4,
        z_dim=2,
        tile_sample_min_height=16,
        tile_sample_min_width=16,
        tile_sample_stride_height=12,
        tile_sample_stride_width=12,
    ):
        self.config = config or _DummyConfig()
        self.spatial_compression_ratio = spatial_compression_ratio
        self.z_dim = z_dim
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width
        self.dtype = torch.float32

        self._enc_feat_map = None
        self._enc_conv_idx = [0]
        self.clear_cache_calls = 0
        self.encoder_feat_idx_at_call = []

    def clear_cache(self):
        self.clear_cache_calls += 1
        self._enc_feat_map = None
        self._enc_conv_idx = [0]

    def encoder(self, x, feat_cache=None, feat_idx=None):  # noqa: ARG002
        self.encoder_feat_idx_at_call.append(list(feat_idx) if feat_idx is not None else None)
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio
        pooled = F.adaptive_avg_pool3d(x, (1, latent_height, latent_width))

        out_channels = 2 * self.z_dim
        repeats = (out_channels + num_channels - 1) // num_channels
        out = pooled.repeat(1, repeats, 1, 1, 1)[:, :out_channels]
        scale = torch.arange(1, out_channels + 1, dtype=out.dtype).view(1, -1, 1, 1, 1)
        return out * scale

    def quant_conv(self, x):
        # Deterministic stand-in for the 1x1x1 causal conv.
        return x * 0.5 + 0.25

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b


def _import_encode_tile_split():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
        DistributedAutoencoderKLQwenImage,
    )

    return DistributedAutoencoderKLQwenImage.encode_tile_split


def _import_encode_tile_exec():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
        DistributedAutoencoderKLQwenImage,
    )

    return DistributedAutoencoderKLQwenImage.encode_tile_exec


def _import_encode_tile_merge():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
        DistributedAutoencoderKLQwenImage,
    )

    return DistributedAutoencoderKLQwenImage.encode_tile_merge


def _reference_tiled_encode(vae, x):
    """Sequential tiled encode mirroring diffusers AutoencoderKLQwenImage.tiled_encode."""
    _, _, num_frames, height, width = x.shape
    latent_height = height // vae.spatial_compression_ratio
    latent_width = width // vae.spatial_compression_ratio

    tile_latent_min_height = vae.tile_sample_min_height // vae.spatial_compression_ratio
    tile_latent_min_width = vae.tile_sample_min_width // vae.spatial_compression_ratio
    tile_latent_stride_height = vae.tile_sample_stride_height // vae.spatial_compression_ratio
    tile_latent_stride_width = vae.tile_sample_stride_width // vae.spatial_compression_ratio

    blend_height = tile_latent_min_height - tile_latent_stride_height
    blend_width = tile_latent_min_width - tile_latent_stride_width

    temporal_compression = 2 ** sum(bool(t) for t in vae.config.temperal_downsample)

    rows = []
    for i in range(0, height, vae.tile_sample_stride_height):
        row = []
        for j in range(0, width, vae.tile_sample_stride_width):
            vae.clear_cache()
            time = []
            frame_range = 1 + (num_frames - 1) // temporal_compression
            for k in range(frame_range):
                vae._enc_conv_idx = [0]
                if k == 0:
                    tile = x[:, :, :1, i : i + vae.tile_sample_min_height, j : j + vae.tile_sample_min_width]
                else:
                    tile = x[
                        :,
                        :,
                        1 + temporal_compression * (k - 1) : 1 + temporal_compression * k,
                        i : i + vae.tile_sample_min_height,
                        j : j + vae.tile_sample_min_width,
                    ]
                tile = vae.encoder(tile, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx)
                tile = vae.quant_conv(tile)
                time.append(tile)
            row.append(torch.cat(time, dim=2))
        rows.append(row)
    vae.clear_cache()

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))

    return torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]


class TestEncodeTileSplit:
    """Tests for encode_tile_split."""

    def test_grid_shape_and_tile_count(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyQwenImageVae()

        # Height/width 28 with stride 12 -> positions 0, 12, 24 -> 3x3 grid
        x = torch.randn(1, 3, 1, 28, 28)
        tiletask_list, grid_spec = encode_tile_split(vae, x)

        assert grid_spec.grid_shape == (3, 3)
        assert len(tiletask_list) == 9
        assert grid_spec.split_dims == (3, 4)
        assert grid_spec.output_dtype == vae.dtype

    def test_temporal_chunking_follows_config(self):
        encode_tile_split = _import_encode_tile_split()

        # temperal_downsample [False, True] -> temporal compression 2
        vae_2x = _DummyQwenImageVae(config=_DummyConfig((False, True)))
        # temperal_downsample [False, True, True] -> temporal compression 4
        vae_4x = _DummyQwenImageVae(config=_DummyConfig((False, True, True)))

        x = torch.randn(1, 3, 9, 16, 16)
        tasks_2x, _ = encode_tile_split(vae_2x, x)
        tasks_4x, _ = encode_tile_split(vae_4x, x)

        # 2x: 1 + (9-1)//2 = 5 chunks; 4x: 1 + (9-1)//4 = 3 chunks
        assert len(tasks_2x[0].tensor) == 5
        assert len(tasks_4x[0].tensor) == 3
        # First chunk is the single leading frame, later chunks are tcr frames.
        assert tasks_2x[0].tensor[0].shape[2] == 1
        assert tasks_2x[0].tensor[1].shape[2] == 2
        assert tasks_4x[0].tensor[1].shape[2] == 4

    def test_tile_spec_latent_dimensions(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyQwenImageVae()

        x = torch.randn(1, 3, 1, 28, 28)
        _, grid_spec = encode_tile_split(vae, x)

        tile_spec = grid_spec.tile_spec
        assert tile_spec["latent_height"] == 7  # 28 // 4
        assert tile_spec["latent_width"] == 7
        assert tile_spec["tile_latent_stride_height"] == 3  # 12 // 4
        assert tile_spec["tile_latent_stride_width"] == 3
        assert tile_spec["blend_height"] == 1  # 16 // 4 - 12 // 4
        assert tile_spec["blend_width"] == 1

    def test_edge_tiles_cropped_and_workloads_set(self):
        encode_tile_split = _import_encode_tile_split()
        vae = _DummyQwenImageVae()

        x = torch.randn(1, 3, 1, 28, 28)
        tiletask_list, _ = encode_tile_split(vae, x)

        assert tiletask_list[0].tensor[0].shape[-2:] == (16, 16)
        assert tiletask_list[-1].tensor[0].shape[-2:] == (4, 4)
        for task in tiletask_list:
            assert task.workload == task.tensor[0].shape[3] * task.tensor[0].shape[4]


class TestEncodeTileExec:
    """Tests for encode_tile_exec."""

    def test_feat_cache_reset_per_chunk_and_quant_conv_applied(self):
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        vae = _DummyQwenImageVae()

        x = torch.randn(1, 3, 5, 16, 16)  # tcr=2 -> 3 chunks
        tiletask_list, _ = encode_tile_split(vae, x)

        result = encode_tile_exec(vae, tiletask_list[0])

        # One latent frame per temporal chunk.
        assert result.shape[2] == 3
        # feat_idx was reset to [0] before every encoder call.
        assert vae.encoder_feat_idx_at_call[-3:] == [[0], [0], [0]]
        # Cache cleared at entry and exit.
        assert vae.clear_cache_calls >= 2


class TestTiledEncodeDispatch:
    """Tests that tiled_encode wires the encode operator into the executor correctly."""

    def test_tiled_encode_dispatches_encode_operator_with_broadcast(self):
        from types import SimpleNamespace

        from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_qwenimage

        x = torch.zeros(1, 3, 1, 32, 48)
        expected = torch.ones(1, 4, 1, 8, 12)
        seen = {}

        class FakeExecutor:
            def execute(self, tensor, operator, broadcast_result=True):
                seen["tensor"] = tensor
                seen["operator"] = operator
                seen["broadcast_result"] = broadcast_result
                return expected

        cls = autoencoder_kl_qwenimage.DistributedAutoencoderKLQwenImage
        vae = SimpleNamespace(
            distributed_executor=FakeExecutor(),
            is_distributed_enabled=lambda: True,
            clear_cache=lambda: seen.setdefault("cache_cleared", True),
        )
        vae.encode_tile_split = cls.encode_tile_split.__get__(vae)
        vae.encode_tile_exec = cls.encode_tile_exec.__get__(vae)
        vae.encode_tile_merge = cls.encode_tile_merge.__get__(vae)

        output = cls.tiled_encode(vae, x)

        assert output is expected
        assert seen["tensor"] is x
        # Encode latents are consumed by the denoiser on every rank, so the
        # merged result must be broadcast.
        assert seen["broadcast_result"] is True
        assert seen["operator"].split.__name__ == "encode_tile_split"
        assert seen["operator"].exec.__name__ == "encode_tile_exec"
        assert seen["operator"].merge.__name__ == "encode_tile_merge"
        assert seen["cache_cleared"] is True


class TestTiledDecodeDispatch:
    """The edit-pipeline swap turns decode parallel on for the first time; pin its wiring."""

    def test_tiled_decode_dispatches_decode_operator(self):
        from types import SimpleNamespace

        from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_qwenimage

        z = torch.zeros(1, 4, 1, 8, 12)
        expected = torch.ones(1, 3, 1, 32, 48)
        seen = {}

        class FakeExecutor:
            def execute(self, tensor, operator, broadcast_result=True):
                seen["tensor"] = tensor
                seen["operator"] = operator
                seen["broadcast_result"] = broadcast_result
                return expected

        cls = autoencoder_kl_qwenimage.DistributedAutoencoderKLQwenImage
        vae = SimpleNamespace(distributed_executor=FakeExecutor(), is_distributed_enabled=lambda: True)
        vae.tile_split = cls.tile_split.__get__(vae)
        vae.tile_exec = cls.tile_exec.__get__(vae)
        vae.tile_merge = cls.tile_merge.__get__(vae)

        output = cls.tiled_decode(vae, z, return_dict=False)

        assert len(output) == 1
        assert output[0] is expected
        assert seen["tensor"] is z
        assert seen["broadcast_result"] is True
        assert seen["operator"].split.__name__ == "tile_split"
        assert seen["operator"].merge.__name__ == "tile_merge"


class _SingleRankExecutor:
    """Real DistributedVaeExecutor logic with rank-0/world-1 collectives stubbed out.

    Runs the genuine split -> balance -> pack -> unpack -> merge machinery on one
    process so CI can exercise the executor path against a real VAE without a
    torch.distributed process group.
    """

    def __new__(cls):
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
            DistributedVaeExecutor,
        )

        executor = object.__new__(DistributedVaeExecutor)
        executor.parallel_size = 1
        executor.world_size = 1
        executor.rank = 0
        executor.parallel_mode = "tile"
        executor.gather_tensors = lambda tensor: [tensor]
        executor._sync_final_result = lambda result, *_args, **_kwargs: result

        def _local_padding_shape(local_results, output_ndim, device):  # noqa: ARG001
            dims = [0] * output_ndim
            for _, tile_tensor in local_results:
                for idx, size in enumerate(tile_tensor.shape):
                    dims[idx] = max(dims[idx], size)
            return [len(local_results), *dims]

        executor._compute_global_padding_shape = _local_padding_shape
        return executor


class TestRealVaeExecutorPath:
    """End-to-end: real tiny AutoencoderKLQwenImage through the real executor logic."""

    def test_distributed_tiled_encode_matches_sequential_on_real_vae(self):
        from diffusers.models.autoencoders import AutoencoderKLQwenImage

        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
            DistributedAutoencoderKLQwenImage,
        )

        torch.manual_seed(0)
        vae = DistributedAutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 1],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        vae.eval()
        vae.distributed_executor = _SingleRankExecutor()
        vae.enable_tiling(
            tile_sample_min_height=16,
            tile_sample_min_width=16,
            tile_sample_stride_height=8,
            tile_sample_stride_width=8,
        )
        vae.is_distributed_enabled = lambda: True

        torch.manual_seed(1)
        x = torch.randn(1, 3, 1, 24, 24)
        with torch.no_grad():
            reference = AutoencoderKLQwenImage.tiled_encode(vae, x)
            distributed = vae.tiled_encode(x)

        assert distributed.shape == reference.shape
        torch.testing.assert_close(distributed, reference, rtol=0, atol=0)


class TestEncodeTileMerge:
    """Tests for encode_tile_merge and end-to-end split/exec/merge equivalence."""

    def test_merged_latent_shape(self):
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        encode_tile_merge = _import_encode_tile_merge()
        vae = _DummyQwenImageVae()

        x = torch.randn(2, 3, 5, 28, 28)  # tcr=2 -> 3 latent frames
        tiletask_list, grid_spec = encode_tile_split(vae, x)
        coord_tensor_map = {task.grid_coord: encode_tile_exec(vae, task) for task in tiletask_list}
        enc = encode_tile_merge(vae, coord_tensor_map, grid_spec)

        assert enc.shape == (2, 2 * vae.z_dim, 3, 7, 7)

    @pytest.mark.parametrize(
        "height,width,num_frames",
        [
            (12, 12, 1),  # single tile, single frame (image editing case)
            (28, 28, 1),  # 3x3 grid with cropped edge tiles
            (20, 28, 1),  # odd-size edge tiles
            (24, 36, 5),  # non-square grid, multi-frame
        ],
    )
    def test_split_exec_merge_matches_sequential_tiled_encode(self, height, width, num_frames):
        encode_tile_split = _import_encode_tile_split()
        encode_tile_exec = _import_encode_tile_exec()
        encode_tile_merge = _import_encode_tile_merge()

        torch.manual_seed(0)
        x = torch.randn(1, 3, num_frames, height, width)

        vae = _DummyQwenImageVae()
        tiletask_list, grid_spec = encode_tile_split(vae, x)
        coord_tensor_map = {task.grid_coord: encode_tile_exec(vae, task) for task in tiletask_list}
        distributed_enc = encode_tile_merge(vae, coord_tensor_map, grid_spec)

        reference_enc = _reference_tiled_encode(_DummyQwenImageVae(), x)

        assert distributed_enc.shape == reference_enc.shape
        torch.testing.assert_close(distributed_enc, reference_enc)
