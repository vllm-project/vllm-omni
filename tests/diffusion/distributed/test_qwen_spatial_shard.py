# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.autoencoders import AutoencoderKLQwenImage

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(cls=DistributedAutoencoderKLQwenImage, channels=4, full_topology=False):
    return cls(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 2, 4, 4] if full_topology else [1, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True] if full_topology else [True],
        input_channels=channels,
        latents_mean=[0, 0],
        latents_std=[1, 1],
    ).eval()


def _executor(mode, size=2, group=None):
    def no_tiles(*args, **kwargs):
        raise AssertionError("spatial decode must not use tile executor")

    return SimpleNamespace(parallel_mode=mode, parallel_size=size, group=group, execute=no_tiles)


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_uninitialized_spatial_decode_rejected(mode):
    vae = _model()
    vae.distributed_executor = _executor(mode)
    with pytest.raises(RuntimeError, match="initialized"):
        vae.decode(torch.randn(2, 2, 1, 3, 3))


def _worker(rank, path, direction):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        # Include the production spatial/temporal stage layout at small channel width.
        cases = [(3, 2, 4, 1, False), (4, 7, 3, 1, False), (4, 2, 3, 3, False), (4, 2, 1, 1, False), (4, 2, 3, 3, True)]
        for channels, batch, extent, frames, full_topology in cases:
            torch.manual_seed(42)
            reference = _model(AutoencoderKLQwenImage, channels, full_topology)
            vae = _model(channels=channels, full_topology=full_topology)
            vae.load_state_dict(reference.state_dict())
            vae.requires_grad_(False)
            vae.distributed_executor = _executor(f"spatial_shard_{direction}", group=dist.group.WORLD)
            vae.enable_tiling()
            shape = (batch, 2, frames, extent, 3) if direction == "height" else (batch, 2, frames, 3, extent)
            z = torch.randn(shape)
            before = {key: value.clone() for key, value in vae.state_dict().items()}
            with torch.no_grad():
                expected = reference.decode(z).sample
                actual = vae.decode(z).sample
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                assert vae.state_dict().keys() == before.keys()
                for key, value in vae.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                assert not any(p.requires_grad for p in vae.parameters())
                assert not any(m.training for m in vae.modules())
                assert all(v is None for v in vae._feat_map)
                # A second request and direct tiled_decode retain the explicit mode.
                torch.testing.assert_close(vae.tiled_decode(z, return_dict=False)[0], expected, rtol=1e-4, atol=1e-5)
                vae.enable_slicing()
                torch.testing.assert_close(vae.decode(z, return_dict=False)[0], expected, rtol=1e-4, atol=1e-5)
                vae.disable_slicing()
                vae.disable_tiling()
                image = torch.randn(batch, channels, 1, 8, 8)
                reference.encode(image)
                vae.encode(image)
                torch.testing.assert_close(vae.decode(z).sample, expected, rtol=1e-4, atol=1e-5)
            for bad_mode in ["tile", "spatial_shard_width" if direction == "height" else "spatial_shard_height"]:
                vae.distributed_executor.parallel_mode = bad_mode
                for decode in (vae.decode, vae.tiled_decode):
                    with pytest.raises(ValueError, match="fresh VAE"):
                        decode(z)
            vae.distributed_executor.parallel_mode = f"spatial_shard_{direction}"
            vae.distributed_executor.group = object()
            with pytest.raises(ValueError, match="fresh VAE"):
                vae.decode(z)
            vae.distributed_executor.group = dist.group.WORLD
            vae.distributed_executor.parallel_size = 1
            with pytest.raises(ValueError, match="parallel"):
                vae.decode(z)
        vae = _model()
        vae.distributed_executor = _executor(f"spatial_shard_{direction}", size=3, group=dist.group.WORLD)
        with pytest.raises(ValueError, match="parallel"):
            vae.decode(z)
        assert not getattr(vae, "_qwen_spatial_shard_config", None)
        vae.distributed_executor = _executor(f"spatial_shard_{direction}")
        with pytest.raises(ValueError, match="process group"):
            vae.decode(z)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("direction", ["height", "width"])
def test_real_qwen_distributed_decode(tmp_path, direction):
    mp.start_processes(_worker, args=(str(tmp_path / "gloo"), direction), nprocs=2, join=True, start_method="spawn")


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_single_rank_full_decode_ignores_tile_threshold(mode):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    vae = _model()
    reference = _model(AutoencoderKLQwenImage)
    reference.load_state_dict(vae.state_dict())
    vae.distributed_executor = _executor(mode, size=1)
    vae.enable_tiling(tile_sample_min_height=2, tile_sample_min_width=2)
    z = torch.randn(2, 2, 2, 3, 3)
    with torch.no_grad():
        expected = reference.decode(z).sample
        torch.testing.assert_close(vae.decode(z).sample, expected)
        torch.testing.assert_close(vae.tiled_decode(z, return_dict=False)[0], expected)
    assert not getattr(vae, "_qwen_spatial_shard_config", None)


def test_cache_cleared_on_decoder_exception():
    vae = _model()
    vae.distributed_executor = _executor("spatial_shard_height", size=1)

    def fail(module, args):
        vae._feat_map[0] = args[0].clone()
        raise RuntimeError("decoder failed")

    handle = vae.decoder.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="decoder failed"):
            vae.decode(torch.randn(2, 2, 1, 3, 3))
    finally:
        handle.remove()
    assert all(v is None for v in vae._feat_map)
    assert vae._conv_idx == [0]
