# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kandinsky 6 VAE patch parallel keeps the single-GPU tile grid."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.models.kandinsky6.modeling_kandinsky6_vae import AutoencoderKLHunyuanVideo


def _stub_vae() -> AutoencoderKLHunyuanVideo:
    vae = AutoencoderKLHunyuanVideo.__new__(AutoencoderKLHunyuanVideo)
    vae.spatial_compression_ratio = 8
    vae.temporal_compression_ratio = 4
    vae.use_tiling = True
    vae.use_framewise_decoding = True
    vae.tile_sample_min_height = 32
    vae.tile_sample_min_width = 32
    vae.tile_sample_min_num_frames = 16
    vae.tile_sample_stride_height = 24
    vae.tile_sample_stride_width = 24
    vae.tile_sample_stride_num_frames = 12
    vae.tile_size = None
    vae.distributed_executor = None
    vae.post_quant_conv = lambda tensor: tensor
    vae._seen_widths = []

    def decoder(tensor: torch.Tensor) -> torch.Tensor:
        vae._seen_widths.append(int(tensor.shape[-1]))
        return tensor + 1

    vae.decoder = decoder
    return vae


@pytest.mark.core_model
@pytest.mark.cpu
def test_distributed_decode_tiles_match_local_tiled_decode():
    vae = _stub_vae()
    latent = torch.arange(1 * 4 * 2 * 8 * 12, dtype=torch.float32).reshape(1, 4, 2, 8, 12)

    local = vae.tiled_decode(latent).sample
    tasks, spec = vae._decode_tile_split(latent)
    assert len(tasks) > 1
    assert max(int(task.tensor.shape[-1]) for task in tasks) < latent.shape[-1]

    decoded = {task.grid_coord: vae._decode_tile_exec(task) for task in tasks}
    merged = vae._decode_tile_merge(decoded, spec)

    torch.testing.assert_close(merged, local)


@pytest.mark.core_model
@pytest.mark.cpu
def test_decode_still_applies_optimal_tiling_when_patch_parallel_is_on():
    vae = _stub_vae()
    seen = {}

    def tiling(shape, device=None):
        del device
        seen["shape"] = tuple(shape)
        return (1, 17, 32, 32), (8, 24, 24)

    from diffusers.models.autoencoders.vae import DecoderOutput

    vae.get_dec_optimal_tiling = tiling
    vae._decode = lambda z, return_dict=True: DecoderOutput(sample=z)
    vae.distributed_executor = SimpleNamespace(parallel_size=2)
    vae.is_distributed_enabled = lambda: True

    out = vae.decode(torch.zeros(1, 16, 8, 8, 12))

    assert seen["shape"] == (1, 16, 8, 8, 12)
    assert tuple(out.sample.shape) == (1, 16, 8, 8, 12)


def _patch_parallel_worker(rank: int, port: str, shapes: dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        import vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor as executor_mod

        world = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.group.WORLD)
        executor_mod.get_world_group = lambda: world

        vae = _stub_vae()
        latent = torch.arange(1 * 4 * 2 * 8 * 12, dtype=torch.float32).reshape(1, 4, 2, 8, 12)
        reference = vae.tiled_decode(latent).sample.clone()
        vae._seen_widths.clear()
        vae.set_parallel_size(2, mode="tile")
        decoded = vae.tiled_decode(latent).sample
        torch.testing.assert_close(decoded, reference)
        if rank == 0:
            shapes["widths"] = list(vae._seen_widths)
            shapes["full_width"] = int(latent.shape[-1])
    finally:
        dist.destroy_process_group()


@pytest.mark.core_model
@pytest.mark.cpu
def test_patch_parallel_decode_splits_tiles_across_ranks():
    manager = mp.Manager()
    shapes = manager.dict()
    port = str(29500 + (os.getpid() % 1000))
    mp.spawn(_patch_parallel_worker, args=(port, shapes), nprocs=2, join=True)

    assert shapes["widths"]
    assert max(shapes["widths"]) < shapes["full_width"]
