# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor as executor_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


class _RecordingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.inputs = []

    def forward(self, z):
        self.inputs.append(z.clone())
        return z.repeat(1, 3, 1, 1)


def _make_vae():
    vae = DistributedAutoencoderKL.__new__(DistributedAutoencoderKL)
    nn.Module.__init__(vae)
    vae.register_to_config(use_post_quant_conv=False, block_out_channels=[1])
    vae.decoder = _RecordingDecoder()
    vae.post_quant_conv = None
    vae.use_tiling = True
    vae.tile_latent_min_size = vae.tile_sample_min_size = 4
    vae.tile_overlap_factor = 0.5
    return vae


def _slicing_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                executor_module,
                "get_world_group",
                lambda: SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.group.WORLD),
            )
            vae = _make_vae()
            vae.init_distributed()
            vae.set_parallel_size(2)
            execute = vae.distributed_executor.execute
            execute_batches = []

            def record_execute(z, operator, **kwargs):
                execute_batches.append(z.shape[0])
                return execute(z, operator, **kwargs)

            patch.setattr(vae.distributed_executor, "execute", record_execute)
            for spatial_size in (4, 6):  # patch and tiled strategies
                for slicing in (False, True):
                    vae.use_slicing = slicing
                    for batch_size in (3, 2, 1):
                        for return_dict in (False, True):
                            z = torch.arange(1, batch_size + 1, dtype=torch.float32).reshape(-1, 1, 1, 1)
                            z = z.expand(-1, 1, spatial_size, spatial_size).clone()
                            vae.decoder.inputs.clear()
                            execute_batches.clear()
                            with torch.no_grad():
                                result = vae.decode(z, return_dict=return_dict)
                            output = result.sample if return_dict else result[0]
                            assert execute_batches == [batch_size]
                            assert vae.decoder.inputs
                            expected_width = 1 if slicing else batch_size
                            assert all(tensor.shape[0] == expected_width for tensor in vae.decoder.inputs)
                            if slicing and batch_size > 1:
                                split, _, _ = vae._strategy_select(z)
                                tasks, grid = split(z)
                                assert grid.split_dims == (0, 2, 3)
                                assert grid.grid_shape[0] == batch_size
                                batch_order = [task.grid_coord[0] for task in tasks]
                                assert batch_order == sorted(batch_order)
                                assert set(batch_order) == set(range(batch_size))
                                assert [task.tile_id for task in tasks] == list(range(len(tasks)))
                                layouts = [None, None]
                                dist.all_gather_object(layouts, [task.grid_coord for task in tasks])
                                assert layouts[0] == layouts[1]
                            if rank == 0:
                                torch.testing.assert_close(output, z.repeat(1, 3, 1, 1), rtol=0, atol=0)
                            else:
                                assert output.numel() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires the CPU Gloo backend")
def test_spatial_decode_preserves_slicing_batch_width_and_rank_order(tmp_path):
    mp.spawn(_slicing_worker, args=(f"file://{tmp_path / 'slicing'}",), nprocs=2, join=True)


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_autoencoder_kl_rejects_unsupported_parallel_modes(mode):
    vae = _make_vae()
    vae.distributed_executor = SimpleNamespace(set_parallel_size=lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="supports only.*tile.*batch"):
        vae.set_parallel_size(2, mode=mode)
