# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise root-only VAE gathering with real NCCL and CPU error collectives."""

from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor as vae_executor
from vllm_omni.platforms import current_omni_platform


@dataclass(frozen=True)
class _WorldGroups:
    device_group: dist.ProcessGroup
    cpu_group: dist.ProcessGroup


class _TileOperator:
    def split(self, tensor):
        return [vae_executor.TileTask(i, (i,), row[None]) for i, row in enumerate(tensor)], vae_executor.GridSpec(
            split_dims=(0,), grid_shape=(tensor.shape[0],)
        )

    def exec(self, task):
        return task.tensor * 2

    def merge(self, tiles, grid):
        return torch.cat([tiles[(i,)] for i in range(grid.grid_shape[0])])


def _run_gather_worker(rank: int, world_size: int, init_file: str):
    """Test dedicated WORLD groups, idle ranks, root OOM agreement, and retry."""
    torch.set_num_threads(1)
    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        ranks = list(range(world_size))
        device_group = dist.new_group(ranks, backend="nccl")
        error_group = dist.new_group(ranks, backend="gloo")
        group = _WorldGroups(device_group=device_group, cpu_group=error_group)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(vae_executor, "get_world_group", lambda: group)
            executor = vae_executor.DistributedVaeExecutor()
        executor.set_parallel_size(world_size - 1)
        tensor = torch.arange((world_size - 1) * 3, dtype=torch.float32, device=device).reshape(world_size - 1, 3)
        operator = _TileOperator()
        for broadcast_result in (False, True):
            output = executor.execute(tensor, operator, broadcast_result=broadcast_result)
            if broadcast_result or rank == 0:
                torch.testing.assert_close(output, tensor * 2, rtol=0, atol=0)
            else:
                assert output.numel() == 0

        with pytest.MonkeyPatch.context() as patch:
            if rank == 0:

                def fail(*args, **kwargs):
                    raise torch.OutOfMemoryError("injected root receive allocation failure")

                patch.setattr(torch, "empty_like", fail)
            with pytest.raises(torch.OutOfMemoryError, match="gather allocation"):
                executor.execute(tensor, operator)
        torch.testing.assert_close(executor.execute(tensor, operator), tensor * 2, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _check_gather(world_size: int, tmp_path):
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA devices")
    torch.multiprocessing.spawn(
        _run_gather_worker, args=(world_size, str(tmp_path / "rendezvous")), nprocs=world_size, join=True
    )


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_root_only_gather_two_gpus(tmp_path):
    _check_gather(2, tmp_path)


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=4)
def test_root_only_gather_four_gpus(tmp_path):
    _check_gather(4, tmp_path)
