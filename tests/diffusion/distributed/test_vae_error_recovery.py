# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor as vae_executor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class TileOperator:
    def split(self, tensor):
        return [vae_executor.TileTask(i, (i,), row[None]) for i, row in enumerate(tensor)], vae_executor.GridSpec(
            split_dims=(0,), grid_shape=(tensor.shape[0],)
        )

    def exec(self, task):
        return task.tensor * 2

    def merge(self, tiles, grid):
        return torch.cat([tiles[(i,)] for i in range(grid.grid_shape[0])])


def run_recovery_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.group.WORLD)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(vae_executor, "get_world_group", lambda: group)
            executor = vae_executor.DistributedVaeExecutor()
        executor.set_parallel_size(2)
        tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        operator = TileOperator()
        for stage, failing_rank in [("decode", 1), ("packing", 1), ("merge", 0), ("gather", 1)]:
            with pytest.MonkeyPatch.context() as patch:
                if rank == failing_rank:

                    def fail(*args, **kwargs):
                        raise torch.OutOfMemoryError("injected OOM")

                    target, attribute = {
                        "decode": (operator, "exec"),
                        "packing": (executor, "_pack_local_tiles"),
                        "merge": (operator, "merge"),
                        "gather": (torch, "empty_like"),
                    }[stage]
                    patch.setattr(target, attribute, fail)
                with pytest.raises(torch.OutOfMemoryError, match="Distributed VAE"):
                    executor.execute(tensor, operator)
            torch.testing.assert_close(executor.execute(tensor, operator), tensor * 2)

        with pytest.raises(RuntimeError, match="Distributed VAE mixed failures failed"):
            with executor._sync_errors("mixed failures"):
                if rank == 0:
                    raise torch.OutOfMemoryError("retryable")
                raise ValueError("not retryable")
        torch.testing.assert_close(executor.execute(tensor, operator), tensor * 2)
    finally:
        dist.destroy_process_group()


def test_all_ranks_observe_failures_and_can_retry(tmp_path):
    torch.multiprocessing.spawn(run_recovery_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
