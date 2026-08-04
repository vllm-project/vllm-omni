# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU NCCL coverage for cooperative checkpoint staging."""

from collections.abc import Iterator
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.model_loader.cooperative_staging import (
    _TorchDistComm,
    cooperative_staging_weights_iterator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]


def _stream() -> Iterator[tuple[str, torch.Tensor]]:
    # Four entries fit in each 200 KiB BF16 output bucket. Twelve entries
    # therefore produce three buckets whose deterministic owners are 0, 1, 0.
    for index in range(12):
        yield f"layer.{index}.weight", torch.full((24576,), float(index), dtype=torch.float32)


def _run_nccl_rank(rank: int, world_size: int, master_port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        device = torch.device("cuda", rank)
        comm = _TorchDistComm(dist.group.WORLD, device)
        count = 0
        for name, tensor in cooperative_staging_weights_iterator(
            _stream(),
            comm=comm,
            bucket_bytes=200 << 10,
            default_dtype=torch.bfloat16,
        ):
            index = int(name.split(".")[1])
            assert tensor.device == device
            assert tensor.dtype is torch.bfloat16
            assert torch.equal(tensor, torch.full_like(tensor, float(index)))
            count += 1

        assert count == 12
        dist.barrier()
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_cooperative_staging_tp2_nccl() -> None:
    """Both real GPU ranks stage owned buckets and receive identical tensors."""
    if not dist.is_nccl_available():
        pytest.skip("NCCL is required for the TP=2 cooperative staging test")

    world_size = 2
    torch.multiprocessing.spawn(
        _run_nccl_rank,
        args=(world_size, get_open_port()),
        nprocs=world_size,
        join=True,
    )
