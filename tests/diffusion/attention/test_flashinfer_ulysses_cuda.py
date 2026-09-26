# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _worker(rank, rendezvous):
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
        _gather_impl,
        _nccl_gather_heads,
        _nccl_scatter_heads,
        _scatter_impl,
        clear_flashinfer_ulysses_communicators,
    )
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    try:
        group = dist.group.WORLD
        for batch, rows, dtype in [(1, 128, torch.bfloat16), (1, 64, torch.bfloat16), (1, 128, torch.float8_e4m3fn)]:
            raw = torch.arange(batch * rows * 8 * 128, device=device).reshape(batch, rows, 8, 128)
            x = ((raw % 101) / 100 + rank).to(dtype)[..., ::2]
            references, outputs = [], []
            for slot in ("q", "k", "v", "g"):
                expected = _nccl_scatter_heads(x, group, 2)
                actual = _scatter_impl(x, group.group_name, 2, slot, False)
                assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
                references.append(expected)
                outputs.append(actual)
            assert len({x.data_ptr() for x in outputs}) == 4
            for expected, actual in zip(references, outputs, strict=True):
                assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
            reversed_actual = _gather_impl(outputs[0], group.group_name, 2, False)
            reversed_expected = _nccl_gather_heads(references[0], group, 2)
            assert torch.equal(reversed_actual.view(torch.uint8), reversed_expected.view(torch.uint8))
            assert torch.equal(reversed_actual.view(torch.uint8), x.contiguous().view(torch.uint8))
        x = torch.randn(2, 64, 8, 64, device=device, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="batch=1"):
            _scatter_impl(x, group.group_name, 2, "q", False)
        with pytest.raises(ValueError, match="batch=1"):
            _gather_impl(x, group.group_name, 2, False)
        os.environ["VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA"] = "0"
        actual = _scatter_impl(x, group.group_name, 2, "q", False)
        assert torch.equal(actual, _nccl_scatter_heads(x, group, 2))
        assert torch.equal(_gather_impl(actual, group.group_name, 2, False), x)
    finally:
        clear_flashinfer_ulysses_communicators()
        dist.destroy_process_group()


@hardware_test(res={"cuda": ["B200"]}, num_cards=2)
def test_two_rank_registered_rdma_matches_nccl(tmp_path, monkeypatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("requires two CUDA devices and RDMA topology")
    pytest.importorskip("flashinfer.comm.ulysses")
    monkeypatch.setenv("FLASHINFER_ULYSSES_PCIE_ROUTE", "rdma")
    monkeypatch.setenv("VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA", "1")
    monkeypatch.setenv("VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES", str(4 * 1024**2))
    context = mp.spawn(_worker, args=(str(tmp_path / "rdma-init"),), nprocs=2, join=False)
    try:
        if not context.join(timeout=300) and not context.join(timeout=30):
            pytest.fail("RDMA workers exceeded their deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
