# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run inside npu run with torchrun --nproc-per-node=<allocated> -m pytest -q <this file>."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]


@pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="Requires a scheduled multi-NPU torchrun job")
def test_production_shmem_matches_hccl():
    import torch_npu  # noqa: F401

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.distributed.a2a_permute import clear_a2a_permute_workspaces
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "Submit device work through npu run"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    group = SimpleNamespace(
        ulysses_group=dist.group.WORLD,
        ulysses_world_size=dist.get_world_size(),
        ulysses_rank=dist.get_rank(),
        ring_world_size=1,
    )
    strategies = [
        UlyssesParallelAttention(group, 2, 1, use_sync, ulysses_a2a_permute=True) for use_sync in (False, True)
    ]
    device = torch.device("npu", local_rank)
    torch.manual_seed(6340 + dist.get_rank())

    def assert_bitwise(actual, expected):
        equal = torch.equal(actual.cpu().contiguous().view(torch.uint8), expected.cpu().contiguous().view(torch.uint8))
        failed = torch.tensor(int(not equal), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        assert failed.item() == 0, "SHMEM/HCCL output differs on at least one rank"

    try:
        with torch.inference_mode():
            # Packed strided views, GQA, workspace growth to the actual model
            # shape, then a smaller request reusing that larger allocation.
            for batch, seq, heads, kv_heads, dim in [
                (2, 17, 8, 8, 32),
                (1, 48, 16, 8, 64),
                (1, 9440, 56, 56, 128),
                (2, 17, 8, 8, 32),
            ]:
                for strategy in strategies:
                    packed = torch.randn(batch, seq, 3, heads, dim, device=device, dtype=torch.bfloat16)
                    query, key, value = packed.unbind(2)
                    key, value = key[:, :, :kv_heads], value[:, :, :kv_heads]
                    metadata = AttentionMetadata(extra={"gate_compress": query})
                    q, k, v, metadata, context = strategy.pre_attention(query, key, value, metadata)
                    for actual, source in zip((q, k, v), (query, key, value)):
                        assert_bitwise(actual, SeqAllToAll4D.apply(dist.group.WORLD, source, 2, 1, False))
                    assert_bitwise(metadata.extra["gate_compress"], q)
                    output = strategy.post_attention(q, context)
                    assert_bitwise(output, query)
                    assert_bitwise(output, SeqAllToAll4D.apply(dist.group.WORLD, q, 1, 2, False))
    finally:
        clear_a2a_permute_workspaces()
        dist.destroy_process_group()
