# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run inside npu run with torchrun --nproc-per-node=<allocated> -m pytest -q <this file>."""

import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A2"}, num_cards=2)]


@pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="Requires a scheduled multi-NPU torchrun job")
def test_production_symmetric_memory_matches_hccl():
    import torch_npu  # noqa: F401

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.distributed.a2a_permute import (
        clear_a2a_permute_workspaces,
        ulysses_o_rev,
        ulysses_qkv_fwd_group,
    )
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "Submit device work through npu run"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    world = dist.get_world_size()
    assert world > 1
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

    def divisible_heads(target):
        return ((target + world - 1) // world) * world

    def assert_bitwise(actual, expected):
        equal = torch.equal(actual.cpu().contiguous().view(torch.uint8), expected.cpu().contiguous().view(torch.uint8))
        failed = torch.tensor(int(not equal), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        assert failed.item() == 0, "Symmetric memory/HCCL output differs on at least one rank"

    try:
        with torch.inference_mode():
            # Packed strided views, GQA, workspace growth to the actual model
            # shape, then a smaller request reusing that larger allocation.
            for batch, seq, heads, kv_heads, dim in [
                (1, 0, divisible_heads(32), divisible_heads(8), 128),  # Hunyuan CFG prefill: joint text only.
                (2, 17, divisible_heads(8), divisible_heads(8), 32),
                (1, 48, divisible_heads(16), divisible_heads(8), 64),
                (1, 9440, divisible_heads(56), divisible_heads(56), 128),
                (1, 0, divisible_heads(32), divisible_heads(8), 128),
                (2, 17, divisible_heads(8), divisible_heads(8), 32),
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

            # Queue exchanges back-to-back without host synchronization, with
            # one rank arriving late. Compare all supported dtypes to HCCL.
            for index, dtype in enumerate((torch.bfloat16, torch.float16, torch.float32)):
                heads = divisible_heads(8)
                kv_heads = divisible_heads(4)
                packed = torch.randn(1, 17, 3, heads, 32, device=device, dtype=dtype)
                query, key, value = packed.unbind(2)
                key, value = key[:, :, :kv_heads], value[:, :, :kv_heads]
                expected_qkv = tuple(
                    SeqAllToAll4D.apply(dist.group.WORLD, tensor, 2, 1, False) for tensor in (query, key, value)
                )
                attention_output = torch.randn(1, 17 * world, heads // world, 32, device=device, dtype=dtype)
                expected_reverse = SeqAllToAll4D.apply(dist.group.WORLD, attention_output, 1, 2, False)
                if index == 0 and dist.get_rank() == world - 1:
                    torch.npu.synchronize()
                    time.sleep(0.05)
                actual_qkv = ulysses_qkv_fwd_group(query, key, value, dist.group.WORLD.group_name, world)
                actual_reverse = ulysses_o_rev(attention_output, dist.group.WORLD.group_name, world)
                for actual, expected in zip((*actual_qkv, actual_reverse), (*expected_qkv, expected_reverse)):
                    assert_bitwise(actual, expected)
    finally:
        clear_a2a_permute_workspaces()
        dist.destroy_process_group()
