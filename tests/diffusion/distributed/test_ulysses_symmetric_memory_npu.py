# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run inside npu run with torchrun --nproc-per-node=<allocated> -m pytest -q <this file>.

Opt-in exchange benchmark (requires the production symmetric-memory dependencies)::

    npu run --gpus 2 --timeout 20m -- env VLLM_RUN_NPU_BENCHMARK=1 \
        NPU_SHMEM_SYMMETRIC_SIZE=4294967296 \
        torchrun --standalone --nproc-per-node=2 -m pytest -s -q \
        tests/diffusion/distributed/test_ulysses_symmetric_memory_npu.py -k benchmark

Use the same card count for npu run and torchrun; 2/4/8 ranks divide every shape.
Pytest -k can also select dtype/layout IDs, for example 'benchmark and bfloat16
and packed'. Times include fresh output allocation, staging, communication and
layout conversion, without projection/attention. Positive reduction means faster.
S is the local input length; reverse input is [B, WORLD*S, H/WORLD, D].
JUnit properties retain raw samples; when exporting, use a distinct XML path in
each torchrun worker to avoid overwrites, with -o junit_family=legacy.
"""

import json
import math
import os
import statistics
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.ulysses_symmetric_memory_npu import _tile_offsets

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A2"}, num_cards=2)]

_BENCHMARK_SHAPES = [
    (1, 1024, 16, 64),
    (2, 1024, 16, 64),
    (1, 4096, 32, 128),
    (2, 4096, 32, 128),
    (1, 16384, 32, 128),
    (2, 16384, 32, 128),
]


@triton.jit
def _probe_tile_offsets(indices, result, seq: tl.constexpr, reverse: tl.constexpr, interleave: tl.constexpr):
    # Runtime int32 inputs reproduce program_id/arange's initial width.
    rank = tl.load(indices)
    batch = tl.load(indices + 1)
    peer = tl.load(indices + 2)
    tile = tl.load(indices + 3)
    base, src, dst, mask = _tile_offsets(
        rank, 2, seq, 16384, reverse, 1, 32, batch, peer, tile, 32768 if interleave == 3 else 0, interleave
    )
    c = tl.arange(0, 32)[None, :]
    tl.store(result + c, base)
    tl.store(result + 32 + c, src)
    tl.store(result + 64 + c, dst)
    tl.store(result + 96 + c, mask.to(tl.int64))


@pytest.fixture(scope="module")
def npu_world():
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="Requires a scheduled multi-NPU torchrun job")
@pytest.mark.parametrize(
    "seq, reverse, interleave, coords, expected",
    [
        (65537, False, 1, (1, 0, 1, 65535), (16384, 2147450880, 2147483648)),
        (65537, False, 1, (1, 0, 1, 65536), (16384, 2147483648, 2147500032)),
        (65537, False, 1, (1, 1, 1, 0), (2147532800, 0, 3221274624)),
        (32769, False, 3, (1, 0, 1, 21846), (49152, 2147549184, 894812160)),
        (32769, False, 3, (1, 1, 1, 0), (3221372928, 0, 1610661888)),
        (65537, True, 1, (1, 0, 1, 65536), (1073758208, 1073741824, 2147500032)),
        (65537, True, 1, (1, 1, 1, 0), (3221274624, 0, 2147532800)),
    ],
)
def test_production_tile_offsets_above_int32(npu_world, seq, reverse, interleave, coords, expected):
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    indices = torch.tensor(coords, device=device, dtype=torch.int32)
    result = torch.empty((4, 32), device=device, dtype=torch.int64)
    # Exercise the production address calculation without allocating multi-GiB
    # buffers or dereferencing the old kernel's wrapped negative addresses.
    _probe_tile_offsets[(1,)](indices, result, seq, reverse, interleave)
    actual = result.cpu()
    base, src, dst = expected
    assert torch.equal(actual[0], torch.full((32,), base, dtype=torch.int64))
    assert torch.equal(actual[1], torch.arange(src, src + 32))
    assert torch.equal(actual[2], torch.arange(dst, dst + 32))
    assert actual[3].eq(1).all(), "The overflowing row is valid, so masking cannot prevent the invalid access"


@pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="Requires a scheduled multi-NPU torchrun job")
def test_production_symmetric_memory_matches_hccl(npu_world):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.distributed.a2a_permute import (
        clear_a2a_permute_workspaces,
        ulysses_o_rev,
        ulysses_qkv_fwd_group,
    )
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

    local_rank = int(os.environ["LOCAL_RANK"])
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
                (1, 513, divisible_heads(30), divisible_heads(30), 128),  # Z-Image's non-power-of-two heads.
                (2, 513, divisible_heads(24), divisible_heads(24), 128),  # Qwen-Image, including a tail tile.
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

            other_stream = torch.npu.Stream(device=device)
            with torch.npu.stream(other_stream):
                with pytest.raises(RuntimeError, match="single NPU stream"):
                    ulysses_qkv_fwd_group(query, key, value, dist.group.WORLD.group_name, world)
            try:
                torch.npu.set_device((local_rank + 1) % world)
                with pytest.raises(RuntimeError, match="original NPU device"):
                    ulysses_qkv_fwd_group(query, key, value, dist.group.WORLD.group_name, world)
            finally:
                torch.npu.set_device(local_rank)
    finally:
        clear_a2a_permute_workspaces()


@pytest.fixture(scope="module")
def symmetric_memory_benchmark_workspace(npu_world):
    from vllm_omni.diffusion.distributed.ulysses_symmetric_memory_npu import NPUUlyssesSymmetricMemory

    workspace = None
    try:
        world = dist.get_world_size()
        if world < 2 or any(heads % world for _, _, heads, _ in _BENCHMARK_SHAPES):
            pytest.skip("WORLD size must divide every benchmark head count and be greater than one")
        # Reserve for the largest grouped QKV case, including FP32.
        workspace = NPUUlyssesSymmetricMemory(3 * max(math.prod(shape) for shape in _BENCHMARK_SHAPES) * 4)
        if dist.get_rank() == 0:
            print(
                "\nExchange latency: ms, mean ± sample SD of 6 segment means per backend; "
                "5 warmups, 3 ABBA blocks, 10 exchanges per segment, slowest rank. "
                "Profiler disabled; positive reduction means faster.\n"
                "| WORLD | dtype | layout | local [B,S,H,D] | operation | HCCL ms | symmetric memory ms | reduction |\n"
                "| ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
                flush=True,
            )
        yield workspace
    finally:
        if workspace is not None:
            workspace.close()


def _benchmark_exchange_pair(hccl, symmetric_memory, device):
    operations = (hccl, symmetric_memory)
    for operation in operations:
        for _ in range(5):
            operation()
        torch.npu.synchronize()
    samples: tuple[list[float], list[float]] = ([], [])
    for _ in range(3):
        for index in (0, 1, 1, 0):
            dist.barrier()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(10):
                operation_result = operations[index]()
                # Both backends allocate owned outputs; release them equally.
                del operation_result
            torch.npu.synchronize()
            elapsed = torch.tensor((time.perf_counter() - started) * 1000 / 10, device=device, dtype=torch.float32)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[index].append(elapsed.item())
    return samples


@pytest.mark.benchmark
@pytest.mark.skipif(
    "LOCAL_RANK" not in os.environ or os.environ.get("VLLM_RUN_NPU_BENCHMARK") != "1",
    reason="Opt-in: VLLM_RUN_NPU_BENCHMARK=1 in a scheduled multi-NPU torchrun job",
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=lambda dtype: str(dtype).split(".")[-1]
)
@pytest.mark.parametrize("layout", ["contiguous", "packed"])
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES, ids=lambda shape: "x".join(map(str, shape)))
def test_symmetric_memory_benchmark(shape, dtype, layout, symmetric_memory_benchmark_workspace, record_property):
    from vllm_omni.diffusion.distributed.comm import all_to_all_4D

    workspace = symmetric_memory_benchmark_workspace
    world, rank = dist.get_world_size(), dist.get_rank()
    device = torch.device("npu", torch.npu.current_device())
    batch, seq, heads, dim = shape
    with torch.inference_mode():
        torch.manual_seed(6340 + rank)
        packed = torch.randn(batch, seq, 3, heads, dim, device=device, dtype=dtype)
        qkv = tuple(tensor.contiguous() if layout == "contiguous" else tensor for tensor in packed.unbind(2))
        attention_output = torch.randn(batch, seq * world, heads // world, dim, device=device, dtype=dtype)

        def hccl_qkv():
            return tuple(all_to_all_4D(tensor, 2, 1) for tensor in qkv)

        def fused_qkv():
            return workspace.qkv(*qkv)

        def hccl_o():
            return all_to_all_4D(attention_output, 1, 2)

        def fused_o():
            return workspace(attention_output, reverse=True)

        operations = {
            "q_forward": (lambda: all_to_all_4D(qkv[0], 2, 1), lambda: workspace(qkv[0])),
            "qkv_forward": (hccl_qkv, fused_qkv),
            "o_reverse": (hccl_o, fused_o),
            "qkv_and_o": (lambda: (*hccl_qkv(), hccl_o()), lambda: (*fused_qkv(), fused_o())),
        }
        for name, (hccl, fused) in operations.items():
            expected, actual = hccl(), fused()
            expected = (expected,) if isinstance(expected, torch.Tensor) else expected
            actual = (actual,) if isinstance(actual, torch.Tensor) else actual
            equal = all(torch.equal(a.view(torch.uint8), e.view(torch.uint8)) for a, e in zip(actual, expected))
            failed = torch.tensor(int(not equal), device=device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            assert failed.item() == 0, f"{name}: symmetric memory/HCCL outputs differ"
            del expected, actual
            hccl_samples, fused_samples = _benchmark_exchange_pair(hccl, fused, device)
            hccl_ms, fused_ms = statistics.mean(hccl_samples), statistics.mean(fused_samples)
            reduction = (hccl_ms - fused_ms) / hccl_ms * 100
            if rank == 0:
                print(
                    f"| {world} | {dtype} | {layout} | {list(shape)} | {name} | "
                    f"{hccl_ms:.4f} ± {statistics.stdev(hccl_samples):.4f} | "
                    f"{fused_ms:.4f} ± {statistics.stdev(fused_samples):.4f} | {reduction:+.1f}% |",
                    flush=True,
                )
                record_property(
                    name,
                    json.dumps(
                        dict(
                            world=world,
                            shape=shape,
                            dtype=str(dtype),
                            layout=layout,
                            qkv_stride=qkv[0].stride(),
                            soc=torch.npu.get_device_name(device),
                            torch_version=str(torch.__version__),
                            hccl_segment_ms=hccl_samples,
                            symmetric_memory_segment_ms=fused_samples,
                            reduction_percent=reduction,
                        )
                    ),
                )
