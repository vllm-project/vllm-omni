# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time strict Ulysses forward and reverse exchanges on four ROCm GPUs.

Run: torchrun --standalone --nproc-per-node=4 benchmarks/diffusion/bench_rocm_ulysses.py

Synthetic inputs reproduce MiniMax-H3's per-rank BF16 producer layout. A
contiguous attention output stands in for FMHA, which is excluded from timing.
"""

import argparse
import itertools
import json
import os
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention


def make_inputs(rows, device):
    shape = (1, rows, 56, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = q.new_empty((1, rows, 3, 56, 128))[:, :, 2].normal_()
    out = q.new_empty((1, 4 * rows, 14, 128)).normal_()
    return q, k, v, out


@torch.inference_mode()
def run(rows, rounds):
    rank = dist.get_rank()
    if dist.get_world_size() != 4 or torch.version.hip is None:
        raise ValueError("Run with four ROCm ranks")
    torch.manual_seed(42 + rank)
    q, k, v, out = make_inputs(rows, torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    group = SimpleNamespace(
        ulysses_group=dist.group.WORLD,
        ulysses_world_size=4,
        ulysses_rank=rank,
        ring_world_size=1,
        allgather_world_size=1,
    )
    ordinary = UlyssesParallelAttention(group, 2, 1, False)
    packed = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    strided = AttentionMetadata(extra={"accepts_strided_ulysses_qkv": True})
    arms = {"separate": (ordinary, None), "packed_dense": (packed, None), "packed_strided": (packed, strided)}

    def exchange(arm):
        strategy, metadata = arms[arm]
        q_recv, k_recv, v_recv, _, ctx = strategy.pre_attention(q, k, v, metadata)
        assert ctx.rocm_packed == (strategy is packed)
        return q_recv, k_recv, v_recv, strategy.post_attention(out, ctx)

    reference = exchange("separate")
    for arm in arms:
        for expected, actual in zip(reference, exchange(arm)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del reference, actual, expected
    orders = list(itertools.permutations(arms))
    timings = {arm: [] for arm in arms}
    # Alternate all six orders to distribute fabric/clock drift between arms.
    # One cycle warms up allocations and RCCL before collecting samples.
    for iteration in range(6 + rounds):
        for arm in orders[iteration % len(orders)]:
            dist.barrier()
            torch.accelerator.synchronize()
            start = time.perf_counter()
            result = exchange(arm)
            torch.accelerator.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            del result
            slowest_ms = q.new_tensor(elapsed_ms, dtype=torch.float64)
            dist.all_reduce(slowest_ms, op=dist.ReduceOp.MAX)
            if iteration >= 6:
                timings[arm].append(slowest_ms.item())

    if rank == 0:
        result = {
            arm: {"median": statistics.median(samples), "min": min(samples), "max": max(samples), "samples": samples}
            for arm, samples in timings.items()
        }
        print(json.dumps({"local_rows": rows, "row_strides": [x.stride(1) for x in (q, k, v)], "max_rank_ms": result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=9440)
    parser.add_argument("--rounds", type=int, default=18)
    args = parser.parse_args()
    if min(args.rounds, args.rows) < 1:
        parser.error("--rounds and --rows must be positive")
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        run(args.rows, args.rounds)
    finally:
        dist.destroy_process_group()
