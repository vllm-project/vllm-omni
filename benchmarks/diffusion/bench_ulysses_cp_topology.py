# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark the dense compute-vs-communication tradeoff behind the composed
Ulysses x AllGather-KV topology, with a per-case profiler breakdown.

Compares two 4-card sequence-parallel layouts for the SAME synthetic attention
shape, sweeping the sequence length:

  * ``usp4+uaa``   : pure Ulysses on 4 cards. With ``heads`` not divisible by 4,
                     advanced_uaa pads the head count (a real +14% attention
                     work when heads=14 -> padded 16), but communication is only
                     the Ulysses all-to-all.
  * ``usp2+cp2``   : Ulysses(2) x AllGather-KV(2). No head padding (heads % 2 == 0
                     for the default 14), but K/V are all-gathered over the
                     orthogonal group (extra communication).

Why ``heads=14`` is the faithful 4-card analogue of MiniMax-H3's 56 heads on
16 cards (usp16+uaa vs usp8+cp2): both have heads/usp = 3.5, so the UAA pad is
56->64 (+14.3%) and 14->16 (+14.3%) respectively, and 56 % 8 == 0 mirrors
14 % 2 == 0 on the composed side. Running 56 heads on 4 cards would silently
disable the UAA padding (56 % 4 == 0) and stop measuring the effect of interest.

For each (topology, seq) case the script reports three numbers from a single
``torch.profiler`` CUDA trace plus a wall-clock median:

  * ``total``  : median wall-clock ms of ``Attention.forward`` (iters runs).
  * ``comm``   : summed CUDA device time of NCCL collectives (all-to-all /
                 all-gather), i.e. the communication cost.
  * ``kernel`` : summed CUDA device time of everything else, i.e. the
                 attention kernel (flash/SDPA).

Run on a box with >= 4 GPUs:

    python benchmarks/diffusion/bench_ulysses_cp_topology.py \
        --heads 14 --head-dim 128 --seqs 16384,32768,65536 --warmup 2 --iters 5
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform

# One entry per topology: (label, ulysses_degree, allgather_degree, ulysses_mode).
_TOPOLOGIES = [
    ("usp4+uaa", 4, 1, "advanced_uaa"),
    ("usp2+cp2", 2, 2, "strict"),
]

# NCCL collective kernels are prefixed with "nccl" in the CUDA trace; that is
# what we count as communication. Everything else is the attention kernel.
_COMM_KEYWORDS = ("nccl", "all_to_all", "all_gather")


def _profile_once(attn: Attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[float, float, list]:
    """Run one forward under a CUDA profiler.

    Returns ``(comm_ms, kernel_ms, comm_ops)`` where ``comm_ops`` lists every
    NCCL collective with its summed device time. Every rank must call this
    (the forward contains collectives); only rank 0 uses the result.
    """
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        attn(q, k, v)
        current_omni_platform.synchronize()

    comm_us = 0.0
    kernel_us = 0.0
    comm_ops: dict[str, float] = {}
    for evt in prof.key_averages():
        # torch >= 2.x names this device_time_total; older builds used cuda_time_total.
        t = getattr(evt, "device_time_total", 0.0) or getattr(evt, "cuda_time_total", 0.0)
        if t <= 0:
            continue
        name = evt.key
        if any(k in name.lower() for k in _COMM_KEYWORDS):
            comm_us += t
            comm_ops[name] = comm_ops.get(name, 0.0) + t / 1000.0
        else:
            kernel_us += t
    return comm_us / 1000.0, kernel_us / 1000.0, sorted(comm_ops.items(), key=lambda x: -x[1])


def _worker(
    rank: int,
    world_size: int,
    topology: tuple,
    heads: int,
    head_dim: int,
    seqs: list[int],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    result_file: str,
) -> None:
    label, ulysses_degree, allgather_degree, ulysses_mode = topology

    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )
    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    torch.set_default_dtype(dtype)

    init_distributed_environment()

    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        allgather_degree=allgather_degree,
        ulysses_mode=ulysses_mode,
        cfg_parallel_size=1,
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        model="benchmark",
        dtype=dtype,
        parallel_config=parallel_config,
        # Force one kernel for both topologies so only the parallel path differs.
        diffusion_attention_backend="TORCH_SDPA",
    )
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        allgather_degree=allgather_degree,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    # Build the attention layer and run the whole sweep inside the forward
    # context: get_ulysses_mode() reads ulysses_mode from it at forward time,
    # so leaving the context would silently drop advanced_uaa back to strict.
    with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
        attn = Attention(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=1.0 / (head_dim**0.5),
            num_kv_heads=None,  # MHA, matching MiniMax-H3 (no num_key_value_heads)
            scatter_idx=2,
            gather_idx=1,
            use_sync=False,
        )

        medians: dict[int, float] = {}
        profiles: dict[int, dict] = {}
        for seq in seqs:
            assert seq % world_size == 0, f"seq {seq} must be divisible by world_size {world_size}"
            s_local = seq // world_size
            q = torch.randn(1, s_local, heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, s_local, heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, s_local, heads, head_dim, device=device, dtype=dtype)

            def _run() -> None:
                # NOTE: every rank must call forward (collectives); only rank 0 clocks.
                attn(q, k, v)

            for _ in range(warmup):
                _run()
            current_omni_platform.synchronize()

            times = []
            for _ in range(iters):
                current_omni_platform.synchronize()
                t0 = time.perf_counter()
                _run()
                current_omni_platform.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            medians[seq] = times[len(times) // 2]

            # One profiled forward per rank. A straggler rank inflates every
            # other rank's measured collective time, so take the rank with the
            # smallest comm time as the true communication cost.
            comm_ms, kernel_ms, comm_ops = _profile_once(attn, q, k, v)
            all_splits = [None] * world_size
            dist.all_gather_object(all_splits, (comm_ms, kernel_ms, comm_ops))
            if rank == 0:
                best_comm, best_kernel, best_ops = min(all_splits, key=lambda x: x[0])
                profiles[seq] = {"comm": best_comm, "kernel": best_kernel, "comm_ops": best_ops}

    destroy_distributed_env()
    if rank == 0:
        with open(result_file, "wb") as f:
            pickle.dump((label, medians, profiles), f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seqs", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    seqs = [int(x) for x in args.seqs.split(",")]
    dtype = getattr(torch, args.dtype)
    world_size = 4
    device_count = current_omni_platform.get_device_count()
    if device_count < world_size:
        raise SystemExit(f"Need {world_size} GPUs, found {device_count}.")

    # Sanity: if heads % 4 == 0, usp4+uaa degrades to strict (no padding) and
    # the comparison stops measuring the effect of interest.
    if args.heads % 4 == 0:
        print(
            f"[warn] heads={args.heads} is divisible by 4, so usp4+uaa will NOT pad "
            f"(use a heads value with heads % 4 != 0 and heads % 2 == 0, e.g. 14, "
            f"to reproduce the 56-head / usp16+uaa tradeoff faithfully)."
        )

    results: dict[str, tuple[dict[int, float], dict[int, dict]]] = {}
    for topology in _TOPOLOGIES:
        label = topology[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            result_file = f.name
        try:
            mp.spawn(
                _worker,
                args=(
                    world_size,
                    topology,
                    args.heads,
                    args.head_dim,
                    seqs,
                    dtype,
                    args.warmup,
                    args.iters,
                    result_file,
                ),
                nprocs=world_size,
                join=True,
            )
            with open(result_file, "rb") as f:
                got_label, medians, profiles = pickle.load(f)
            assert got_label == label, (got_label, label)
            results[label] = (medians, profiles)
        finally:
            if os.path.exists(result_file):
                os.remove(result_file)

    print(f"\nheads={args.heads} head_dim={args.head_dim} dtype={args.dtype} warmup={args.warmup} iters={args.iters}")
    for seq in seqs:
        print(f"\n=== seq={seq} ===")
        for label, *_ in _TOPOLOGIES:
            medians, profiles = results[label]
            total = medians[seq]
            p = profiles[seq]
            print(f"  {label:<10} total={total:>8.3f}ms  comm={p['comm']:>8.3f}ms  kernel={p['kernel']:>8.3f}ms")
            for op_name, op_ms in p["comm_ops"]:
                print(f"      {op_name:<50} {op_ms:>8.3f}ms")


if __name__ == "__main__":
    main()
