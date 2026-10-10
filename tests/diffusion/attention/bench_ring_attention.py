# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Measure a complete Ring attention call on each rank.

Run this same script against both source checkouts, without a profiler:
    torchrun --standalone --nproc-per-node=8 \
        tests/diffusion/attention/bench_ring_attention.py \
        --sequence=9450 --heads=40 --head-dim=128 --backend=FA3

Sequence length is local to each rank. Samples use the slowest rank's CUDA
event interval, including host submission gaps, rather than kernel-time sums.
This is an operator benchmark; it does not measure model or serving latency.
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=positive_int, default=4095)
    parser.add_argument("--heads", type=positive_int, default=40)
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--backend", choices=["FA", "FA3"], default="FA3")
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--rounds", type=positive_int, default=10)
    parser.add_argument("--calls-per-round", type=positive_int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2 to report sample standard deviation")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    # Import the framework after selecting this rank's device.
    from vllm_omni.diffusion.attention.backends import ring_flash_attn
    from vllm_omni.diffusion.attention.backends.ring import ring_utils
    from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType

    dist.init_process_group("nccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", local_rank)
        torch.manual_seed(42 + rank)
        shape = (1, args.sequence, args.heads, args.head_dim)
        q, k, v = [torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(3)]

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return ring_flash_attn.ring_flash_attn_forward(
                dist.group.WORLD,
                q,
                k,
                v,
                softmax_scale=args.head_dim**-0.5,
                causal=False,
                attn_type=AttnType[args.backend],
            )

        for _ in range(args.warmup):
            run()
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        retry_start = torch.accelerator.memory_stats()["num_alloc_retries"]
        samples = []
        for _ in range(args.rounds):
            dist.barrier()
            torch.accelerator.synchronize()
            start, end = torch.Event(device="cuda", enable_timing=True), torch.Event(device="cuda", enable_timing=True)
            start.record()
            for _ in range(args.calls_per_round):
                run()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end) / args.calls_per_round, device=device, dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())

        stats = torch.accelerator.memory_stats()
        memory = torch.tensor(
            [
                stats["allocated_bytes.all.peak"],
                stats["reserved_bytes.all.peak"],
                stats["num_alloc_retries"] - retry_start,
            ],
            device=device,
            dtype=torch.int64,
        )
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            source_paths = [Path(inspect.getfile(module)) for module in (ring_flash_attn, ring_utils)]
            fused_path = source_paths[1].with_name("fused_merge.py")
            if fused_path.exists():
                source_paths.append(fused_path)
            allocated, reserved, retries = memory.tolist()
            result = {
                "scope": "unprofiled complete Ring call; MAX rank time per round",
                "settings": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "world_size": world_size,
                "shape_per_rank": shape,
                "dtype": "bfloat16",
                "gpu": torch.cuda.get_device_name(local_rank),
                "gpu_memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                "samples_ms": samples,
                "mean_ms": statistics.mean(samples),
                "sample_stddev_ms": statistics.stdev(samples),
                "peak_allocated_bytes_max_rank": allocated,
                "peak_reserved_bytes_max_rank": reserved,
                "allocator_retries_max_rank": retries,
            }
            text = json.dumps(result, indent=2) + "\n"
            print(text, flush=True)
            if args.output is not None:
                with args.output.open("x") as stream:
                    stream.write(text)
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
