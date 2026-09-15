# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark diffusion LoRA expand-and-accumulate implementations.

This compares the previous functional expand path with direct accumulation into
the selected output slice. It is a synthetic kernel microbenchmark and does not
load a diffusion model.

Example:
    python benchmarks/kernels/benchmark_diffusion_lora_expand.py \
        --m 1,16,128,512,4096,16384 \
        --in-dim 4096 --out-dim 4096 --rank 64 \
        --dtype bfloat16 --packed --warmup 30 \
        --small-iterations 500 --large-iterations 100
"""

import argparse
import math
import statistics
from collections.abc import Callable

import torch

MiB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="1,16,128,512,4096,16384")
    parser.add_argument("--in-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Accumulate into the second half of a packed output tensor.",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--small-iterations", type=int, default=500)
    parser.add_argument("--large-iterations", type=int, default=100)
    parser.add_argument("--large-m-threshold", type=int, default=1024)
    return parser.parse_args()


def percentile(samples: list[float], probability: float) -> float:
    ordered = sorted(samples)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure_latency_ms(operation: Callable[[], None], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    torch.accelerator.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def measure_extra_peak_mib(operation: Callable[[], None]) -> float:
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    allocated = torch.accelerator.memory_allocated()
    operation()
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated()
    return (peak - allocated) / MiB


def print_header(args: argparse.Namespace, dtype: torch.dtype) -> None:
    device_name = torch.cuda.get_device_name()
    print(f"device={device_name} torch={torch.__version__} dtype={dtype}")
    print(f"in_dim={args.in_dim} out_dim={args.out_dim} rank={args.rank} packed={args.packed} warmup={args.warmup}")
    print("M       iters  old_p10  old_p50  old_p90  new_p10  new_p50  new_p90  speedup  old_MiB  new_MiB  max_abs")


def benchmark_shape(m: int, args: argparse.Namespace, dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    output_width = args.out_dim * (2 if args.packed else 1)
    offset = args.out_dim if args.packed else 0
    output_slice = slice(offset, offset + args.out_dim)
    scale = 1 / math.sqrt(args.in_dim)

    x = torch.randn(m, args.in_dim, device=device, dtype=dtype)
    lora_a = torch.randn(args.rank, args.in_dim, device=device, dtype=dtype) * scale
    lora_b = torch.randn(args.out_dim, args.rank, device=device, dtype=dtype) / math.sqrt(args.rank)
    initial_output = torch.randn(m, output_width, device=device, dtype=dtype)

    def baseline(output: torch.Tensor) -> None:
        buffer = x @ lora_a.t()
        delta = buffer @ lora_b.t()
        selected = output[:, output_slice]
        selected[:] = selected + delta

    def optimized(output: torch.Tensor) -> None:
        buffer = x @ lora_a.t()
        selected = output[:, output_slice]
        torch.addmm(selected, buffer, lora_b.t(), out=selected)

    reference = initial_output.clone()
    candidate = initial_output.clone()
    baseline(reference)
    optimized(candidate)
    torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=1e-2)
    max_abs = (candidate - reference).abs().max().item()

    old_output = initial_output.clone()
    new_output = initial_output.clone()
    iterations = args.large_iterations if m >= args.large_m_threshold else args.small_iterations

    old_peak = measure_extra_peak_mib(lambda: baseline(old_output))
    new_peak = measure_extra_peak_mib(lambda: optimized(new_output))
    old_samples = measure_latency_ms(lambda: baseline(old_output), args.warmup, iterations)
    new_samples = measure_latency_ms(lambda: optimized(new_output), args.warmup, iterations)

    old_p50 = statistics.median(old_samples)
    new_p50 = statistics.median(new_samples)
    print(
        f"{m:<7} {iterations:<6} "
        f"{percentile(old_samples, 0.1):>8.5f} {old_p50:>8.5f} {percentile(old_samples, 0.9):>8.5f} "
        f"{percentile(new_samples, 0.1):>8.5f} {new_p50:>8.5f} {percentile(new_samples, 0.9):>8.5f} "
        f"{old_p50 / new_p50:>8.3f} {old_peak:>8.2f} {new_peak:>8.2f} {max_abs:>8.5f}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.warmup < 0 or args.small_iterations <= 0 or args.large_iterations <= 0:
        raise ValueError("warmup must be non-negative and iteration counts must be positive")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    print_header(args, dtype)
    with torch.inference_mode():
        for m in (int(value) for value in args.m.split(",")):
            if m <= 0:
                raise ValueError(f"M must be positive, got {m}")
            benchmark_shape(m, args, dtype)


if __name__ == "__main__":
    main()
