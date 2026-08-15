# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for diffusion pinned-staging weight loads.

Example:
    python benchmarks/diffusion/bench_weight_load_staging.py --size-mb 512 --repeats 5
"""

import argparse
import statistics
import time
from collections.abc import Callable

import torch


def _measure(operation: Callable[[], None], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        torch.accelerator.synchronize()
        started = time.perf_counter()
        operation()
        torch.accelerator.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _print_result(name: str, seconds: float, transferred_bytes: int) -> None:
    gib = transferred_bytes / (1 << 30)
    print(f"{name:<30} {seconds * 1000:9.2f} ms  {gib / seconds:8.2f} GiB/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=int, default=256, help="FP32 source tensor size in MiB")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.size_mb <= 0 or args.repeats <= 0:
        parser.error("--size-mb and --repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    source_bytes = args.size_mb << 20
    numel = source_bytes // torch.float32.itemsize
    pageable_fp32 = torch.empty(numel, dtype=torch.float32)
    pinned_fp32 = torch.empty(numel, dtype=torch.float32, pin_memory=True)
    pinned_bf16 = torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)
    gpu_fp32 = torch.empty(numel, dtype=torch.float32, device="cuda")
    gpu_bf16 = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
    output_bytes = pinned_bf16.numel() * pinned_bf16.element_size()

    pinned_fp32.copy_(pageable_fp32)
    pinned_bf16.copy_(pageable_fp32)

    cases = [
        ("pageable fp32 H2D", lambda: gpu_fp32.copy_(pageable_fp32), source_bytes),
        ("pinned fp32 H2D", lambda: gpu_fp32.copy_(pinned_fp32, non_blocking=True), source_bytes),
        (
            "pinned converting H2D",
            lambda: gpu_bf16.copy_(pinned_fp32, non_blocking=True),
            output_bytes,
        ),
        (
            "fused CPU cast + pinned H2D",
            lambda: (pinned_bf16.copy_(pageable_fp32), gpu_bf16.copy_(pinned_bf16, non_blocking=True)),
            output_bytes,
        ),
    ]

    print(f"CUDA device: {torch.cuda.get_device_name()} | FP32 source: {args.size_mb} MiB")
    for name, operation, transferred_bytes in cases:
        operation()
        seconds = _measure(operation, args.repeats)
        _print_result(name, seconds, transferred_bytes)


if __name__ == "__main__":
    main()
