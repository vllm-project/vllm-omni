# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Microbenchmark pageable versus bounded pinned diffusion weight copies."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from vllm_omni.diffusion.model_loader.pinned_staging import (
    pinned_staging_weights_iterator,
    release_pinned_staging_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _synchronize() -> None:
    torch.accelerator.synchronize()


def _direct_samples(
    source: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
) -> list[float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        target.copy_(source)
        _synchronize()
        samples.append(time.perf_counter() - started)
    return samples


def _staged_samples(
    source: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    capacity_bytes: int,
) -> list[float]:
    staged = pinned_staging_weights_iterator(
        iter((str(index), source) for index in range(iterations)),
        capacity_bytes=capacity_bytes,
        min_bytes=1,
    )
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        _, tensor = next(staged)
        target.copy_(tensor)
        _synchronize()
        samples.append(time.perf_counter() - started)
    staged.close()
    release_pinned_staging_cache()
    return samples


def _summary(name: str, samples: list[float], nbytes: int) -> None:
    mean = statistics.mean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f"{name}: {mean * 1000:.3f} ± {deviation * 1000:.3f} ms, {nbytes / mean / (1 << 30):.2f} GiB/s")


def main() -> None:
    args = parse_args()
    if args.size_mib <= 0 or args.warmups < 0 or args.iterations <= 0:
        raise ValueError("size-mib and iterations must be positive; warmups must be non-negative")

    dtype = getattr(torch, args.dtype)
    nbytes = args.size_mib << 20
    numel = nbytes // dtype.itemsize
    source = torch.arange(numel, dtype=dtype)
    target = torch.empty(numel, dtype=dtype, device=args.device)

    _direct_samples(source, target, args.warmups)
    _staged_samples(source, target, args.warmups, nbytes)
    direct = _direct_samples(source, target, args.iterations)
    staged = _staged_samples(source, target, args.iterations, nbytes)

    assert torch.equal(target.cpu(), source)
    _summary("pageable", direct, nbytes)
    _summary("pinned", staged, nbytes)
    print(f"speedup: {statistics.mean(direct) / statistics.mean(staged):.3f}x")


if __name__ == "__main__":
    main()
