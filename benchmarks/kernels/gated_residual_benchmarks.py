# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the shared diffusion gated-residual operator.

Example:
    python benchmarks/kernels/gated_residual_benchmarks.py \
        --batch-size 2 --tokens 32760 --hidden-size 5120 --gate-layout batch
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from vllm_omni.diffusion.layers.gated_residual import gated_residual


def _measure_ms(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def _gate_shape(layout: str, batch_size: int, tokens: int, hidden_size: int) -> tuple[int, ...]:
    if layout == "global":
        return (hidden_size,)
    if layout == "batch":
        return (batch_size, 1, hidden_size)
    return (batch_size, tokens, hidden_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32760)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--gate-layout", choices=("global", "batch", "token"), default="batch")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    dtype = getattr(torch, args.dtype)
    shape = (args.batch_size, args.tokens, args.hidden_size)
    gate_shape = _gate_shape(args.gate_layout, *shape)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    branch = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn(gate_shape, device="cuda", dtype=dtype)

    def eager() -> torch.Tensor:
        return residual + branch * gate

    def fused() -> torch.Tensor:
        return gated_residual(residual, branch, gate)

    torch.testing.assert_close(fused(), eager(), rtol=0, atol=0)

    for name, function in (("eager", eager), ("fused", fused)):
        samples = _measure_ms(
            function,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        mean = statistics.mean(samples)
        stddev = statistics.pstdev(samples)
        print(f"{name:>5}: {mean:.4f} ms ± {stddev:.4f} ms")


if __name__ == "__main__":
    main()
