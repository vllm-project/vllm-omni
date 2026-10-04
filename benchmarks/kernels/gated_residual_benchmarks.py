# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Microbenchmark the shared diffusion gated-residual operator.

Example:
    python -m benchmarks.kernels.gated_residual_benchmarks \
        --batch-size 2 --tokens 32760 --hidden-size 5120 --gate-layout batch
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from importlib.metadata import version

import torch
from vllm.platforms import current_platform

from vllm_omni.diffusion.layers.ops import gated_residual


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


def _peak_extra_mib(function: Callable[[], torch.Tensor]) -> float:
    """Measure peak tensor allocation above the already allocated inputs."""
    torch.accelerator.synchronize()
    baseline = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    output = function()
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - baseline
    del output
    return peak / (1024**2)


def _gate_shape(layout: str, batch_size: int, tokens: int, hidden_size: int) -> tuple[int, ...]:
    if layout == "global":
        return (hidden_size,)
    if layout == "batch":
        return (batch_size, 1, hidden_size)
    return (batch_size, tokens, hidden_size)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32760)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--gate-layout", choices=("global", "batch", "token"), default="batch")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    shape = (args.batch_size, args.tokens, args.hidden_size)
    gate_shape = _gate_shape(args.gate_layout, *shape)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    branch = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn(gate_shape, device="cuda", dtype=dtype)

    def eager() -> torch.Tensor:
        return residual + branch * gate

    def shared() -> torch.Tensor:
        return gated_residual(residual, branch, gate)

    torch.testing.assert_close(shared(), eager(), rtol=0, atol=0)
    print("Correctness: exact match with eager PyTorch")
    print(f"GPU: {current_platform.get_device_name()}")
    print(
        f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; "
        f"Triton: {version('triton')}; vLLM: {version('vllm')}"
    )
    print(
        f"Shape: {shape}; gate: {gate_shape} ({args.gate_layout}); dtype: {args.dtype}; seed: {args.seed}; "
        f"warmup: {args.warmup}; iterations: {args.iterations}; repeats: {args.repeats}"
    )

    for name, function in (("eager", eager), ("gated_residual", shared)):
        samples = _measure_ms(
            function,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        mean = statistics.mean(samples)
        stddev = statistics.pstdev(samples)
        peak_mib = _peak_extra_mib(function)
        print(f"{name:>14}: {mean:.4f} ms ± {stddev:.4f} ms; peak extra allocation: {peak_mib:.2f} MiB")


if __name__ == "__main__":
    main()
