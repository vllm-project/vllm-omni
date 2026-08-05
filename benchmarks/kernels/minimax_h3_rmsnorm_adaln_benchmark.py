# SPDX-License-Identifier: Apache-2.0
"""Benchmark MiniMax-H3 RMSNorm plus indexed AdaLN fusion."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from vllm.triton_utils import triton

from vllm_omni.diffusion.models.minimax_h3.fused_ops import (
    fused_rmsnorm_indexed_scale_shift_bf16,
    indexed_scale_shift_bf16_,
)


def _baseline(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = F.rms_norm(x, (x.shape[-1],), weight, eps)
    assert indexed_scale_shift_bf16_(output, shift, scale, indices)
    return output


def _fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = fused_rmsnorm_indexed_scale_shift_bf16(x, weight, shift, scale, indices, eps)
    assert output is not None
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 7, 31, 128, 257, 512, 1024, 2048, 8192])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(20260806)
    device = torch.device("cuda")
    hidden = 5376
    num_modulations = 16
    eps = 1e-5

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")
    print(f"Warmup: {args.warmup}; iterations: {args.iterations}")
    print()
    print("| Tokens | Hidden | Baseline (ms) | Fused (ms) | Speedup | Max abs error |")
    print("|---:|---:|---:|---:|---:|---:|")

    for tokens in args.tokens:
        try:
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
            weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
            shift = torch.randn(num_modulations, hidden, device=device, dtype=torch.bfloat16)
            scale = torch.randn(num_modulations, hidden, device=device, dtype=torch.bfloat16)
            indices = torch.randint(num_modulations, (tokens,), device=device, dtype=torch.int64)

            expected = _baseline(x, weight, shift, scale, indices, eps)
            actual = _fused(x, weight, shift, scale, indices, eps)
            max_abs = (actual.float() - expected.float()).abs().max().item()
            torch.accelerator.synchronize()
            baseline_ms = triton.testing.do_bench(
                lambda: _baseline(x, weight, shift, scale, indices, eps),
                warmup=args.warmup,
                rep=args.iterations,
            )
            fused_ms = triton.testing.do_bench(
                lambda: _fused(x, weight, shift, scale, indices, eps),
                warmup=args.warmup,
                rep=args.iterations,
            )
            print(
                f"| {tokens} | {hidden} | {baseline_ms:.4f} | {fused_ms:.4f} | "
                f"{baseline_ms / fused_ms:.2f}x | {max_abs:.8f} |"
            )
        except torch.OutOfMemoryError:
            print(f"| {tokens} | {hidden} | skipped (OOM) | skipped (OOM) | - | - |")


if __name__ == "__main__":
    main()
