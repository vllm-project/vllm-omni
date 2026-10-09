# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark Qwen-Image Q/K RMSNorm + interleaved RoPE.

Example:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_qwen_image_qk_norm_rope.py \
      --seq-len 4096 --warmup 20 --iters 100 --include-compiled --profile
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_qk_norm_rope,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--packed-qkv-view", action="store_true")
    parser.add_argument("--include-compiled", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _make_inputs(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(2026)
    if args.packed_qkv_view:
        qkv_dim = (args.heads + args.kv_heads + args.kv_heads) * args.head_dim
        qkv = torch.randn(args.batch, args.seq_len, qkv_dim, device=device, dtype=dtype)
        q, k, _ = qkv.split(
            [
                args.heads * args.head_dim,
                args.kv_heads * args.head_dim,
                args.kv_heads * args.head_dim,
            ],
            dim=-1,
        )
        q = q.unflatten(-1, (args.heads, args.head_dim))
        k = k.unflatten(-1, (args.kv_heads, args.head_dim))
    else:
        q = torch.randn(args.batch, args.seq_len, args.heads, args.head_dim, device=device, dtype=dtype)
        k = torch.randn(args.batch, args.seq_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)

    q_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
    angles = torch.randn(args.seq_len, args.head_dim // 2, device=device, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return q, k, q_weight, k_weight, freqs


def _native(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    rope = RotaryEmbedding(is_neox_style=False)
    cos = freqs.real.to(q.dtype)
    sin = freqs.imag.to(q.dtype)
    return rope(q, cos, sin), rope(k, cos, sin)


def _fused(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    rope_table = torch.cat((freqs.real, freqs.imag), dim=-1)
    rope_table = rope_table.unsqueeze(0).expand(batch, -1, -1).reshape(batch * seq_len, head_dim)
    out_q, out_k = fused_qk_norm_rope(
        q.reshape(batch * seq_len, num_heads, head_dim),
        k.reshape(batch * seq_len, num_kv_heads, head_dim),
        q_weight,
        k_weight,
        rope_table,
        eps,
        interleaved=True,
    )
    return (
        out_q.reshape(batch, seq_len, num_heads, head_dim),
        out_k.reshape(batch, seq_len, num_kv_heads, head_dim),
    )


def _measure(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _stats(samples: list[float]) -> dict[str, float]:
    samples = sorted(samples)
    return {
        "median_ms": samples[len(samples) // 2],
        "p90_ms": samples[int(len(samples) * 0.9)],
        "mean_ms": sum(samples) / len(samples),
    }


def _profile(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> str:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    eps = 1e-6
    q, k, q_weight, k_weight, freqs = _make_inputs(args, dtype, device)

    def native():
        return _native(q, k, q_weight, k_weight, freqs, eps)

    def fused():
        return _fused(q, k, q_weight, k_weight, freqs, eps)

    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "packed_qkv_view": args.packed_qkv_view,
        "fused_fast_path_supported": _fused_cuda_supported(
            q.reshape(args.batch * args.seq_len, args.heads, args.head_dim),
            k.reshape(args.batch * args.seq_len, args.kv_heads, args.head_dim),
            args.head_dim,
            args.head_dim,
            interleaved=True,
        ),
    }

    native_samples = _measure(native, args.warmup, args.iters)
    fused_samples = _measure(fused, args.warmup, args.iters)
    expected_q, expected_k = native()
    actual_q, actual_k = fused()
    torch.accelerator.synchronize()

    native_stats = _stats(native_samples)
    fused_stats = _stats(fused_samples)
    results["native"] = native_stats
    results["fused"] = fused_stats
    results["speedup_vs_native"] = native_stats["median_ms"] / fused_stats["median_ms"]
    results["max_abs_q"] = (actual_q.float() - expected_q.float()).abs().max().item()
    results["max_abs_k"] = (actual_k.float() - expected_k.float()).abs().max().item()
    results["mean_abs_q"] = (actual_q.float() - expected_q.float()).abs().mean().item()
    results["mean_abs_k"] = (actual_k.float() - expected_k.float()).abs().mean().item()

    if args.include_compiled:
        compiled_native = torch.compile(native, dynamic=True, fullgraph=True)
        compiled_fused = torch.compile(fused, dynamic=True, fullgraph=True)
        compiled_native_stats = _stats(_measure(compiled_native, args.warmup, args.iters))
        compiled_fused_stats = _stats(_measure(compiled_fused, args.warmup, args.iters))
        results["compiled_native"] = compiled_native_stats
        results["compiled_fused"] = compiled_fused_stats
        results["compiled_fused_speedup_vs_compiled_native"] = (
            compiled_native_stats["median_ms"] / compiled_fused_stats["median_ms"]
        )

    print(json.dumps(results, indent=2))
    if args.profile:
        print("\n# fused profile")
        print(_profile(fused))


if __name__ == "__main__":
    main()
