# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rotary_emb_torch
from vllm_omni.diffusion.models.qwen_image.fused_qk_norm_rope import (
    _qwen_image_fused_qk_norm_rope_triton,
    qwen_image_fused_qk_norm_rope,
    qwen_image_qk_norm_rope_fast_path_supported,
)


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
    parser.add_argument("--launch-config-sweep", action="store_true")
    parser.add_argument(
        "--launch-configs",
        default="2:2,2:3,4:3,4:4",
        help="Comma-separated Triton launch configs as num_warps:num_stages.",
    )
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _parse_launch_configs(value: str) -> list[tuple[int, int]]:
    configs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            warps_str, stages_str = item.split(":", maxsplit=1)
            num_warps = int(warps_str)
            num_stages = int(stages_str)
        except ValueError as exc:
            raise ValueError(f"Invalid launch config {item!r}; expected num_warps:num_stages") from exc
        if num_warps <= 0 or num_stages <= 0:
            raise ValueError(f"Launch config values must be positive, got {item!r}")
        configs.append((num_warps, num_stages))
    if not configs:
        raise ValueError("At least one launch config is required")
    return configs


def _make_inputs(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(2026)
    if args.packed_qkv_view:
        qkv_dim = (args.heads + args.kv_heads + args.kv_heads) * args.head_dim
        qkv = torch.randn(args.batch, args.seq_len, qkv_dim, device=device, dtype=dtype)
        q, k, _v = qkv.split(
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

    q_weight = torch.randn(args.head_dim, device=device, dtype=torch.float32)
    k_weight = torch.randn(args.head_dim, device=device, dtype=torch.float32)
    freqs = torch.randn(args.seq_len, args.head_dim // 2, device=device, dtype=torch.float32)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return q, k, q_weight, k_weight, cos, sin


def _eager(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    return (
        apply_rotary_emb_torch(q, cos, sin, interleaved=True),
        apply_rotary_emb_torch(k, cos, sin, interleaved=True),
    )


def _existing_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    rope: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    return rope(q, cos, sin), rope(k, cos, sin)


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
    q, k, q_weight, k_weight, cos, sin = _make_inputs(args, dtype, device)
    rope = RotaryEmbedding(is_neox_style=False)

    def eager():
        return _eager(q, k, q_weight, k_weight, cos, sin, eps)

    def fused():
        return qwen_image_fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps)

    def existing_rope():
        return _existing_rope(q, k, q_weight, k_weight, cos, sin, eps, rope)

    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "packed_qkv_view": args.packed_qkv_view,
        "fused_fast_path_supported": qwen_image_qk_norm_rope_fast_path_supported(q, cos),
    }

    eager_samples = _measure(eager, args.warmup, args.iters)
    fused_samples = _measure(fused, args.warmup, args.iters)
    existing_rope_samples = _measure(existing_rope, args.warmup, args.iters)
    ref_q, ref_k = eager()
    out_q, out_k = fused()
    torch.accelerator.synchronize()

    eager_stats = _stats(eager_samples)
    fused_stats = _stats(fused_samples)
    existing_rope_stats = _stats(existing_rope_samples)
    results["eager"] = eager_stats
    results["fused"] = fused_stats
    results["existing_rope"] = existing_rope_stats
    results["speedup_vs_eager"] = eager_stats["median_ms"] / fused_stats["median_ms"]
    results["speedup_vs_existing_rope"] = existing_rope_stats["median_ms"] / fused_stats["median_ms"]
    results["max_abs_q"] = (out_q.float() - ref_q.float()).abs().max().item()
    results["max_abs_k"] = (out_k.float() - ref_k.float()).abs().max().item()
    results["mean_abs_q"] = (out_q.float() - ref_q.float()).abs().mean().item()
    results["mean_abs_k"] = (out_k.float() - ref_k.float()).abs().mean().item()

    if args.include_compiled:
        compiled_eager = torch.compile(eager, dynamic=True, fullgraph=True)
        compiled_fused = torch.compile(fused, dynamic=True, fullgraph=True)
        compiled_existing_rope = torch.compile(existing_rope, dynamic=True, fullgraph=True)
        compiled_samples = _measure(compiled_eager, args.warmup, args.iters)
        compiled_fused_samples = _measure(compiled_fused, args.warmup, args.iters)
        compiled_existing_rope_samples = _measure(compiled_existing_rope, args.warmup, args.iters)
        compiled_stats = _stats(compiled_samples)
        compiled_fused_stats = _stats(compiled_fused_samples)
        compiled_existing_rope_stats = _stats(compiled_existing_rope_samples)
        results["compiled_eager"] = compiled_stats
        results["compiled_fused"] = compiled_fused_stats
        results["compiled_existing_rope"] = compiled_existing_rope_stats
        results["speedup_vs_compiled_eager"] = compiled_stats["median_ms"] / fused_stats["median_ms"]
        results["compiled_fused_speedup_vs_compiled_eager"] = (
            compiled_stats["median_ms"] / compiled_fused_stats["median_ms"]
        )
        results["compiled_fused_speedup_vs_compiled_existing_rope"] = (
            compiled_existing_rope_stats["median_ms"] / compiled_fused_stats["median_ms"]
        )

    if args.launch_config_sweep:
        launch_results: dict[str, dict[str, float]] = {}
        for num_warps, num_stages in _parse_launch_configs(args.launch_configs):
            name = f"warps{num_warps}_stages{num_stages}"

            def fused_with_launch_config(
                num_warps: int = num_warps,
                num_stages: int = num_stages,
            ):
                return _qwen_image_fused_qk_norm_rope_triton(
                    q,
                    k,
                    q_weight,
                    k_weight,
                    cos,
                    sin,
                    eps,
                    q.shape[-1],
                    cos.shape[-1] * 2,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            samples = _measure(fused_with_launch_config, args.warmup, args.iters)
            stats = _stats(samples)
            stats["speedup_vs_eager"] = eager_stats["median_ms"] / stats["median_ms"]
            stats["speedup_vs_existing_rope"] = existing_rope_stats["median_ms"] / stats["median_ms"]
            launch_results[name] = stats

        best_name, best_stats = min(launch_results.items(), key=lambda item: item[1]["median_ms"])
        results["launch_config_sweep"] = launch_results
        results["launch_config_best_by_median"] = {
            "name": best_name,
            **best_stats,
        }

    print(json.dumps(results, indent=2))
    if args.profile:
        print("\n# fused profile")
        print(_profile(fused))


if __name__ == "__main__":
    main()
