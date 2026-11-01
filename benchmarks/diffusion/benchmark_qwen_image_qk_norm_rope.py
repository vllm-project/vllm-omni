# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark Qwen-Image Q/K RMSNorm + interleaved RoPE, with and without the fused kernels.

``--mode per-stream`` (default) times one stream: the eager ``F.rms_norm`` + ``RotaryEmbedding``
chain against ``fused_qk_norm_rope``.

``--mode joint`` times what one dual-stream block does to hand Q/K/V to attention:

* ``native``: both streams through the eager chain, then the three text-first ``torch.cat`` calls;
* ``per_stream``: both streams through ``fused_qk_norm_rope`` (main's default above the token
  gate), then the same three cats;
* ``joint``: one ``fused_joint_qkv_norm_rope`` launch. Its ``[B*(S_txt+S_img), D]`` table is packed
  once per forward in the model, so it is built outside the timed region.

Examples:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_qwen_image_qk_norm_rope.py \
      --seq-len 4096 --warmup 20 --iters 100 --include-compiled --profile

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_qwen_image_qk_norm_rope.py \
      --mode joint --txt-len 512 --img-len 4096 --batches 1,2,4,8
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_joint_qkv_norm_rope,
    fused_qk_norm_rope,
    pack_qk_norm_rope_table,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


@functools.cache
def _rope() -> RotaryEmbedding:
    # Built once, as the transformer keeps one RotaryEmbedding per attention block.
    return RotaryEmbedding(is_neox_style=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("per-stream", "joint"), default="per-stream")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batches", default=None, help="joint mode: comma-separated batch sizes (overrides --batch)")
    parser.add_argument("--seq-len", type=int, default=4096, help="per-stream mode: tokens per stream")
    parser.add_argument("--txt-len", type=int, default=512, help="joint mode: text tokens")
    parser.add_argument("--img-len", type=int, default=4096, help="joint mode: image tokens (1024x1024 -> 4096)")
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
    cos = freqs.real.to(q.dtype)
    sin = freqs.imag.to(q.dtype)
    rope = _rope()
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


def _measure(fn: Callable[[], object], warmup: int, iters: int) -> list[float]:
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


def _profile(fn: Callable[[], object]) -> str:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _run_per_stream(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> None:
    eps = 1e-6
    q, k, q_weight, k_weight, freqs = _make_inputs(args, dtype, device)

    def native():
        return _native(q, k, q_weight, k_weight, freqs, eps)

    def fused():
        return _fused(q, k, q_weight, k_weight, freqs, eps)

    results: dict[str, object] = {
        "mode": "per-stream",
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
    results["max_abs_q"] = _max_abs(actual_q, expected_q)
    results["max_abs_k"] = _max_abs(actual_k, expected_k)
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


def _make_joint_inputs(args: argparse.Namespace, batch: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(2026)

    def stream(seq_len: int):
        q = torch.randn(batch, seq_len, args.heads, args.head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seq_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, seq_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
        q_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
        k_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
        angles = torch.randn(seq_len, args.head_dim // 2, device=device, dtype=torch.float32)
        freqs = torch.polar(torch.ones_like(angles), angles)
        return q, k, v, q_weight, k_weight, freqs

    return stream(args.txt_len), stream(args.img_len)


def _run_joint_batch(args: argparse.Namespace, batch: int, dtype: torch.dtype, device: torch.device) -> dict:
    eps = 1e-6
    txt, img = _make_joint_inputs(args, batch, dtype, device)
    txt_q, txt_k, txt_v, txt_qw, txt_kw, txt_freqs = txt
    img_q, img_k, img_v, img_qw, img_kw, img_freqs = img
    joint_freqs = torch.cat((txt_freqs, img_freqs), dim=0)
    # Packed once per forward in QwenImageTransformer2DModel, shared by all blocks.
    table = pack_qk_norm_rope_table(
        joint_freqs.real, joint_freqs.imag, batch, dtype=torch.float32, activation_dtype=dtype
    )
    if table is None:
        raise RuntimeError("the fused CUDA path is unavailable for these inputs (needs BF16 on CUDA with Triton)")

    def native():
        tq, tk = _native(txt_q, txt_k, txt_qw, txt_kw, txt_freqs, eps)
        iq, ik = _native(img_q, img_k, img_qw, img_kw, img_freqs, eps)
        return torch.cat([tq, iq], dim=1), torch.cat([tk, ik], dim=1), torch.cat([txt_v, img_v], dim=1)

    def per_stream():
        tq, tk = _fused(txt_q, txt_k, txt_qw, txt_kw, txt_freqs, eps)
        iq, ik = _fused(img_q, img_k, img_qw, img_kw, img_freqs, eps)
        return torch.cat([tq, iq], dim=1), torch.cat([tk, ik], dim=1), torch.cat([txt_v, img_v], dim=1)

    def joint():
        return fused_joint_qkv_norm_rope(
            txt_q, txt_k, txt_v, img_q, img_k, img_v, txt_qw, txt_kw, img_qw, img_kw, table, eps
        )

    arms = {"native": native, "per_stream": per_stream, "joint": joint}
    stats = {name: _stats(_measure(fn, args.warmup, args.iters)) for name, fn in arms.items()}
    outputs = {name: fn() for name, fn in arms.items()}
    torch.accelerator.synchronize()
    result = {
        "batch": batch,
        "txt_len": args.txt_len,
        "img_len": args.img_len,
        "joint_tokens": batch * (args.txt_len + args.img_len),
        **stats,
        "joint_speedup_vs_native": stats["native"]["median_ms"] / stats["joint"]["median_ms"],
        "joint_speedup_vs_per_stream": stats["per_stream"]["median_ms"] / stats["joint"]["median_ms"],
        "per_stream_speedup_vs_native": stats["native"]["median_ms"] / stats["per_stream"]["median_ms"],
        "max_abs_joint_vs_per_stream": max(_max_abs(a, b) for a, b in zip(outputs["joint"], outputs["per_stream"])),
        "max_abs_joint_vs_native": max(_max_abs(a, b) for a, b in zip(outputs["joint"], outputs["native"])),
    }
    if args.profile:
        result["joint_profile"] = _profile(joint)
    return result


def _run_joint(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> None:
    batches = [int(b) for b in args.batches.split(",")] if args.batches else [args.batch]
    cases = [_run_joint_batch(args, batch, dtype, device) for batch in batches]
    profiles = [case.pop("joint_profile", None) for case in cases]
    results: dict[str, object] = {
        "mode": "joint",
        "device": torch.cuda.get_device_name(),
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": cases,
    }
    print(json.dumps(results, indent=2))
    print("\n| B | tokens | native ms | per-stream ms | joint ms | joint vs native | joint vs per-stream |")
    print("|---|---|---|---|---|---|---|")
    for case in cases:
        print(
            f"| {case['batch']} | {case['joint_tokens']} | {case['native']['median_ms']:.4f} "
            f"| {case['per_stream']['median_ms']:.4f} | {case['joint']['median_ms']:.4f} "
            f"| {case['joint_speedup_vs_native']:.2f}x | {case['joint_speedup_vs_per_stream']:.2f}x |"
        )
    for case, prof in zip(cases, profiles):
        if prof:
            print(f"\n# joint profile, batch {case['batch']}")
            print(prof)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    if args.mode == "joint":
        _run_joint(args, dtype, device)
    else:
        _run_per_stream(args, dtype, device)


if __name__ == "__main__":
    main()
