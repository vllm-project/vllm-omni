# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark the fused Q/K RMSNorm + RoPE kernels with and without fusion, per model geometry.

``--preset`` selects a model's attention geometry and the kernel flags its transformer passes:

* dual-stream (``joint`` mode: Flux.2, FLUX.1, HunyuanVideo-1.5, Ovis-Image, Qwen-Image): what one
  block does to hand Q/K/V to attention, ``native`` (eager RMSNorm + RoPE per stream, then the three
  text-first ``torch.cat`` calls), ``per_stream`` (``fused_qk_norm_rope`` per stream, then the same
  cats) and ``joint`` (one ``fused_joint_qkv_norm_rope`` launch; its table is packed once per forward
  in the model, so it is built outside the timed region);
* single-stream (``single`` mode: Z-Image's unified image+text sequence): ``native`` against
  ``fused_qk_norm_rope``.

Any preset value can be overridden on the command line.

Examples:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_fused_qk_norm_rope.py \
      --preset flux2 --batches 1,2,4,8

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_fused_qk_norm_rope.py \
      --preset hunyuan15 --txt-len 256 --img-len 32760 --batches 1

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_fused_qk_norm_rope.py \
      --preset z_image --seq-len 4608 --batches 1,2,4,8 --profile
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    fused_joint_qkv_norm_rope,
    fused_qk_norm_rope,
    fused_qk_norm_rope_available,
    pack_qk_norm_rope_table,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


# heads / head_dim from the released checkpoints; token counts are 1024x1024 image generation with the
# text budget each pipeline pads to (HunyuanVideo-1.5 is listed at the same image geometry so the models
# are comparable; pass --img-len for a video-sized sequence). cuda_rms_norm mirrors the flag each
# transformer passes to the ops.
def _joint(heads: int, *, cuda_rms_norm: bool = True, txt_len: int = 512, img_len: int = 4096) -> dict[str, object]:
    return {
        "mode": "joint",
        "heads": heads,
        "head_dim": 128,
        "txt_len": txt_len,
        "img_len": img_len,
        "cuda_rms_norm": cuda_rms_norm,
    }


PRESETS: dict[str, dict[str, object]] = {
    "flux2": _joint(48),
    "flux1": _joint(24),
    "hunyuan15": _joint(16),
    "ovis_image": _joint(24),
    "qwen_image": _joint(24, cuda_rms_norm=False),
    "z_image": {"mode": "single", "heads": 30, "head_dim": 128, "seq_len": 4608, "cuda_rms_norm": True},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--mode", choices=("joint", "single"), default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None, help="defaults to --heads")
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--txt-len", type=int, default=None, help="joint mode: text tokens")
    parser.add_argument("--img-len", type=int, default=None, help="joint mode: image/video tokens")
    parser.add_argument("--seq-len", type=int, default=None, help="single mode: tokens of the unified sequence")
    parser.add_argument("--batches", default="1", help="comma-separated batch sizes")
    cuda_rms = parser.add_mutually_exclusive_group()
    cuda_rms.add_argument("--cuda-rms-norm", dest="cuda_rms_norm", action="store_true", default=None)
    cuda_rms.add_argument("--no-cuda-rms-norm", dest="cuda_rms_norm", action="store_false")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    for key in ("mode", "heads", "head_dim", "txt_len", "img_len", "seq_len", "cuda_rms_norm"):
        if getattr(args, key) is None and key in preset:
            setattr(args, key, preset[key])
    if args.mode is None or args.heads is None or args.head_dim is None:
        parser.error("--preset or all of --mode/--heads/--head-dim are required")
    if args.mode == "joint" and (args.txt_len is None or args.img_len is None):
        parser.error("joint mode needs --txt-len and --img-len")
    if args.mode == "single" and args.seq_len is None:
        parser.error("single mode needs --seq-len")
    if args.kv_heads is None:
        args.kv_heads = args.heads
    if args.cuda_rms_norm is None:
        args.cuda_rms_norm = False
    return args


@functools.cache
def _rope() -> RotaryEmbedding:
    # Built once, as the transformers keep one RotaryEmbedding per attention block.
    return RotaryEmbedding(is_neox_style=False)


def _stream(seq_len: int, heads: int, kv_heads: int, head_dim: int, batch: int, device: torch.device):
    dtype = torch.bfloat16
    q = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, kv_heads, head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    angles = torch.randn(seq_len, head_dim // 2, device=device, dtype=torch.float32)
    return q, k, v, q_weight, k_weight, torch.cos(angles), torch.sin(angles)


def _native(q, k, q_weight, k_weight, cos, sin, eps):
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    rope = _rope()
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    return rope(q, cos, sin), rope(k, cos, sin)


def _fused(q, k, q_weight, k_weight, cos, sin, eps, cuda_rms_norm):
    batch, seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    rope_table = torch.cat((cos, sin), dim=-1).unsqueeze(0).expand(batch, -1, -1).reshape(batch * seq_len, head_dim)
    out_q, out_k = fused_qk_norm_rope(
        q.reshape(batch * seq_len, num_heads, head_dim),
        k.reshape(batch * seq_len, num_kv_heads, head_dim),
        q_weight,
        k_weight,
        rope_table,
        eps,
        interleaved=True,
        cuda_rms_norm=cuda_rms_norm,
    )
    return out_q.reshape(batch, seq_len, num_heads, head_dim), out_k.reshape(batch, seq_len, num_kv_heads, head_dim)


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


def _run_single(args: argparse.Namespace, batch: int, device: torch.device) -> dict:
    eps = 1e-6
    torch.manual_seed(2026)
    q, k, _, q_weight, k_weight, cos, sin = _stream(
        args.seq_len, args.heads, args.kv_heads, args.head_dim, batch, device
    )
    if not fused_qk_norm_rope_available(device, torch.bfloat16, args.head_dim, args.head_dim):
        raise RuntimeError("the fused CUDA path is unavailable here (needs BF16 on CUDA with Triton)")

    def native():
        return _native(q, k, q_weight, k_weight, cos, sin, eps)

    def fused():
        return _fused(q, k, q_weight, k_weight, cos, sin, eps, args.cuda_rms_norm)

    arms = {"native": native, "fused": fused}
    stats = {name: _stats(_measure(fn, args.warmup, args.iters)) for name, fn in arms.items()}
    outputs = {name: fn() for name, fn in arms.items()}
    torch.accelerator.synchronize()
    result = {
        "batch": batch,
        "seq_len": args.seq_len,
        "tokens": batch * args.seq_len,
        **stats,
        "fused_speedup_vs_native": stats["native"]["median_ms"] / stats["fused"]["median_ms"],
        "max_abs_fused_vs_native": max(_max_abs(a, b) for a, b in zip(outputs["fused"], outputs["native"])),
    }
    if args.profile:
        result["profile"] = _profile(fused)
    return result


def _run_joint(args: argparse.Namespace, batch: int, device: torch.device) -> dict:
    eps = 1e-6
    torch.manual_seed(2026)
    txt_q, txt_k, txt_v, txt_qw, txt_kw, txt_cos, txt_sin = _stream(
        args.txt_len, args.heads, args.kv_heads, args.head_dim, batch, device
    )
    img_q, img_k, img_v, img_qw, img_kw, img_cos, img_sin = _stream(
        args.img_len, args.heads, args.kv_heads, args.head_dim, batch, device
    )
    # Packed once per forward in the transformer and shared by all of its blocks.
    table = pack_qk_norm_rope_table(
        torch.cat((txt_cos, img_cos)),
        torch.cat((txt_sin, img_sin)),
        batch,
        dtype=torch.float32,
        activation_dtype=torch.bfloat16,
        cuda_rms_norm=args.cuda_rms_norm,
    )
    if table is None:
        raise RuntimeError("the fused CUDA path is unavailable here (needs BF16 on CUDA with Triton)")

    def native():
        tq, tk = _native(txt_q, txt_k, txt_qw, txt_kw, txt_cos, txt_sin, eps)
        iq, ik = _native(img_q, img_k, img_qw, img_kw, img_cos, img_sin, eps)
        return torch.cat([tq, iq], dim=1), torch.cat([tk, ik], dim=1), torch.cat([txt_v, img_v], dim=1)

    def per_stream():
        tq, tk = _fused(txt_q, txt_k, txt_qw, txt_kw, txt_cos, txt_sin, eps, args.cuda_rms_norm)
        iq, ik = _fused(img_q, img_k, img_qw, img_kw, img_cos, img_sin, eps, args.cuda_rms_norm)
        return torch.cat([tq, iq], dim=1), torch.cat([tk, ik], dim=1), torch.cat([txt_v, img_v], dim=1)

    def joint():
        return fused_joint_qkv_norm_rope(
            txt_q,
            txt_k,
            txt_v,
            img_q,
            img_k,
            img_v,
            txt_qw,
            txt_kw,
            img_qw,
            img_kw,
            table,
            eps,
            cuda_rms_norm=args.cuda_rms_norm,
        )

    arms = {"native": native, "per_stream": per_stream, "joint": joint}
    stats = {name: _stats(_measure(fn, args.warmup, args.iters)) for name, fn in arms.items()}
    outputs = {name: fn() for name, fn in arms.items()}
    torch.accelerator.synchronize()
    result = {
        "batch": batch,
        "txt_len": args.txt_len,
        "img_len": args.img_len,
        "tokens": batch * (args.txt_len + args.img_len),
        **stats,
        "joint_speedup_vs_native": stats["native"]["median_ms"] / stats["joint"]["median_ms"],
        "joint_speedup_vs_per_stream": stats["per_stream"]["median_ms"] / stats["joint"]["median_ms"],
        "per_stream_speedup_vs_native": stats["native"]["median_ms"] / stats["per_stream"]["median_ms"],
        "max_abs_joint_vs_per_stream": max(_max_abs(a, b) for a, b in zip(outputs["joint"], outputs["per_stream"])),
        "max_abs_joint_vs_native": max(_max_abs(a, b) for a, b in zip(outputs["joint"], outputs["native"])),
    }
    if args.profile:
        result["profile"] = _profile(joint)
    return result


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    batches = [int(b) for b in args.batches.split(",")]
    runner = _run_joint if args.mode == "joint" else _run_single
    cases = [runner(args, batch, device) for batch in batches]
    profiles = [case.pop("profile", None) for case in cases]
    results: dict[str, object] = {
        "preset": args.preset,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "cuda_rms_norm": args.cuda_rms_norm,
        "dtype": "bf16",
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": cases,
    }
    print(json.dumps(results, indent=2))
    if args.mode == "joint":
        print("\n| B | tokens | native ms | per-stream ms | joint ms | joint vs native | joint vs per-stream |")
        print("|---|---|---|---|---|---|---|")
        for case in cases:
            print(
                f"| {case['batch']} | {case['tokens']} | {case['native']['median_ms']:.4f} "
                f"| {case['per_stream']['median_ms']:.4f} | {case['joint']['median_ms']:.4f} "
                f"| {case['joint_speedup_vs_native']:.2f}x | {case['joint_speedup_vs_per_stream']:.2f}x |"
            )
    else:
        print("\n| B | tokens | native ms | fused ms | fused vs native |")
        print("|---|---|---|---|---|")
        for case in cases:
            print(
                f"| {case['batch']} | {case['tokens']} | {case['native']['median_ms']:.4f} "
                f"| {case['fused']['median_ms']:.4f} | {case['fused_speedup_vs_native']:.2f}x |"
            )
    for case, prof in zip(cases, profiles):
        if prof:
            print(f"\n# profile, batch {case['batch']}")
            print(prof)


if __name__ == "__main__":
    main()
