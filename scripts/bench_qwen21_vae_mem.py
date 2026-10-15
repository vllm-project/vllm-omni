#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark for Qwen-Image-2.1 VAE memory optimizations (#8775 / #8776).

Measures peak allocated + reserved CUDA memory for untitled single-frame
decode with the opt-in on vs off.  The VAE is constructed from scratch with
the same tiny dims as ``test_qwen_image_vae_optimizations.py`` so this script
runs end-to-end in a few seconds on any CUDA-capable machine — useful as a
local smoke check before a real model run, and as a reproducible baseline
for upstream CI.

Usage:
    python scripts/bench_qwen21_vae_mem.py                  # default 1024² latent
    python scripts/bench_qwen21_vae_mem.py --size 512       # 512² latent
    python scripts/bench_qwen21_vae_mem.py --warmup 3 --iters 20
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def build_tiny_vae(device: torch.device, dtype: torch.dtype = torch.float32):
    """Construct a QwenImage VAE (toy dims, no pretrained weights)."""
    # Import lazily so that --help works even if diffusers / vllm_omni are
    # missing; users who just want to see the argparse output shouldn't
    # need the full dependency chain.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "autoencoder_kl_qwenimage",
        "vllm_omni/diffusion/models/qwen_image/autoencoder_kl_qwenimage.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoencoder_kl_qwenimage"] = mod
    spec.loader.exec_module(mod)

    vae_cls = mod.AutoencoderKLQwenImage
    vae = vae_cls(
        base_dim=16,
        z_dim=4,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True],
    )
    vae.eval()
    vae = vae.to(device=device, dtype=dtype)
    return vae


def measure_decode(vae, z, *, skip_cache: bool, warmup: int, iters: int):
    """Run decode ``warmup + iters`` times, report memory and timing stats."""
    vae.skip_single_frame_vae_cache = bool(skip_cache)

    # Warm up (includes JIT / cuDNN init) before we touch CUDA memory
    # counters — otherwise the first run's allocations pollute the peak.
    with torch.no_grad():
        for _ in range(warmup):
            _ = vae._decode(z, return_dict=False)[0]

    if z.device.type == "cuda":
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats(z.device)

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = vae._decode(z, return_dict=False)[0]
            times.append((time.perf_counter() - t0) * 1000.0)

    if z.device.type == "cuda":
        peak_alloc = torch.accelerator.max_memory_allocated(z.device)
        peak_reserved = torch.accelerator.max_memory_reserved(z.device)
    else:
        peak_alloc = peak_reserved = 0

    return {
        "peak_alloc": peak_alloc,
        "peak_reserved": peak_reserved,
        "mean_ms": sum(times) / len(times),
        "median_ms": sorted(times)[len(times) // 2],
        "iters": iters,
    }


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--size",
        type=int,
        default=1024,
        help="spatial size of the latent input in pixels (after VAE spatial compression)",
    )
    ap.add_argument("--warmup", type=int, default=5, help="warmup iterations before measurement")
    ap.add_argument("--iters", type=int, default=20, help="measured iterations")
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(
            "[WARN] CUDA not available; running on CPU — memory counters "
            "will all read 0. Use --warmup/--iters for timing only."
        )
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Device    : {device}")
    print(f"Dtype     : {dtype}")
    print(f"Latent sz : {args.size}² → [{args.size}² px after decode]")
    print(f"Warmup / iter : {args.warmup} / {args.iters}")
    print("=" * 60)

    vae = build_tiny_vae(device, dtype)
    latent_h = args.size // vae.spatial_compression_ratio
    latent_w = args.size // vae.spatial_compression_ratio

    z = torch.randn(1, vae.z_dim, 1, latent_h, latent_w, device=device, dtype=dtype)

    # --- Baseline (opt-in off) ---
    baseline = measure_decode(vae, z, skip_cache=False, warmup=args.warmup, iters=args.iters)

    # --- Opt-in on ---
    option = measure_decode(vae, z, skip_cache=True, warmup=args.warmup, iters=args.iters)

    print()
    print(f"{'Mode':<22s} {'Peak alloc':>12s} {'Peak reserved':>14s} {'Median ms':>10s}")
    print("-" * 60)
    print(
        f"{'skip_cache = off':<22s} {human(baseline['peak_alloc']):>12s} "
        f"{human(baseline['peak_reserved']):>14s} {baseline['median_ms']:>10.2f}"
    )
    print(
        f"{'skip_cache = on':<22s} {human(option['peak_alloc']):>12s} "
        f"{human(option['peak_reserved']):>14s} {option['median_ms']:>10.2f}"
    )
    print()

    if device.type == "cuda":
        delta_alloc = baseline["peak_alloc"] - option["peak_alloc"]
        delta_reserved = baseline["peak_reserved"] - option["peak_reserved"]
        pct_alloc = 100.0 * delta_alloc / baseline["peak_alloc"] if baseline["peak_alloc"] else 0
        pct_reserved = 100.0 * delta_reserved / baseline["peak_reserved"] if baseline["peak_reserved"] else 0

        print(f"Allocation saved : {human(delta_alloc)} ({pct_alloc:.1f}% reduction)")
        print(f"Reservation saved: {human(delta_reserved)} ({pct_reserved:.1f}% reduction)")

        # Health check: opt-in must NOT regress memory
        if option["peak_alloc"] > baseline["peak_alloc"] * 1.01:
            print("\n[FAIL] opt-in increased allocated memory — regression!")
            return 1
        print("\n[PASS] opt-in does not regress peak allocated memory.")
    else:
        print("(Memory counters unavailable on CPU — skipping regression check.)")

    # --- Byte-identical sanity ---
    vae.skip_single_frame_vae_cache = False
    with torch.no_grad():
        ref = vae._decode(z, return_dict=False)[0].cpu()
    vae.skip_single_frame_vae_cache = True
    with torch.no_grad():
        opt = vae._decode(z, return_dict=False)[0].cpu()
    if torch.equal(ref, opt):
        print("[PASS] output is byte-identical (skip_cache on vs off).")
    else:
        print(f"[FAIL] outputs differ — max_abs_diff={(ref - opt).abs().max().item()}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
