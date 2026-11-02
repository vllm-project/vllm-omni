# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure the SeedVR2 VAE with and without the fused per-frame GroupNorm + SiLU.

No model weights are needed: the released VAE architecture is built with random
weights, and every variant runs on the same weights and inputs::

    python benchmarks/kernels/benchmark_seedvr2_vae_frame_norm.py --frames 5 --height 480 --width 848
    python benchmarks/kernels/benchmark_seedvr2_vae_frame_norm.py --frames 17 --height 720 --width 1280 --tiling

Variants of the GroupNorm + SiLU used by ``_frame_norm_silu``:

- ``unfused``: transpose to ``(b t) c h w``, GroupNorm, transpose back, SiLU.
- ``fused``: ``fused_group_norm_silu(per_frame=True, round_norm_output=True)``.
- ``skip``: identity, only to time how much of encode and decode the op takes.

This is a VAE-stage benchmark, not SeedVR2 serving latency. No CUDA device
produces a JSON skip record and exit status zero.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
from torch.nn import functional as F

import vllm_omni.diffusion.models.seedvr2.vae as vae_module
from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE

FUSED = vae_module.fused_group_norm_silu


def unfused(x, weight, bias, num_groups, eps, *, per_frame, round_norm_output):
    """The path before the fused op, with the fused op's signature."""
    n, c, t = x.shape[:3]
    frames = x.transpose(1, 2).reshape(n * t, c, *x.shape[3:])
    out = F.silu(F.group_norm(frames, num_groups, weight, bias, eps))
    return out.reshape(n, t, c, *x.shape[3:]).transpose(1, 2)


def skip(x, weight, bias, num_groups, eps, *, per_frame, round_norm_output):
    """Identity in place of GroupNorm + SiLU. Its outputs are not compared."""
    return x


VARIANTS = {"unfused": unfused, "fused": FUSED, "skip": skip}


def measure(operation: Callable, warmup: int, iterations: int) -> tuple[dict, torch.Tensor]:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    output = operation()
    torch.accelerator.synchronize()
    stats = {
        "p50_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "extra_peak_mib": (torch.accelerator.max_memory_allocated() - before) / 1024**2,
    }
    return stats, output


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).abs()
    return {
        "max_abs": error.max().item(),
        "mean_abs": error.mean().item(),
        "mismatch_fraction": (actual != expected).float().mean().item(),
        "finite": bool(actual.isfinite().all() and expected.isfinite().all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=5, help="video frames, 4n+1")
    parser.add_argument("--height", type=int, default=480, help="divisible by 16")
    parser.add_argument("--width", type=int, default=848, help="divisible by 16")
    parser.add_argument("--tiling", action="store_true", help="temporal VAE tiling, as with --vae-use-tiling")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if (args.frames - 1) % 4 or args.height % 16 or args.width % 16:
        parser.error("frames must be 4n+1, and height and width divisible by 16")

    if not torch.cuda.is_available():
        print(json.dumps({"benchmark": "seedvr2_vae_frame_norm", "skipped": "no CUDA device"}))
        return

    torch.manual_seed(0)
    vae = SeedVR2VAE().to(device="cuda", dtype=torch.float16).eval()
    vae.use_tiling = args.tiling
    video = torch.rand(1, 3, args.frames, args.height, args.width, device="cuda", dtype=torch.float16) * 2 - 1
    latent_shape = (1, 16, (args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    latent = torch.randn(latent_shape, device="cuda", dtype=torch.float16)
    stages = {"encode": lambda: vae.encode(video).mode(), "decode": lambda: vae.decode(latent)}

    config = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "input": [1, 3, args.frames, args.height, args.width],
        "tiling": args.tiling,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    results, outputs = {}, {}
    with torch.no_grad():
        for variant, operation in VARIANTS.items():
            vae_module.fused_group_norm_silu = operation
            for stage, run in stages.items():
                results[stage, variant], outputs[stage, variant] = measure(run, args.warmup, args.iterations)
                print(json.dumps({**config, "stage": stage, "variant": variant, **results[stage, variant]}))
        vae_module.fused_group_norm_silu = FUSED

    for stage in stages:
        before, after, without = (results[stage, name]["p50_ms"] for name in ("unfused", "fused", "skip"))
        summary = {
            "stage": stage,
            "speedup": before / after,
            "op_share_unfused": (before - without) / before,
            "op_share_fused": (after - without) / after,
            "extra_peak_mib_unfused": results[stage, "unfused"]["extra_peak_mib"],
            "extra_peak_mib_fused": results[stage, "fused"]["extra_peak_mib"],
            "fused_vs_unfused": difference(outputs[stage, "fused"], outputs[stage, "unfused"]),
        }
        print(json.dumps({**config, **summary}))


if __name__ == "__main__":
    main()
