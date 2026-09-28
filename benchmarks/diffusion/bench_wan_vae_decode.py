# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark and exactness check for the Wan VAE decoder fast path.

Loads a diffusers Wan VAE (by default the Cosmos3 one), decodes seeded latents
for the requested video size at each ``--fast-path`` level and reports wall
time, peak memory and, against the ``off`` level, bitwise equality, max abs
difference and PSNR. ``--profile`` adds a torch.profiler CUDA kernel table with
the convolution / layout-transpose share.

Single GPU::

    python benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano
    python benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano \
        --fast-path off,lossless --frames 33 --profile
    python benchmarks/diffusion/bench_wan_vae_decode.py --tiny --frames 9 --size 256x256

Multi-GPU VAE parallel decode (launch one process per GPU with ``torchrun``; the
decode is timed across all ranks and only rank 0 prints)::

    torchrun --nproc-per-node 2 benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano \
        --vae-patch-parallel-size 2 --vae-parallel-mode tile

``tile`` distributes the VAE's own spatial tiles over the ranks (it needs the
video to exceed the tile size, 256x256 pixels by default) and ``spatial_shard_*``
shards every decoder feature map along height or width with halo exchange. Both
return the full video on rank 0 only, so equality and PSNR are computed there.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Iterable

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    VAE_FAST_PATH_LEVELS,
    install_wan_vae_fastpath,
)

TINY_CONFIG = dict(
    base_dim=32,
    decoder_base_dim=64,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
VAE_PARALLEL_MODES = ("tile", "spatial_shard_height", "spatial_shard_width")
_CONV_TAGS = ("conv", "cudnn", "gemm", "cutlass", "xmma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano", help="HF id or local path of the pipeline")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--tiny", action="store_true", help="Use a small random VAE instead of loading weights")
    parser.add_argument("--size", default="1280x720", help="Output video WxH")
    parser.add_argument("--frames", type=int, default=189, help="Output frame count (1 + 4k)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--fast-path", default=",".join(VAE_FAST_PATH_LEVELS), help="Comma-separated levels")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="Print a torch.profiler CUDA kernel table")
    parser.add_argument("--profile-rows", type=int, default=25)
    parser.add_argument("--save-output", default=None, help="Save the `off` decode output to this .pt path")
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="VAE parallel decode over this many ranks (run under torchrun with as many processes)",
    )
    parser.add_argument(
        "--vae-parallel-mode",
        choices=VAE_PARALLEL_MODES,
        default="tile",
        help="tile: distribute the VAE's spatial tiles; spatial_shard_*: shard feature maps with halo exchange",
    )
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    """Join the torchrun process group (if any) and return ``(rank, world_size)``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
    initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size)
    return rank, world_size


def load_vae(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device, parallel: bool
) -> OmniAutoencoderKLWan:
    cls = DistributedAutoencoderKLWan if parallel else OmniAutoencoderKLWan
    if args.tiny:
        torch.manual_seed(args.seed)
        vae = cls(**TINY_CONFIG).to(device=device, dtype=dtype).eval()
        if parallel:
            vae.init_distributed()
    else:
        vae = cls.from_pretrained(args.model, subfolder=args.subfolder, torch_dtype=dtype).to(device).eval()
    if parallel:
        # Same setup as the registry: parallel decode needs tiling on.
        vae.use_tiling = True
        vae.set_parallel_size(args.vae_patch_parallel_size, mode=args.vae_parallel_mode)
    return vae


def make_latents(vae: OmniAutoencoderKLWan, args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    width, height = (int(v) for v in args.size.lower().split("x"))
    spatial = int(getattr(vae.config, "scale_factor_spatial", 16))
    temporal = int(getattr(vae.config, "scale_factor_temporal", 4))
    if width % spatial or height % spatial or (args.frames - 1) % temporal:
        raise SystemExit(f"size must be a multiple of {spatial} and frames must be 1 + {temporal}k")
    shape = (1, vae.config.z_dim, (args.frames - 1) // temporal + 1, height // spatial, width // spatial)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    mean = getattr(vae.config, "latents_mean", None)
    std = getattr(vae.config, "latents_std", None)
    if mean is not None and std is not None:
        mean = torch.as_tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        latents = latents * std + mean
    return latents


@torch.inference_mode()
def decode(vae: OmniAutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents, return_dict=False)[0]


def sync() -> None:
    """Wait for this rank's GPU and, under torchrun, for every other rank."""
    torch.accelerator.synchronize()
    if dist.is_initialized():
        dist.barrier()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)  # outputs live in [-1, 1]


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    int_dtype = torch.int16 if a.element_size() == 2 else torch.int32
    return bool((a.contiguous().view(int_dtype) == b.contiguous().view(int_dtype)).all())


def profile_decode(vae: OmniAutoencoderKLWan, latents: torch.Tensor, rows: int, verbose: bool) -> None:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        decode(vae, latents)
        sync()
    if not verbose:
        return
    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    events.sort(key=lambda e: e.device_time_total, reverse=True)
    total = sum(e.device_time_total for e in events if e.device_type.name == "CUDA")
    if total == 0:
        total = sum(e.device_time_total for e in events)
    conv = sum(
        e.device_time_total
        for e in events
        if e.device_type.name == "CUDA" and any(tag in e.key.lower() for tag in _CONV_TAGS)
    )
    transpose = sum(e.device_time_total for e in events if e.device_type.name == "CUDA" and "nhwc" in e.key.lower())
    print(
        f"  GPU kernel time: {total / 1e3:.1f} ms  conv-like: {100 * conv / total:.1f}%  "
        f"nchw<->nhwc transposes: {100 * transpose / total:.1f}%"
    )
    print(f"  {'kernel':90s} {'ms':>9s} {'%':>6s} {'calls':>7s}")
    shown = 0
    for e in events:
        if e.device_type.name != "CUDA":
            continue
        share = 100 * e.device_time_total / total
        print(f"  {e.key[:90]:90s} {e.device_time_total / 1e3:9.2f} {share:6.1f} {e.count:7d}")
        shown += 1
        if shown >= rows:
            break


def run_level(
    args: argparse.Namespace,
    level: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    rank: int,
    parallel: bool,
) -> tuple[torch.Tensor, dict]:
    vae = load_vae(args, dtype, device, parallel)
    report = install_wan_vae_fastpath(vae, level=level)
    latents = make_latents(vae, args, dtype, device)
    for _ in range(args.warmup):
        decode(vae, latents)
    sync()
    torch.accelerator.reset_peak_memory_stats()
    timings = []
    output = None
    for _ in range(args.iters):
        sync()
        start = time.perf_counter()
        output = decode(vae, latents)
        sync()
        timings.append(time.perf_counter() - start)
    peak = torch.tensor(torch.accelerator.max_memory_allocated() / 2**30, device=device)
    if dist.is_initialized():
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)  # the largest rank decides whether the config fits
    stats = {
        "installed": report.installed,
        "reason": report.reason,
        "fused_silu": report.fused_silu_dtypes,
        "patched": dict(report.patched),
        "time_s": min(timings),
        "peak_gib": peak.item(),
    }
    if args.profile:
        if rank == 0:
            print(f"[{level}] profile")
        profile_decode(vae, latents, args.profile_rows, verbose=rank == 0)
    assert output is not None
    del vae
    torch.accelerator.empty_cache()
    return output.detach(), stats


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    rank, world_size = init_distributed()
    parallel = args.vae_patch_parallel_size > 1
    if parallel and world_size < args.vae_patch_parallel_size:
        raise SystemExit(
            f"--vae-patch-parallel-size {args.vae_patch_parallel_size} needs torchrun with at least that many "
            f"processes (WORLD_SIZE={world_size})"
        )
    device = torch.device("cuda", torch.accelerator.current_device_index()) if world_size > 1 else torch.device("cuda")
    dtype = DTYPES[args.dtype]
    levels: Iterable[str] = [level.strip() for level in args.fast_path.split(",") if level.strip()]
    for level in levels:
        if level not in VAE_FAST_PATH_LEVELS:
            raise SystemExit(f"unknown fast-path level {level!r}; choose from {VAE_FAST_PATH_LEVELS}")
    if "off" not in levels:
        levels = ["off", *levels]

    if rank == 0:
        parallel_desc = f" vae_parallel={args.vae_parallel_mode}x{args.vae_patch_parallel_size}" if parallel else ""
        print(
            f"model={'tiny' if args.tiny else args.model} size={args.size} frames={args.frames} "
            f"dtype={args.dtype} world_size={world_size}{parallel_desc}"
        )
    golden = None
    baseline_time = None
    try:
        for level in levels:
            output, stats = run_level(args, level, dtype, device, rank=rank, parallel=parallel)
            if level == "off":
                golden = output
                baseline_time = stats["time_s"]
                if args.save_output and rank == 0:
                    torch.save(output.cpu(), args.save_output)
            assert golden is not None
            if rank != 0:
                # Parallel decode assembles the video on rank 0 only; the other
                # ranks just keep the collectives in lockstep.
                del output
                continue
            speedup = baseline_time / stats["time_s"] if baseline_time else float("nan")
            line = (
                f"[{level:13s}] decode {stats['time_s'] * 1e3:9.1f} ms  speedup {speedup:5.2f}x  "
                f"peak {stats['peak_gib']:5.2f} GiB  installed={stats['installed']}"
            )
            if level != "off":
                equal = torch.equal(output, golden)
                line += (
                    f"  torch_equal={equal} bits_equal={bits_equal(output, golden)} "
                    f"max_abs_diff={(output.float() - golden.float()).abs().max().item():.3e} "
                    f"psnr={psnr(output, golden):.2f} dB"
                )
                if stats["installed"]:
                    line += f"  fused_silu={stats['fused_silu'] or 'off'}"
                else:
                    line += f"  reason={stats['reason']}"
            print(line)
            del output
            torch.accelerator.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
