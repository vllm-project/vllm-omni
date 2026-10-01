# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark and exactness check for the Ming-Image Qwen VAE decoder.

Loads the Qwen VAE used by Ming-Image (``AutoencoderKLQwenImage``), decodes the
same seeded latents through the single-rank full decoder and through the tile /
spatial-shard decode paths, and reports latency, peak memory, speedup, max/mean
absolute difference and PSNR against the full reference.

Single GPU::

    python benchmarks/diffusion/bench_qwen_vae_decode.py \
        --model /path/to/checkpoint --vae-patch-parallel-size 1

Multi-GPU (one process per GPU; rank 0 prints)::

    torchrun --nproc-per-node 2 benchmarks/diffusion/bench_qwen_vae_decode.py \
        --model /path/to/checkpoint --vae-patch-parallel-size 2 \
        --vae-parallel-mode all

``tile`` distributes the VAE's spatial tiles and is intentionally lossy at tile
boundaries. ``spatial_shard_height`` / ``spatial_shard_width`` shard every decoder
feature map with halo exchange and are expected to be fp-lossless.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Iterable

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLQwenImage

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
VAE_PARALLEL_MODES = ("tile", "spatial_shard_height", "spatial_shard_width")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="inclusionAI/Ming-Image-0.1-Design-Layer", help="HF id or local path")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--size", default="1024x1024", help="Output image WxH")
    parser.add_argument("--frames", type=int, default=1, help="Number of RGBA images decoded in one batch")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--vae-parallel-mode",
        choices=[*VAE_PARALLEL_MODES, "all"],
        default="all",
        help="Parallel decode mode to benchmark (default: all three parallel modes)",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="VAE parallel decode over this many ranks (run under torchrun with as many processes)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
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
) -> AutoencoderKLQwenImage:
    local_vae_only = os.path.isdir(args.model) and not os.path.isdir(os.path.join(args.model, args.subfolder))
    load_kwargs: dict[str, object] = {"torch_dtype": dtype}
    if not local_vae_only:
        load_kwargs["subfolder"] = args.subfolder
    cls = DistributedAutoencoderKLQwenImage if parallel else AutoencoderKLQwenImage
    return cls.from_pretrained(args.model, **load_kwargs).to(device).eval()


def make_latents(
    vae: AutoencoderKLQwenImage,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
):
    width, height = (int(v) for v in args.size.lower().split("x"))
    spatial = int(getattr(vae, "spatial_compression_ratio", 8))
    if width % spatial or height % spatial:
        raise SystemExit(f"size must be a multiple of {spatial}")
    shape = (args.frames, vae.config.z_dim, 1, height // spatial, width // spatial)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


@torch.inference_mode()
def decode(vae: DistributedAutoencoderKLQwenImage, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents, return_dict=False)[0]


def sync() -> None:
    torch.accelerator.synchronize()
    if dist.is_initialized():
        dist.barrier()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)


def run_mode(
    args: argparse.Namespace,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    rank: int,
    parallel: bool,
) -> tuple[torch.Tensor, dict]:
    vae = load_vae(args, dtype, device, parallel)
    latents = make_latents(vae, args, dtype, device)

    if mode == "reference":
        vae.use_tiling = False
        if parallel:
            vae.set_parallel_size(1, mode="tile")
    else:
        # On a single rank there is no sharding to apply; use full decode for the
        # spatial modes and tiled decode for ``tile`` (matching production fallback).
        vae.use_tiling = mode == "tile"
        if parallel:
            vae.set_parallel_size(args.vae_patch_parallel_size, mode=mode)

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
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    stats = {"time_s": min(timings), "peak_gib": peak.item()}
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
    modes: Iterable[str] = VAE_PARALLEL_MODES if args.vae_parallel_mode == "all" else [args.vae_parallel_mode]

    if rank == 0:
        parallel_desc = f" vae_parallel={args.vae_parallel_mode}x{args.vae_patch_parallel_size}" if parallel else ""
        print(
            f"model={args.model} size={args.size} frames={args.frames} "
            f"dtype={args.dtype} world_size={world_size}{parallel_desc}"
        )

    golden = None
    baseline_time = None
    try:
        output, stats = run_mode(args, "reference", dtype, device, rank=rank, parallel=False)
        golden = output
        baseline_time = stats["time_s"]
        if rank == 0:
            print(f"[{'reference':19s}] decode {stats['time_s'] * 1e3:9.1f} ms  peak {stats['peak_gib']:5.2f} GiB")
        del output

        for mode in modes:
            output, stats = run_mode(args, mode, dtype, device, rank=rank, parallel=parallel)
            if rank != 0:
                del output
                continue
            speedup = baseline_time / stats["time_s"] if baseline_time else float("nan")
            diff = (output.float() - golden.float()).abs()
            line = (
                f"[{mode:19s}] decode {stats['time_s'] * 1e3:9.1f} ms  speedup {speedup:5.2f}x  "
                f"peak {stats['peak_gib']:5.2f} GiB  max_delta={diff.max().item():.3e} "
                f"mean_delta={diff.mean().item():.3e}  psnr={psnr(output, golden):.2f} dB"
            )
            print(line)
            del output
            torch.accelerator.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
