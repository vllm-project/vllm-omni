# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare MammothModa2 VAE decode on one versus two GPUs.

Run from the repo root with ``python -m torch.distributed.run --standalone
--nproc-per-node=2 --module benchmarks.diffusion.bench_mammoth_vae_patch_parallel``.
The script loads only
``gen_vae`` weights, never the AR or DiT stages. Without ``--weights-shard``
it is a randomly initialized mechanism/throughput pilot, not a quality test.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Return finite-input decode error metrics; PSNR assumes output range [-1, 1]."""
    if reference.shape != actual.shape or not reference.numel():
        raise ValueError("decode outputs must have the same nonempty shape")
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("decode outputs must be finite")
    ref = reference.double()
    delta = actual.double() - ref
    mse = delta.square().mean().item()
    relative_l2 = delta.norm().item() / max(ref.norm().item(), 1e-12)
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": relative_l2,
        "psnr_db": float("inf") if mse == 0 else 20 * math.log10(2 / math.sqrt(mse)),
    }


def validate_decode_output(
    output: torch.Tensor, *, expected_shape: tuple[int, ...], expected_dtype: torch.dtype
) -> None:
    """Check the native decode result before metric collection casts it to FP32."""
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"decode output shape {tuple(output.shape)} != {expected_shape}")
    if output.dtype != expected_dtype:
        raise ValueError(f"decode output dtype {output.dtype} != {expected_dtype}")
    if not torch.isfinite(output).all():
        raise ValueError("decode output must be finite")


def json_safe(value):
    """Keep exact-match infinite PSNR from producing non-standard JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_revision() -> tuple[str, list[str]]:
    """Record Git provenance when available, without requiring a Git checkout."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        changes = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("VLLM_OMNI_SOURCE_COMMIT", "unknown"), ["Git metadata unavailable"]
    return commit, changes


def center_stripe_error_profile(
    reference: torch.Tensor, actual: torch.Tensor, *, half_width: int = 32
) -> dict[str, float]:
    """Compare center-stripe error with off-center error, without assuming a tile boundary."""
    if reference.shape != actual.shape or reference.ndim != 4 or reference.shape[-1] <= 2 * half_width:
        raise ValueError("expected equal BCHW outputs wider than the center stripe")
    error = (actual.float() - reference.float()).abs()
    middle = error.shape[-1] // 2
    center = error[..., middle - half_width : middle + half_width]
    off_center = torch.cat((error[..., : middle - half_width], error[..., middle + half_width :]), dim=-1)
    return {"center_mean_abs": center.mean().item(), "off_center_mean_abs": off_center.mean().item()}


def tile_boundary_error_profile(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    row_limit: int,
    rank_grid: tuple[tuple[int, ...], ...],
    half_width: int = 16,
) -> dict:
    """Measure output error at actual tile joins, grouped by rank ownership."""
    if reference.shape != actual.shape or reference.ndim != 4:
        raise ValueError("expected equal BCHW outputs")
    if row_limit < 1 or half_width < 1 or any(size < 1 for size in grid_shape):
        raise ValueError("invalid tile geometry")
    if len(rank_grid) != grid_shape[0] or any(len(row) != grid_shape[1] for row in rank_grid):
        raise ValueError("rank_grid must match grid_shape")
    height, width = reference.shape[-2:]
    if height > grid_shape[0] * row_limit or width > grid_shape[1] * row_limit:
        raise ValueError("tile grid does not cover output")

    cross_rank = torch.zeros((height, width), dtype=torch.bool)
    same_rank = torch.zeros_like(cross_rank)
    vertical_boundaries = [col * row_limit for col in range(1, grid_shape[1]) if col * row_limit < width]
    horizontal_boundaries = [row * row_limit for row in range(1, grid_shape[0]) if row * row_limit < height]
    for row in range(grid_shape[0]):
        y0, y1 = row * row_limit, min((row + 1) * row_limit, height)
        for col, x in enumerate(vertical_boundaries, start=1):
            target = cross_rank if rank_grid[row][col - 1] != rank_grid[row][col] else same_rank
            target[y0:y1, max(0, x - half_width) : min(width, x + half_width)] = True
    for col in range(grid_shape[1]):
        x0, x1 = col * row_limit, min((col + 1) * row_limit, width)
        for row, y in enumerate(horizontal_boundaries, start=1):
            target = cross_rank if rank_grid[row - 1][col] != rank_grid[row][col] else same_rank
            target[max(0, y - half_width) : min(height, y + half_width), x0:x1] = True
    same_rank &= ~cross_rank
    interior = ~(cross_rank | same_rank)
    error = (actual.float() - reference.float()).abs()

    def region(mask: torch.Tensor) -> dict[str, float | int | None]:
        pixels = int(mask.sum().item())
        return {"pixels": pixels, "mean_abs": error[..., mask].mean().item() if pixels else None}

    return {
        "vertical_boundaries": vertical_boundaries,
        "horizontal_boundaries": horizontal_boundaries,
        "cross_rank": region(cross_rank),
        "same_rank": region(same_rank),
        "interior": region(interior),
    }


def tile_rank_layout(
    vae, latents: torch.Tensor, pp_size: int
) -> tuple[tuple[int, int], int, tuple[tuple[int, ...], ...]]:
    """Reconstruct the tile grid and ranks using the executor's own scheduler."""
    tasks, grid = vae.tile_split(latents)
    assigned = vae.distributed_executor._balance_tasks(tasks, pp_size)
    rank_by_coord = {task.grid_coord: rank for rank, rank_tasks in enumerate(assigned) for task in rank_tasks}
    rank_grid = tuple(
        tuple(rank_by_coord[(row, col)] for col in range(grid.grid_shape[1])) for row in range(grid.grid_shape[0])
    )
    return grid.grid_shape, grid.tile_spec["row_limit"], rank_grid


def tile_boundary_evidence(
    vae,
    latents: torch.Tensor,
    tiled_reference: torch.Tensor,
    parallel_output: torch.Tensor,
    *,
    pp_size: int,
    half_width: int = 16,
) -> dict:
    """Tie boundary error measurements to the actual split and rank schedule."""
    grid_shape, row_limit, rank_grid = tile_rank_layout(vae, latents, pp_size)
    return {
        "grid_shape": list(grid_shape),
        "row_limit": row_limit,
        "rank_grid": [list(row) for row in rank_grid],
        "half_width": half_width,
        "error_vs_tiled": tile_boundary_error_profile(
            tiled_reference,
            parallel_output,
            grid_shape=grid_shape,
            row_limit=row_limit,
            rank_grid=rank_grid,
            half_width=half_width,
        ),
    }


def _save_comparison_images(
    reference: torch.Tensor, tiled_reference: torch.Tensor, actual: torch.Tensor, output: Path
) -> None:
    from PIL import Image

    for name, image in (("pp1", reference), ("pp1-tiled", tiled_reference), ("pp2", actual)):
        rgb = ((image[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(rgb).save(output.with_name(f"{output.stem}-{name}.png"))
    for name, baseline in (("absdiff", reference), ("absdiff-vs-tiled", tiled_reference)):
        difference = (actual[0] - baseline[0]).abs().mean(dim=0)
        heatmap = (difference / max(difference.max().item(), 1e-12) * 255).to(torch.uint8).numpy()
        Image.fromarray(heatmap).save(output.with_name(f"{output.stem}-{name}.png"))


def comparison_modes() -> list[tuple[int, bool]]:
    """Include a single-GPU tiled control to isolate parallelism from tiling."""
    return [(1, False), (1, True), (2, True)]


def prepare_memory_measurement(*, clear_cache: bool = False) -> None:
    """Reset allocated peaks without perturbing the allocator unless requested."""
    if clear_cache:
        torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()


def _load_vae_weights(vae: torch.nn.Module, path: Path) -> int:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as shard:
        names = [name for name in shard.keys() if name.startswith("gen_vae.")]
        state = {name.removeprefix("gen_vae."): shard.get_tensor(name) for name in names}
    vae.load_state_dict(state, strict=True)
    return len(names)


def _run_mode(
    vae, latents: torch.Tensor, *, pp_size: int, tiled: bool, warmup: int, iterations: int, clear_cache: bool = False
) -> tuple[dict, torch.Tensor | None]:
    rank = dist.get_rank()
    vae.use_tiling = tiled
    vae.set_parallel_size(pp_size, mode="tile")
    original_latents = latents.detach().float().cpu()
    prepare_memory_measurement(clear_cache=clear_cache)
    samples_ms = []
    output = None
    first_output = None
    with torch.inference_mode():
        for iteration in range(warmup + iterations):
            dist.barrier()
            torch.accelerator.synchronize()
            start = time.perf_counter()
            if pp_size > 1 or rank == 0:
                result = vae.decode(latents, return_dict=False)[0]
            else:
                result = None
            torch.accelerator.synchronize()
            local_ms = (time.perf_counter() - start) * 1000
            elapsed = torch.tensor(local_ms, dtype=torch.float64, device=latents.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if iteration >= warmup:
                samples_ms.append(elapsed.item())
            if rank == 0:
                output = result
                if iteration == warmup:
                    first_output = result.detach().float().cpu()
    peak = torch.tensor(
        [torch.accelerator.max_memory_allocated(), torch.accelerator.max_memory_reserved()],
        dtype=torch.int64,
        device=latents.device,
    )
    gathered = [torch.zeros_like(peak) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, peak)
    if output is not None:
        scale = 2 ** (len(vae.config.block_out_channels) - 1)
        validate_decode_output(
            output,
            expected_shape=(
                latents.shape[0],
                vae.config.out_channels,
                latents.shape[-2] * scale,
                latents.shape[-1] * scale,
            ),
            expected_dtype=latents.dtype,
        )
        output_dtype = str(output.dtype)
        # Freeze each mode's output before later modes reuse CUDA allocator memory.
        output = output.detach().float().cpu()
    else:
        output_dtype = None
    record = {
        "pp_size": pp_size,
        "use_tiling": tiled,
        "latency_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": sorted(samples_ms)[math.ceil(0.95 * len(samples_ms)) - 1],
        "peak_allocated_bytes_per_rank": [item[0].item() for item in gathered],
        "peak_reserved_bytes_per_rank": [item[1].item() for item in gathered],
        "latent_mutation_max_abs": (latents.float().cpu() - original_latents).abs().max().item(),
        "output_dtype": output_dtype,
    }
    if rank == 0:
        record["repeatability"] = error_metrics(first_output, output)
    return record, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True, help="MammothModa2 root config.json")
    parser.add_argument("--weights-shard", type=Path, help="Checkpoint shard containing gen_vae.* keys")
    parser.add_argument("--latent-size", type=int, default=64, help="Square latent H=W; 64 means 512px output")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch algorithms")
    parser.add_argument(
        "--clear-cache-between-modes",
        action="store_true",
        help="Clear CUDA allocator cache before each mode; may perturb BF16 outputs on RTX 3090",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.latent_size < 2 or args.warmup < 0 or args.iterations < 1:
        parser.error("latent-size must be >=2, warmup >=0, and iterations >=1")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This comparison requires exactly two ranks, got {world_size}")
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank, backend="nccl")
    initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size, backend="nccl")
    try:
        torch.manual_seed(2026)
        config = json.loads(args.model_config.read_text())["gen_vae_config"]
        vae = DistributedAutoencoderKL.from_config(config)
        weight_count = _load_vae_weights(vae, args.weights_shard) if args.weights_shard else 0
        dtype = getattr(torch, args.dtype)
        vae.to(device=f"cuda:{local_rank}", dtype=dtype).eval()
        latents = torch.randn(
            (1, config["latent_channels"], args.latent_size, args.latent_size),
            generator=torch.Generator().manual_seed(7),
        )
        latents = (latents / config["scaling_factor"] + config["shift_factor"]).to(
            device=f"cuda:{local_rank}", dtype=dtype
        )

        results = [
            _run_mode(
                vae,
                latents,
                pp_size=pp_size,
                tiled=tiled,
                warmup=args.warmup,
                iterations=args.iterations,
                clear_cache=args.clear_cache_between_modes,
            )
            for pp_size, tiled in comparison_modes()
        ]
        (baseline, reference), (tiled_baseline, tiled_reference), (parallel, actual) = results
        if rank == 0:
            assert reference is not None and tiled_reference is not None and actual is not None
            reference_cpu = reference
            tiled_reference_cpu = tiled_reference
            actual_cpu = actual
            largest_error_index = torch.unravel_index(
                (actual_cpu - tiled_reference_cpu).abs().argmax(), actual_cpu.shape
            )
            commit, worktree_changes = source_revision()
            record = {
                "commit": commit,
                "worktree_changes": worktree_changes,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path("vllm_omni/diffusion/registry.py"),
                        Path("vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py"),
                        Path("vllm_omni/diffusion/models/mammoth_moda2/mammothmoda2_dit_model.py"),
                        Path(__file__).resolve(),
                    )
                },
                "gpu": torch.cuda.get_device_name(local_rank),
                "gpu_count": world_size,
                "torch": torch.__version__,
                "latent_shape": list(latents.shape),
                "dtype": args.dtype,
                "deterministic": args.deterministic,
                "clear_cache_between_modes": args.clear_cache_between_modes,
                "peak_reserved_bytes_note": (
                    "Allocator cache can carry across modes when clear_cache_between_modes=false"
                ),
                "latent_transform": "normal(seed=7) / scaling_factor + shift_factor",
                "tile_latent_min_size": vae.tile_latent_min_size,
                "tile_overlap_factor": vae.tile_overlap_factor,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "weight_source": str(args.weights_shard) if args.weights_shard else "random_initialization",
                "loaded_vae_tensors": weight_count,
                "baseline": baseline,
                "tiled_baseline": tiled_baseline,
                "parallel": parallel,
                "output_shape": list(actual.shape),
                "error": error_metrics(reference_cpu, actual_cpu),
                "error_vs_tiled": error_metrics(tiled_reference_cpu, actual_cpu),
                "tiled_vs_untiled": error_metrics(reference_cpu, tiled_reference_cpu),
                "center_stripe_error": center_stripe_error_profile(reference_cpu, actual_cpu),
                "center_stripe_error_vs_tiled": center_stripe_error_profile(tiled_reference_cpu, actual_cpu),
                "tile_boundaries": tile_boundary_evidence(vae, latents, tiled_reference_cpu, actual_cpu, pp_size=2)
                if args.latent_size > vae.tile_latent_min_size
                else None,
                "max_error_index_vs_tiled": [int(index) for index in largest_error_index],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            record = json_safe(record)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            _save_comparison_images(reference_cpu, tiled_reference_cpu, actual_cpu, args.output)
            print(json.dumps(record, indent=2, allow_nan=False), flush=True)
    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
