# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark the MammothModa2 ``gen_vae`` decode memory modes.

The MammothModa2 DiT stage decodes latents with its own ``gen_vae`` (a
Flux-shaped ``AutoencoderKL``), so ``vae_use_slicing`` / ``vae_use_tiling`` act
on the VAE decode alone.  This script loads the real ``gen_vae`` weights out of
a MammothModa2 checkpoint, decodes seeded latents at every requested output size
and batch size under each mode combination, and reports decode latency (mean
with min-max over the measured iterations), peak torch memory and the deviation
from the untiled, unsliced baseline.

It isolates the option from the AR stage, which dominates end-to-end latency
(~89% of the wall time in the recipe's end-to-end table) and never touches the
VAE -- use it to judge the memory modes, and the recipe's end-to-end table to
judge the serving path.

``--model`` points at a local checkpoint directory (the VAE lives in one shard
of the main checkpoint, under the ``gen_vae.`` prefix); fetch it first if
needed::

    hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview \
        --include config.json '*.safetensors.index.json' 'model-00006-of-00008.safetensors'

Single GPU::

    python benchmarks/diffusion/bench_mammoth_moda2_vae_decode.py \
        --model ./MammothModa2-Preview --sizes 1024,1536,2048,3072

Slicing only splits the batch, so a batch > 1 is where it shows up::

    python benchmarks/diffusion/bench_mammoth_moda2_vae_decode.py \
        --model ./MammothModa2-Preview --sizes 1024,1536 --batches 1,4

Rows that do not fit in the free device memory (the card may be shared) are
reported as skipped instead of aborting the sweep, and are not retried.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import diffusers
import torch
from diffusers import AutoencoderKL
from safetensors import safe_open

# (use_slicing, use_tiling) per mode name, in reporting order.
MODES: dict[str, tuple[bool, bool]] = {
    "baseline": (False, False),
    "slicing": (True, False),
    "tiling": (False, True),
    "slicing+tiling": (True, True),
}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
OOM_MARKERS = ("out of memory", "CUDA error: out of memory")
# Calibrated from the recorded sweep at 1024x1024 batch 1 in bf16: a single
# untiled decode peaks at 2,638 MiB, of which ~350 MiB is the decoder itself;
# the activation part then scales with the output area (5,687 MiB at 1536, 9,953
# at 2048, 22,146 at 3072).  Used only to skip rows the device cannot fit before
# any allocation is attempted, so a shared card does not spend minutes in
# allocator retries; OOM is still handled if the estimate is wrong.  The area
# scaling under-predicts at the top end (it reads 20,942 for the 22,146 MiB row),
# so the decision adds ESTIMATE_SAFETY headroom.
WEIGHTS_MIB = 350.0
ACTIVATION_MIB_1024 = 2288.0
ESTIMATE_SAFETY = 1.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local MammothModa2 checkpoint directory")
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Read the VAE from this subfolder as a diffusers export (no `gen_vae.` prefix) "
        "instead of from the checkpoint's main shards",
    )
    parser.add_argument("--sizes", default="1024,1536,2048,3072", help="Comma-separated square output sizes")
    parser.add_argument("--batches", default="1", help="Comma-separated batch sizes")
    parser.add_argument("--modes", default=",".join(MODES), help="Comma-separated subset of: " + ", ".join(MODES))
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Also write the markdown table to this path")
    return parser.parse_args()


def load_vae(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> AutoencoderKL:
    """Build ``AutoencoderKL`` from the checkpoint's ``gen_vae_config`` and load its weights."""
    root = Path(args.model)
    if not root.is_dir():
        raise SystemExit(f"--model must be a local checkpoint directory, got {args.model!r}")
    config_dir = root / args.subfolder if args.subfolder else root
    config_path = config_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"no config.json under {config_dir}")
    raw = json.loads(config_path.read_text())
    if args.subfolder:
        vae = AutoencoderKL.from_pretrained(config_dir, torch_dtype=dtype)
        return vae.to(device).eval()
    vae_config = raw.get("gen_vae_config")
    if vae_config is None:
        raise SystemExit(f"{config_path} has no `gen_vae_config`; pass --subfolder for a diffusers export")
    vae = AutoencoderKL.from_config(vae_config)
    state = load_gen_vae_state(root, dtype)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the VAE config: missing={missing[:5]} unexpected={unexpected[:5]}")
    # ``from_config`` builds fp32 and ``load_state_dict`` copies into that dtype,
    # so cast the module -- the pipeline decodes in the model's dtype too.
    return vae.to(device=device, dtype=dtype).eval()


def _shards_for_gen_vae(root: Path) -> list[Path]:
    """Safetensors shards that hold ``gen_vae.*``, per the checkpoint's HF index."""
    index_path = next(iter(sorted(root.glob("*.safetensors.index.json"))), None)
    if index_path is None:
        return sorted(root.glob("*.safetensors"))
    index = json.loads(index_path.read_text())
    names = sorted({name for key, name in index["weight_map"].items() if key.startswith("gen_vae.")})
    if not names:
        raise SystemExit(f"{index_path} maps no `gen_vae.` tensors")
    return [root / name for name in names]


def load_gen_vae_state(root: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Read the ``gen_vae.`` tensors, dropping the prefix the checkpoint uses."""
    state: dict[str, torch.Tensor] = {}
    for shard in _shards_for_gen_vae(root):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("gen_vae."):
                    state[key[len("gen_vae.") :]] = handle.get_tensor(key).to(dtype)
    if not state:
        raise SystemExit(f"no `gen_vae.` tensors found in {root}")
    return state


def downscale(vae: AutoencoderKL) -> int:
    return 2 ** (len(vae.config.block_out_channels) - 1)


def make_latents(
    vae: AutoencoderKL, size: int, batch: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    """Seeded latents, scaled and shifted exactly as the pipeline feeds them to ``decode``."""
    factor = downscale(vae)
    if size % factor:
        raise SystemExit(f"--sizes must be multiples of {factor}, got {size}")
    shape = (batch, int(vae.config.latent_channels), size // factor, size // factor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    if vae.config.scaling_factor is not None:
        latents = latents.div(vae.config.scaling_factor)
    if vae.config.shift_factor is not None:
        latents = latents.add(vae.config.shift_factor)
    return latents


@torch.inference_mode()
def decode(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents, return_dict=False)[0]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)  # outputs live in [-1, 1]


def run_row(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    size: int,
    batch: int,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    use_slicing, use_tiling = MODES[mode]
    vae.use_slicing = use_slicing
    vae.use_tiling = use_tiling
    latents = make_latents(vae, size, batch, dtype, device, args.seed)
    # The mode is only exercised above the checkpoint's tile threshold.
    tiling_active = bool(
        use_tiling and (latents.shape[-1] > vae.tile_latent_min_size or latents.shape[-2] > vae.tile_latent_min_size)
    )
    for _ in range(args.warmup):
        decode(vae, latents)
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    timings = []
    output = None
    for _ in range(args.iters):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        output = decode(vae, latents)
        torch.accelerator.synchronize()
        timings.append(time.perf_counter() - start)
    peak_mib = torch.accelerator.max_memory_allocated() / 2**20
    assert output is not None
    samples_ms = [1e3 * timing for timing in timings]
    stats = {
        "ms": sum(samples_ms) / len(samples_ms),
        "ms_min": min(samples_ms),
        "ms_max": max(samples_ms),
        "peak_mib": peak_mib,
        "tiling_active": tiling_active,
        "slicing_active": bool(use_slicing and batch > 1),
    }
    del latents
    return output.detach().float().cpu(), stats


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(marker in str(exc) for marker in OOM_MARKERS)


def free_mib(device: torch.device) -> int:
    free, _ = torch.accelerator.get_memory_info(device)
    return int(free // 2**20)


def estimate_peak_mib(vae: AutoencoderKL, size: int, batch: int, use_slicing: bool, use_tiling: bool) -> float:
    """Rough decode peak, to skip rows the device cannot fit (see the constants)."""
    # Tiling caps the spatial extent at one tile, so the per-image activation is
    # the 1024x1024 figure (measured 2,623 MiB at 1536, 2,650 at 1536 batch 2
    # sliced); below the threshold, or untiled, it scales with the output area,
    # and the whole batch is decoded at once unless slicing is on.
    tiled = use_tiling and size // downscale(vae) > vae.tile_latent_min_size
    per_image = ACTIVATION_MIB_1024 if tiled else ACTIVATION_MIB_1024 * (size / 1024) ** 2
    return (WEIGHTS_MIB + per_image * (1 if use_slicing else batch)) * ESTIMATE_SAFETY


def describe_device(device: torch.device) -> str:
    # get_device_name has no portable equivalent and is display-only; the memory
    # numbers below come from the accelerator API.
    name = torch.cuda.get_device_name(device)
    _, total = torch.accelerator.get_memory_info(device)
    return f"{name} ({total // 2**20} MiB total, {free_mib(device)} MiB free before the sweep)"


def header(args: argparse.Namespace, vae: AutoencoderKL, dtype: torch.dtype, device: torch.device) -> list[str]:
    return [
        f"model={args.model} subfolder={args.subfolder or 'gen_vae.* (checkpoint shards)'} dtype={args.dtype}",
        f"device={describe_device(device)}",
        f"torch={torch.__version__} diffusers={diffusers.__version__}",
        f"warmup={args.warmup} iters={args.iters} seed={args.seed} batch={args.batches}",
        f"tile_sample_min_size={vae.tile_sample_min_size} tile_latent_min_size={vae.tile_latent_min_size} "
        f"tile_overlap_factor={vae.tile_overlap_factor} scale={downscale(vae)}",
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    sizes = [int(size) for size in args.sizes.split(",") if size.strip()]
    batches = [int(batch) for batch in args.batches.split(",") if batch.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r}; choose from {list(MODES)}")

    vae = load_vae(args, dtype, device)
    for line in header(args, vae, dtype, device):
        print(line)
    print()

    columns = [
        "Size",
        "Batch",
        "Mode",
        "Slicing",
        "Tiling",
        "Decode ms mean (min-max)",
        "Peak MiB",
        "PSNR dB",
        "Max abs diff",
    ]
    rows: list[str] = []
    references: dict[tuple[int, int], torch.Tensor] = {}
    for batch in batches:
        for size in sizes:
            for mode in modes:
                label = f"{size}x{size} B{batch} {mode}"
                estimate = estimate_peak_mib(vae, size, batch, *MODES[mode])
                available = free_mib(device)
                if estimate > available:
                    rows.append(
                        f"| {size}x{size} | {batch} | {mode} | - | - | skipped | "
                        f"needs ~{estimate:.0f} MiB, {available} MiB free | - | - |"
                    )
                    print(f"[{label:28s}] skipped: needs ~{estimate:.0f} MiB, {available} MiB free on the device")
                    continue
                try:
                    output, stats = run_row(args, vae, size, batch, mode, dtype, device)
                except BaseException as exc:  # noqa: BLE001 - report, free, and keep sweeping
                    if not is_oom(exc):
                        raise
                    torch.accelerator.empty_cache()
                    gc.collect()
                    rows.append(
                        f"| {size}x{size} | {batch} | {mode} | - | - | skipped | "
                        f"OOM ({free_mib(device)} MiB free) | - | - |"
                    )
                    print(f"[{label:28s}] skipped: out of memory ({free_mib(device)} MiB free on the device)")
                    continue
                if mode == "baseline":
                    references[(size, batch)] = output
                golden = references.get((size, batch))
                psnr_text = diff_text = "-"
                if golden is not None and mode != "baseline":
                    max_diff = (output - golden).abs().max().item()
                    diff_text = f"{max_diff:.4f}"
                    psnr_text = "identical" if max_diff == 0 else f"{psnr(output, golden):.2f}"
                rows.append(
                    f"| {size}x{size} | {batch} | {mode} | "
                    f"{'yes' if stats['slicing_active'] else 'no'} | "
                    f"{'yes' if stats['tiling_active'] else 'no'} | "
                    f"{stats['ms']:.1f} ({stats['ms_min']:.1f}-{stats['ms_max']:.1f}) | "
                    f"{stats['peak_mib']:.0f} | {psnr_text} | {diff_text} |"
                )
                print(
                    f"[{label:28s}] {stats['ms']:9.1f} ms (min {stats['ms_min']:.1f}, max {stats['ms_max']:.1f})  "
                    f"peak {stats['peak_mib']:7.0f} MiB  "
                    f"tiling={stats['tiling_active']} slicing={stats['slicing_active']}  "
                    f"psnr={psnr_text} max_abs_diff={diff_text}"
                )
                del output
                torch.accelerator.empty_cache()

    print()
    table = [f"| {' | '.join(columns)} |", f"| {' | '.join('---' for _ in columns)} |", *rows]
    print("\n".join(table))
    if args.output:
        Path(args.output).write_text("\n".join([*header(args, vae, dtype, device), "", *table, ""]))
        print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
