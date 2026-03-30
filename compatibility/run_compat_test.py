#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""run_compat_test.py — Feature compatibility test runner (Python)

Uses batch_text_to_image.py for efficient batch processing with a structured
two-level test matrix:

  Baseline 1  : pure baseline — no acceleration features
  Baseline 2  : --baseline-feature enabled alone
  Addon 1..N  : Baseline-2 arguments + one addon feature each

Each run saves:
  <output_dir>/<baseline_feature>/baseline/batch_generation.log
  <output_dir>/<baseline_feature>/baseline/batch_generation.exitcode
  <output_dir>/<baseline_feature>/baseline/prompt_NN.png
  <output_dir>/<baseline_feature>/baseline/prompt_NN.exitcode
  <output_dir>/<baseline_feature>/<config_name>/batch_generation.log
  <output_dir>/<baseline_feature>/<config_name>/prompt_NN.{png,exitcode}

Note: Individual prompt_NN.log files are not created; all logs are in batch_generation.log

Supported features (13 total across 4 categories):
  Acceleration (cache):  teacache, cache_dit
  Parallelism:           cfg_parallel, ulysses, ring, tp, hsdp
  Memory optimization:   cpu_offload, layerwise_offload, vae_patch_parallel(*), fp8, gguf
  Extensions:            lora

  (*) vae_patch_parallel is addon-only: must be stacked on top of a parallel baseline
  (tp/cfg_parallel/ulysses/ring) and its size must match that baseline's parallel degree.

GPU requirements:
  Maximum 2 GPUs required for any single feature or combination.
  All parallel features use gpu_multiplier=2 (i.e. exactly 2 GPUs).
  Single-GPU features (cache, quantization, CPU offload, LoRA) use gpu_multiplier=1.
  vae_patch_parallel reuses the GPUs already allocated by its baseline (gpu_multiplier=1).

Usage:
  python run_compat_test.py \\
      --baseline-feature cfg_parallel \\
      --addons teacache cache_dit ulysses ring \\
      --model Qwen/Qwen-Image-2512 \\
      --output-dir ./compat_results \\
      --steps 30 \\
      --num-prompts 20 \\
      --prompt-file ./prompts.txt

  # Test LoRA (requires a valid adapter path)
  python run_compat_test.py \\
      --baseline-feature lora \\
      --lora-path /path/to/my/lora_adapter \\
      --model Tongyi-MAI/Z-Image-Turbo

  python analyze_compat_results.py \\
      --results-dir ./compat_results/cfg_parallel \\
      --charts

Extending the feature registry:
  Add a new entry to FEATURE_REGISTRY with:
    args            : list of CLI tokens passed to batch_text_to_image.py
    gpu_multiplier  : parallel dimension (1 = no extra GPUs, 2 = doubles GPU count, …)
    lossy           : True if the feature may degrade image quality (cache/quant methods)
    label           : human-readable display name for reports/charts
    category        : one of acceleration / parallelism / memory / extension
    note            : short description / compatibility notes
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Feature registry ──────────────────────────────────────────────────────────
#
# gpu_multiplier: how many GPUs this feature alone needs.
#   Combined GPU count = product of all active features' multipliers.
# lossy: True if this feature trades quality for speed (cache methods).

FEATURE_REGISTRY: dict[str, dict] = {
    # ── Acceleration: cache methods (lossy) ──────────────────────────────────
    "teacache": {
        "args": ["--cache-backend", "tea_cache"],
        "gpu_multiplier": 1,
        "lossy": True,
        "label": "TeaCache",
        "category": "acceleration",
        "note": "Adaptive caching using modulated inputs. Minor quality trade-off.",
    },
    "cache_dit": {
        "args": ["--cache-backend", "cache_dit"],
        "gpu_multiplier": 1,
        "lossy": True,
        "label": "Cache-DiT",
        "category": "acceleration",
        "note": "DBCache + TaylorSeer + SCM caching. Not compatible with teacache.",
    },
    # ── Acceleration: parallelism methods (lossless) ─────────────────────────
    "cfg_parallel": {
        "args": ["--cfg-parallel-size", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "CFG-Parallel (×2)",
        "category": "parallelism",
        "note": "Splits CFG positive/negative branches across 2 GPUs.",
    },
    "ulysses": {
        "args": ["--ulysses-degree", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "Ulysses SP (×2)",
        "category": "parallelism",
        "note": "Sequence parallelism via all-to-all communication. Lossless.",
    },
    "ring": {
        "args": ["--ring-degree", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "Ring SP (×2)",
        "category": "parallelism",
        "note": "Sequence parallelism via ring-based communication. Lossless.",
    },
    "tp": {
        "args": ["--tensor-parallel-size", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "Tensor Parallel (×2)",
        "category": "parallelism",
        "note": "Shards model weights across 2 GPUs. Not compatible with hsdp.",
    },
    "hsdp": {
        "args": ["--use-hsdp", "--hsdp-shard-size", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "HSDP (shard×2)",
        "category": "parallelism",
        "note": (
            "Weight sharding via FSDP2, redistributed on-demand at runtime. "
            "Not compatible with tp/dp. Best for large models (14B+)."
        ),
    },
    # ── Memory optimization ───────────────────────────────────────────────────
    "cpu_offload": {
        "args": ["--enable-cpu-offload"],
        "gpu_multiplier": 1,
        "lossy": False,
        "label": "CPU Offload (Module-level)",
        "category": "memory",
        "note": "Offloads DiT + text encoder to CPU. Single-card only.",
    },
    "layerwise_offload": {
        "args": ["--enable-layerwise-offload"],
        "gpu_multiplier": 1,
        "lossy": False,
        "label": "CPU Offload (Layerwise)",
        "category": "memory",
        "note": "Keeps only one transformer block on GPU at a time. Single-card only.",
    },
    "vae_patch_parallel": {
        "args": ["--vae-patch-parallel-size", "2"],
        "gpu_multiplier": 1,
        "lossy": False,
        "label": "VAE Patch Parallel (×2)",
        "category": "memory",
        "addon_only": True,
        "note": (
            "Must be used as an addon on top of a parallel baseline (tp, cfg_parallel, ulysses, ring). "
            "--vae-patch-parallel-size must equal the product of the baseline's parallel sizes. "
            "Does not add extra GPUs — reuses GPUs already allocated by the baseline."
        ),
    },
    "fp8": {
        "args": ["--quantization", "fp8"],
        "gpu_multiplier": 1,
        "lossy": True,
        "label": "FP8 Quantization",
        "category": "memory",
        "note": "FP8 W8A8 on Ada/Hopper GPUs, weight-only on older hardware.",
    },
    "gguf": {
        "args": ["--quantization", "gguf"],
        "gpu_multiplier": 1,
        "lossy": True,
        "label": "GGUF Quantization",
        "category": "memory",
        "note": "Native GGUF transformer-only weights (Q4, Q8, etc.). Not compatible with fp8.",
    },
    # ── Extensions ────────────────────────────────────────────────────────────
    "lora": {
        "args": ["--lora-path", ""],
        "gpu_multiplier": 1,
        "lossy": False,
        "label": "LoRA Inference",
        "category": "extension",
        "note": (
            "Low-Rank Adaptation inference. Requires --lora-path to a valid adapter directory. "
            "Override the empty string via: FEATURE_REGISTRY['lora']['args'][1] = '/path/to/adapter'"
        ),
    },
}

# ── Conflict rules ────────────────────────────────────────────────────────────
#
# CONFLICT_RULES: pairwise feature combinations that must never run together.
# Each entry is (feature_a, feature_b, reason).
# A config is SKIPPED (not failed) when both features are active simultaneously.
#
# SINGLE_CARD_ONLY: features restricted to a single GPU.
# They conflict with any feature whose gpu_multiplier > 1.

CONFLICT_RULES: list[tuple[str, str, str]] = [
    (
        "tp",
        "hsdp",
        "Tensor Parallel and HSDP are not compatible",
    ),
    (
        "teacache",
        "cache_dit",
        "TeaCache and Cache-DiT are not compatible",
    ),
    (
        "layerwise_offload",
        "cpu_offload",
        "Layerwise CPU offloading and module-level CPU offloading are not compatible",
    ),
    (
        "fp8",
        "gguf",
        "FP8 quantization and GGUF quantization are not compatible",
    ),
]

# Features that only support single-card execution.
# They implicitly conflict with every feature whose gpu_multiplier > 1.
SINGLE_CARD_ONLY: frozenset[str] = frozenset({"layerwise_offload"})


def check_conflicts(features: list[str]) -> str | None:
    """Return a human-readable conflict reason if the feature combination is invalid.

    Returns None when the combination is valid.
    Checks:
      1. Pairwise incompatibilities defined in CONFLICT_RULES.
      2. Single-card-only features combined with any multi-GPU feature.
    """
    feature_set = set(features)

    for fa, fb, reason in CONFLICT_RULES:
        if fa in feature_set and fb in feature_set:
            return reason

    for f in features:
        if f in SINGLE_CARD_ONLY:
            multi_gpu = [g for g in features if g != f and FEATURE_REGISTRY[g]["gpu_multiplier"] > 1]
            if multi_gpu:
                return f"'{f}' supports single-card only and cannot be combined with multi-GPU feature(s): {multi_gpu}"

    return None


NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, noise"

# Default prompt file bundled alongside this script
_DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts.txt"


def load_prompts(prompt_file: str | Path) -> list[str]:
    """Load prompts from a text file (one prompt per line, blank lines ignored)."""
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip()]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts


# ── Helper utilities ──────────────────────────────────────────────────────────


def _log(msg: str, level: str = "INFO") -> None:
    colors = {
        "INFO": "\033[36m",
        "OK": "\033[32m",
        "WARN": "\033[33m",
        "FAIL": "\033[31m",
    }
    reset = "\033[0m"
    c = colors.get(level, "")
    print(f"{c}[{level:4s}]{reset}  {msg}")


def get_gpu_count() -> int:
    """Return the number of CUDA GPUs visible to PyTorch."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


# ── Config builder ────────────────────────────────────────────────────────────


def build_configs(baseline_feature: str, addons: list[str]) -> list[dict]:
    """Return the ordered list of test configs for the given baseline + addons.

    Parameters
    ----------
    baseline_feature:
        Key from FEATURE_REGISTRY used as the primary feature under test.
        Must not be an ``addon_only`` feature (e.g. vae_patch_parallel).
    addons:
        Zero or more additional feature keys to stack on top of the baseline.

    Returns
    -------
    list[dict]  ordered: baseline1 → baseline2 → addon-1 … addon-N
    """
    unknown = [f for f in [baseline_feature, *addons] if f not in FEATURE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}\nAvailable: {sorted(FEATURE_REGISTRY)}")

    if FEATURE_REGISTRY[baseline_feature].get("addon_only"):
        addon_only_features = [k for k, v in FEATURE_REGISTRY.items() if v.get("addon_only")]
        raise ValueError(
            f"'{baseline_feature}' is addon-only and cannot be used as --baseline-feature.\n"
            f"Addon-only features: {addon_only_features}\n"
            f"Use it in --addons instead, e.g.:\n"
            f"  --baseline-feature tp --addons {baseline_feature}"
        )

    bf = FEATURE_REGISTRY[baseline_feature]
    configs: list[dict] = []

    # Baseline 1 — no features (never conflicts)
    configs.append(
        {
            "name": "baseline",
            "role": "baseline1",
            "features": [],
            "args": [],
            "gpu_req": 1,
            "lossy": False,
            "skip_reason": None,
        }
    )

    # Baseline 2 — baseline feature only
    configs.append(
        {
            "name": baseline_feature,
            "role": "baseline2",
            "features": [baseline_feature],
            "args": list(bf["args"]),
            "gpu_req": bf["gpu_multiplier"],
            "lossy": bf["lossy"],
            "skip_reason": check_conflicts([baseline_feature]),
        }
    )

    # Addons — baseline feature + one addon each
    for addon in addons:
        af = FEATURE_REGISTRY[addon]
        combo = [baseline_feature, addon]
        gpu_req = bf["gpu_multiplier"] * af["gpu_multiplier"]
        configs.append(
            {
                "name": f"{baseline_feature}+{addon}",
                "role": "addon",
                "features": combo,
                "args": list(bf["args"]) + list(af["args"]),
                "gpu_req": gpu_req,
                "lossy": bf["lossy"] or af["lossy"],
                "skip_reason": check_conflicts(combo),
            }
        )

    return configs


# ── I/O helpers ───────────────────────────────────────────────────────────────


def write_manifest(
    row_dir: Path,
    baseline_feature: str,
    addons: list[str],
    configs: list[dict],
    args: argparse.Namespace,
    gpu_count: int,
) -> None:
    manifest = {
        "baseline_feature": baseline_feature,
        "addons": addons,
        "model": args.model,
        "output_dir": str(row_dir),
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_prompts": args.num_prompts,
        "gpu_count": gpu_count,
        "configs": [c["name"] for c in configs],
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = row_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written → {manifest_path}")


def write_config_meta(
    cfg_dir: Path,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    meta = {
        "name": cfg["name"],
        "role": cfg["role"],
        "features": cfg["features"],
        "extra_args": " ".join(cfg["args"]),
        "total_gpus": cfg["gpu_req"],
        "is_lossy": cfg["lossy"],
        "model": args.model,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_prompts": args.num_prompts,
    }
    (cfg_dir / "config_info.json").write_text(json.dumps(meta, indent=2))


# ── Per-prompt runner ─────────────────────────────────────────────────────────


def run_batch_prompts(
    batch_script: Path,
    cfg: dict,
    prompt_file: Path,
    num_prompts: int,
    cfg_dir: Path,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Run batch_text_to_image.py for all prompts; return (n_ok, n_fail)."""
    # Use cfg_dir as output directory for batch generation
    cmd = [
        sys.executable,
        str(batch_script),
        "--model",
        args.model,
        "--prompt-file",
        str(prompt_file),
        "--negative-prompt",
        f"'{NEGATIVE_PROMPT}'",
        "--output",
        str(cfg_dir),
        "--num-inference-steps",
        str(args.steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--cfg-scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        *cfg["args"],
    ]

    print(f"  Running batch generation for {num_prompts} prompts...", flush=True)

    if dry_run:
        print(f"    DRY-RUN: {' '.join(cmd)}")
        return num_prompts, 0

    # Run batch generation, streaming output to screen and log file simultaneously.
    # start_new_session=True puts the subprocess in its own process group so that
    # when the main batch script exits (e.g. due to a worker crash), we can send
    # SIGTERM to the entire group and unblock the stdout pipe read loop.
    t0 = time.monotonic()
    output_chunks: list[bytes] = []
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as proc:
        pgid = os.getpgid(proc.pid)
        assert proc.stdout is not None

        def _reap_group() -> None:
            """Kill the process group once the main process exits."""
            proc.wait()
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        reaper = threading.Thread(target=_reap_group, daemon=True)
        reaper.start()

        try:
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                output_chunks.append(line)
        finally:
            # Ensure the whole process group is cleaned up (covers KeyboardInterrupt
            # and the normal case where main proc exited but workers linger).
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        reaper.join(timeout=10)
        rc = proc.returncode if proc.returncode is not None else proc.wait()
    elapsed_ms = (time.monotonic() - t0) * 1_000

    # Save batch log
    batch_log_path = cfg_dir / "batch_generation.log"
    batch_log_path.write_bytes(b"".join(output_chunks))

    # Save batch exit code
    batch_rc_path = cfg_dir / "batch_generation.exitcode"
    batch_rc_path.write_text(str(rc))

    if rc != 0:
        _log(f"BATCH FAIL (rc={rc})", "FAIL")
        return 0, num_prompts

    # Rename generated images from image_NNNN.png to prompt_NN.png
    # and create minimal metadata files for compatibility with analyze script
    n_ok = 0
    n_fail = 0
    for idx in range(num_prompts):
        src_img = cfg_dir / f"image_{idx:04d}.png"
        dst_img = cfg_dir / f"prompt_{idx:02d}.png"
        rc_path = cfg_dir / f"prompt_{idx:02d}.exitcode"

        if src_img.exists():
            src_img.rename(dst_img)
            rc_path.write_text("0")
            n_ok += 1
        else:
            rc_path.write_text("1")
            n_fail += 1

    _log(f"Batch completed: {n_ok} OK / {n_fail} FAIL in {elapsed_ms:.0f}ms", "OK" if n_fail == 0 else "WARN")

    return n_ok, n_fail


# ── Per-config runner ─────────────────────────────────────────────────────────


def run_config(
    batch_script: Path,
    cfg: dict,
    prompt_file: Path,
    num_prompts: int,
    cfg_dir: Path,
    args: argparse.Namespace,
    dry_run: bool,
) -> tuple[int, int]:
    """Run all prompts for one config using batch generation; return (n_ok, n_fail)."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    write_config_meta(cfg_dir, cfg, args)

    sep = "─" * 72
    print(f"\n{sep}")
    _log(f"Config : {cfg['name']}  [{cfg['role']}]")
    _log(f"Args   : {' '.join(cfg['args']) or '(none)'}")
    _log(f"GPUs   : {cfg['gpu_req']}  |  Lossy: {cfg['lossy']}")
    print(sep)

    # Use batch processing instead of per-prompt processing
    n_ok, n_fail = run_batch_prompts(batch_script, cfg, prompt_file, num_prompts, cfg_dir, args, dry_run)

    _log(f"Config '{cfg['name']}': {n_ok} OK / {n_fail} FAIL / {num_prompts} total")
    return n_ok, n_fail


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Feature compatibility test runner — Python replacement for run_compat_test.sh.\n\n"
            "Runs text_to_image.py with a structured two-level matrix:\n"
            "  Baseline 1 = no features,  Baseline 2 = --baseline-feature alone,\n"
            "  Addons = Baseline-2 + each feature in --addons."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available features for --baseline-feature / --addons:\n"
            + "\n".join(f"  {name:<14} {info['label']}" for name, info in FEATURE_REGISTRY.items())
            + "\n\nExample:\n"
            "  python run_compat_test.py \\\n"
            "      --baseline-feature cfg_parallel \\\n"
            "      --addons teacache cache_dit ulysses ring \\\n"
            "      --model Qwen/Qwen-Image-2512 \\\n"
            "      --output-dir ./compat_results \\\n"
            "      --steps 30 --num-prompts 20\n\n"
            "  python analyze_compat_results.py \\\n"
            "      --results-dir ./compat_results/cfg_parallel --charts"
        ),
    )
    p.add_argument(
        "--baseline-feature",
        default="cfg_parallel",
        choices=sorted(FEATURE_REGISTRY),
        metavar="FEATURE",
        help=(
            "Primary feature under test.  "
            "Baseline-1 = no features; Baseline-2 = this feature alone.  "
            f"(default: cfg_parallel; choices: {sorted(FEATURE_REGISTRY)})"
        ),
    )
    p.add_argument(
        "--addons",
        nargs="*",
        default=[],
        choices=sorted(FEATURE_REGISTRY),
        metavar="FEATURE",
        help=(
            "Zero or more features stacked on top of --baseline-feature "
            "(e.g. teacache cache_dit ulysses ring). "
            "Each creates one additional config: <baseline>+<addon>."
        ),
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2512",
        help="HuggingFace model ID or local path.",
    )
    p.add_argument(
        "--output-dir",
        default="compat_results",
        help="Root directory for images and logs.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion denoising steps.",
    )
    p.add_argument("--height", type=int, default=1024, help="Image height in pixels.")
    p.add_argument("--width", type=int, default=1024, help="Image width in pixels.")
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale forwarded to text_to_image.py.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--prompt-file",
        default=str(_DEFAULT_PROMPT_FILE),
        help=("Path to a text file with one prompt per line. (default: prompts.txt next to this script)"),
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to use (taken from the top of --prompt-file).",
    )
    p.add_argument(
        "--lora-path",
        default="",
        metavar="PATH",
        help=(
            "Path to a LoRA adapter directory.  Required when --baseline-feature lora "
            "or lora is listed in --addons.  Automatically injected into the 'lora' "
            "feature's args at runtime."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    # Inject --lora-path into the lora feature args at runtime
    if args.lora_path:
        FEATURE_REGISTRY["lora"]["args"] = ["--lora-path", args.lora_path]
    elif "lora" in ([args.baseline_feature] + (args.addons or [])):
        print(
            "[WARN] 'lora' feature selected but --lora-path is empty. "
            "Pass --lora-path /path/to/adapter or the lora run will fail.",
            file=sys.stderr,
        )

    # Locate batch_text_to_image.py in the same directory as this script
    script_dir = Path(__file__).resolve().parent
    batch_script = script_dir / "batch_text_to_image.py"
    if not batch_script.exists():
        print(f"[ERROR] batch_text_to_image.py not found at: {batch_script}", file=sys.stderr)
        return 1

    # Validate and build config matrix
    try:
        configs = build_configs(args.baseline_feature, args.addons or [])
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        all_prompts = load_prompts(args.prompt_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    num_prompts = max(1, min(args.num_prompts, len(all_prompts)))
    prompts = all_prompts[:num_prompts]
    gpu_count = get_gpu_count()
    row_dir = Path(args.output_dir) / args.baseline_feature
    row_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary prompt file with only the prompts we need
    # This file will be shared across all configs to avoid redundant file creation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        for prompt in prompts:
            tf.write(prompt + "\n")
        temp_prompt_file = Path(tf.name)

    try:
        # Header
        print()
        print("=" * 72)
        print("  Feature Compatibility Test (Batch Mode)")
        print(f"  Baseline feature : {args.baseline_feature}")
        print(f"  Addons           : {args.addons or '(none)'}")
        print(f"  Model            : {args.model}")
        print(f"  GPUs available   : {gpu_count}")
        print(f"  Configs          : {len(configs)}")
        print(f"  Prompts          : {num_prompts}")
        print(f"  Steps            : {args.steps}  |  {args.width}×{args.height}")
        print(f"  Output dir       : {row_dir}")
        print("=" * 72)

        write_manifest(row_dir, args.baseline_feature, args.addons or [], configs, args, gpu_count)

        total_ok = total_fail = 0
        total_skip_conflict = total_skip_gpu = 0

        for cfg in configs:
            # ── Skip: known feature conflict ──────────────────────────────────
            if cfg["skip_reason"]:
                _log(
                    f"SKIP '{cfg['name']}' — {cfg['skip_reason']}",
                    "WARN",
                )
                total_skip_conflict += 1
                continue

            # ── Skip: insufficient GPUs ───────────────────────────────────────
            if cfg["gpu_req"] > gpu_count:
                _log(
                    f"SKIP '{cfg['name']}' — requires {cfg['gpu_req']} GPUs, only {gpu_count} available",
                    "WARN",
                )
                total_skip_gpu += 1
                continue

            cfg_dir = row_dir / cfg["name"]
            n_ok, n_fail = run_config(batch_script, cfg, temp_prompt_file, num_prompts, cfg_dir, args, args.dry_run)
            total_ok += n_ok
            total_fail += n_fail
    finally:
        # Clean up temporary prompt file
        temp_prompt_file.unlink(missing_ok=True)

    # Final summary
    total_skip = total_skip_conflict + total_skip_gpu
    print()
    print("=" * 72)
    print("  RUN COMPLETE")
    print(f"  Baseline feature : {args.baseline_feature}")
    print(f"  OK               : {total_ok}")
    print(f"  FAIL             : {total_fail}")
    print(f"  SKIP (conflict)  : {total_skip_conflict}  (incompatible feature pairs)")
    print(f"  SKIP (GPU)       : {total_skip_gpu}  (insufficient GPUs)")
    print(f"  SKIP total       : {total_skip}  (configs, not prompts)")
    print(f"  Output           : {row_dir}")
    print("=" * 72)
    print()
    print("Next step — run the analysis script:")
    print(f"  python analyze_compat_results.py --results-dir {row_dir} --charts")
    print()

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
