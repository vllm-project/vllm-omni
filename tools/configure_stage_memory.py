#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interactive tool to configure multi-stage TTS/Omni pipelines.

Detects GPUs, shows available memory, and helps configure:
  - GPU device assignment per stage
  - gpu_memory_utilization per stage
  - async_chunk (streaming vs non-streaming)
  - enforce_eager vs CUDA graph compilation
  - max_batch_size per stage

Usage:
    python tools/configure_stage_memory.py --config qwen3_tts.yaml
    python tools/configure_stage_memory.py --config qwen3_tts.yaml --auto
    python tools/configure_stage_memory.py --config qwen3_tts.yaml --auto --streaming
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf


def get_model_size_gib(model: str) -> float | None:
    """Get model weight size in GiB from HuggingFace model info."""
    try:
        from huggingface_hub import model_info

        info = model_info(model)
        if info.safetensors and info.safetensors.total:
            # params * 2 bytes (bf16)
            return info.safetensors.total * 2 / (1024**3)
    except Exception:
        pass
    return None


def get_gpu_info() -> list[dict]:
    """Detect GPUs and return their memory info."""
    gpus = []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        for i in range(torch.accelerator.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "id": i,
                    "name": props.name,
                    "total_gib": total / (1024**3),
                    "free_gib": free / (1024**3),
                    "used_gib": (total - free) / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}", file=sys.stderr)
    return gpus


def print_gpu_table(gpus: list[dict]) -> None:
    """Print GPU info table."""
    print("\n  Available GPUs:")
    print(f"  {'ID':>3}  {'Name':<30}  {'Free':>8}  {'Total':>8}  {'Used':>8}  {'CC':>5}")
    print(f"  {'---':>3}  {'----':<30}  {'----':>8}  {'-----':>8}  {'----':>8}  {'--':>5}")
    for g in gpus:
        print(
            f"  {g['id']:>3}  {g['name']:<30}  {g['free_gib']:>7.1f}G"
            f"  {g['total_gib']:>7.1f}G  {g['used_gib']:>7.1f}G"
            f"  {g['compute_capability']:>5}"
        )
    print()


def print_config_summary(config: dict, stages: list[dict]) -> None:
    """Print full config summary."""
    async_chunk = config.get("async_chunk", False)
    print(f"  async_chunk: {async_chunk}")
    print(
        f"  {'Stage':>5}  {'Model Stage':<15}  {'Device':>6}  {'GPU Mem':>8}"
        f"  {'Eager':>6}  {'Async Sched':>11}  {'Batch':>5}"
    )
    print(
        f"  {'-----':>5}  {'-----------':<15}  {'------':>6}  {'-------':>8}"
        f"  {'-----':>6}  {'-----------':>11}  {'-----':>5}"
    )
    for s in stages:
        print(
            f"  {s['stage_id']:>5}  {s['model_stage']:<15}  {s['device']:>6}"
            f"  {s['gpu_mem']:>7.3f}  {'yes' if s['enforce_eager'] else 'no':>6}"
            f"  {'yes' if s['async_scheduling'] else 'no':>11}"
            f"  {s['max_batch_size']:>5}"
        )
    print()


def extract_stages(config: dict) -> list[dict]:
    """Extract stage info from config."""
    stages = []
    for stage_arg in config.get("stage_args", []):
        ea = stage_arg.get("engine_args", {})
        rt = stage_arg.get("runtime", {})
        stages.append(
            {
                "stage_id": stage_arg.get("stage_id", 0),
                "stage_type": stage_arg.get("stage_type", "llm"),
                "model_stage": ea.get("model_stage", "unknown"),
                "device": str(rt.get("devices", "0")),
                "gpu_mem": ea.get("gpu_memory_utilization", 0.9),
                "enforce_eager": ea.get("enforce_eager", False),
                "async_scheduling": ea.get("async_scheduling", False),
                "max_batch_size": rt.get("max_batch_size", 1),
                "worker_type": ea.get("worker_type", "ar"),
            }
        )
    return stages


def auto_configure(
    config: dict,
    stages: list[dict],
    gpus: list[dict],
    headroom_gib: float = 1.5,
    model_size_gib: float | None = None,
    streaming: bool | None = None,
    latency_optimized: bool = False,
) -> tuple[dict, list[dict]]:
    """Auto-configure all settings."""
    # async_chunk
    if streaming is not None:
        config["async_chunk"] = streaming

    # GPU memory: use model size to compute what's actually needed,
    # capped by available memory.
    device_stages: dict[str, list[int]] = {}
    for i, s in enumerate(stages):
        device_stages.setdefault(s["device"], []).append(i)

    for device, indices in device_stages.items():
        # Handle multi-device strings like "0,1,2,3" (tensor-parallel).
        # Use the first device for memory query; skip auto-sizing if invalid.
        try:
            gpu_id = int(device.split(",")[0])
        except ValueError:
            continue
        if gpu_id >= len(gpus):
            continue
        gpu = gpus[gpu_id]
        num = len(indices)

        # What's available per stage
        avail_per_stage = (gpu["free_gib"] - headroom_gib) / max(num, 1)

        # What the model actually needs per stage (weights + KV cache headroom)
        if model_size_gib is not None:
            needed_per_stage = model_size_gib / max(num, 1) + 3.0  # +3G for KV cache
        else:
            needed_per_stage = avail_per_stage  # no model info, use all available

        # Take the smaller of available and needed
        allocated = min(avail_per_stage, needed_per_stage)
        util = round(max(allocated / gpu["total_gib"], 0.04), 3)
        util = min(util, 0.95)

        for idx in indices:
            stages[idx]["gpu_mem"] = util

    # Per-stage optimizations
    for s in stages:
        if latency_optimized:
            # CUDA graphs for AR stages (lower latency)
            if s["worker_type"] == "ar":
                s["enforce_eager"] = False
                s["async_scheduling"] = True
            # Generation stages always eager (no KV cache)
            if s["worker_type"] == "generation":
                s["enforce_eager"] = True
                s["async_scheduling"] = True

    return config, stages


def interactive_configure(config: dict, stages: list[dict], gpus: list[dict]) -> tuple[dict, list[dict]]:
    """Interactive mode."""
    gpu_ids = [str(g["id"]) for g in gpus]

    # async_chunk
    current_async = config.get("async_chunk", False)
    val = input(f"  Enable streaming (async_chunk)? [{'Y' if current_async else 'N'}]: ").strip().lower()
    if val in ("y", "yes", "true", "1"):
        config["async_chunk"] = True
    elif val in ("n", "no", "false", "0"):
        config["async_chunk"] = False
    print()

    for s in stages:
        print(f"  Stage {s['stage_id']} ({s['model_stage']}, {s['worker_type']}):")

        # Device (accepts single id or comma-separated like "0,1,2")
        default_dev = s["device"]
        while True:
            dev = input(f"    GPU device [{default_dev}]: ").strip() or default_dev
            dev_ids = [d.strip() for d in dev.split(",")]
            if all(d in gpu_ids for d in dev_ids):
                s["device"] = dev
                break
            print(f"    Invalid. Choose from: {', '.join(gpu_ids)} (comma-separated for multi-GPU)")

        # GPU memory (use first device for memory query)
        first_dev = int(s["device"].split(",")[0])
        gpu = gpus[first_dev]
        same_device = sum(1 for st in stages if st["device"] == s["device"])
        suggested = round((gpu["free_gib"] - 1.5) / same_device / gpu["total_gib"], 3)
        suggested = max(suggested, 0.04)
        suggested = min(suggested, 0.95)
        while True:
            val = input(f"    gpu_memory_utilization [{suggested:.3f}]: ").strip()
            if not val:
                s["gpu_mem"] = suggested
                break
            try:
                v = float(val)
                if 0.01 <= v <= 0.99:
                    s["gpu_mem"] = round(v, 3)
                    break
                print("    Must be between 0.01 and 0.99")
            except ValueError:
                print("    Invalid number")

        # enforce_eager
        if s["worker_type"] == "ar":
            current = s["enforce_eager"]
            hint = "no=CUDA graphs (faster), yes=eager (debug)" if not current else "yes=eager, no=CUDA graphs (faster)"
            val = input(f"    enforce_eager [{('yes' if current else 'no')}] ({hint}): ").strip().lower()
            if val in ("y", "yes", "true", "1"):
                s["enforce_eager"] = True
            elif val in ("n", "no", "false", "0"):
                s["enforce_eager"] = False

        # max_batch_size
        current_bs = s["max_batch_size"]
        val = input(f"    max_batch_size [{current_bs}]: ").strip()
        if val:
            try:
                s["max_batch_size"] = int(val)
            except ValueError:
                pass

        print()

    return config, stages


def apply_to_config(config: dict, stages: list[dict]) -> dict:
    """Apply stage settings back to config dict."""
    config = copy.deepcopy(config)
    for stage_arg, s in zip(config["stage_args"], stages):
        stage_arg.setdefault("runtime", {})["devices"] = s["device"]
        stage_arg.setdefault("runtime", {})["max_batch_size"] = s["max_batch_size"]
        ea = stage_arg.setdefault("engine_args", {})
        ea["gpu_memory_utilization"] = s["gpu_mem"]
        ea["enforce_eager"] = s["enforce_eager"]
        ea["async_scheduling"] = s["async_scheduling"]
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Configure multi-stage TTS/Omni pipelines for your hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - prompts for every setting
  python tools/configure_stage_memory.py --config qwen3_tts.yaml

  # Auto mode - detect GPUs and set optimal values
  python tools/configure_stage_memory.py --config qwen3_tts.yaml --auto

  # Auto mode optimized for low latency with streaming
  python tools/configure_stage_memory.py --config qwen3_tts.yaml --auto --streaming --low-latency

  # Save to a different file
  python tools/configure_stage_memory.py --config qwen3_tts.yaml --auto -o my_config.yaml
""",
    )
    parser.add_argument("--config", required=True, help="Path to stage config YAML")
    parser.add_argument("--model", help="HuggingFace model name (to query weight size for smart allocation)")
    parser.add_argument("--auto", action="store_true", help="Auto-configure without prompts")
    parser.add_argument("--output", "-o", help="Output path (default: overwrite input)")
    parser.add_argument("--headroom", type=float, default=1.5, help="GiB headroom to leave free (default: 1.5)")
    parser.add_argument("--streaming", action="store_true", default=None, help="Enable async_chunk streaming")
    parser.add_argument("--no-streaming", action="store_true", help="Disable async_chunk streaming")
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="Optimize for latency (CUDA graphs for AR, async scheduling)",
    )
    args = parser.parse_args()

    streaming = None
    if args.streaming:
        streaming = True
    elif args.no_streaming:
        streaming = False

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    stages = extract_stages(config)
    gpus = get_gpu_info()

    if not gpus:
        print("No GPUs detected. Cannot configure.", file=sys.stderr)
        sys.exit(1)

    # Query model size from HuggingFace
    model_size_gib = None
    if args.model:
        model_size_gib = get_model_size_gib(args.model)
        if model_size_gib:
            print(f"\n  Model: {args.model} ({model_size_gib:.1f} GiB in bf16)")
        else:
            print(f"\n  Model: {args.model} (could not determine size)")
    else:
        print("\n  Tip: pass --model <name> to auto-size based on HuggingFace weight info")

    print(f"  Config: {config_path}")
    print_gpu_table(gpus)
    print("  Before:")
    print_config_summary(config, stages)

    if args.auto:
        config, stages = auto_configure(
            config, stages, gpus, args.headroom, model_size_gib, streaming, args.low_latency
        )
    else:
        config, stages = interactive_configure(config, stages, gpus)

    config = apply_to_config(config, stages)

    print("  After:")
    print_config_summary(config, extract_stages(config))

    output_path = Path(args.output) if args.output else config_path
    if output_path == config_path:
        backup = config_path.with_suffix(".yaml.bak")
        shutil.copy2(config_path, backup)
        print(f"  Backup: {backup}")

    OmegaConf.save(OmegaConf.create(config), output_path)
    print(f"  Saved:  {output_path}")


if __name__ == "__main__":
    main()
