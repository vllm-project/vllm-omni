#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interactive tool to configure multi-stage TTS/Omni pipelines.

Detects GPUs, shows available memory, and helps configure:
  - GPU device assignment per stage
  - gpu_memory_utilization per stage
  - async_chunk (streaming vs non-streaming)
  - enforce_eager vs CUDA graph compilation
  - max_num_seqs per stage

The config is a deploy YAML (``vllm_omni/deploy/*.yaml``) with a top-level
``stages:`` list.

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


def _fmt(value: object, spec: str = "") -> str:
    """Format a stage value, showing unset (model default) values as ``-``."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(value, spec)


def print_config_summary(config: dict, stages: list[dict]) -> None:
    """Print full config summary."""
    async_chunk = config.get("async_chunk", False)
    print(f"  async_chunk: {async_chunk}")
    print(
        f"  {'Stage':>5}  {'Model Stage':<15}  {'Device':>6}  {'GPU Mem':>8}"
        f"  {'Eager':>6}  {'Async Sched':>11}  {'Seqs':>5}"
    )
    print(
        f"  {'-----':>5}  {'-----------':<15}  {'------':>6}  {'-------':>8}"
        f"  {'-----':>6}  {'-----------':>11}  {'-----':>5}"
    )
    for s in stages:
        print(
            f"  {s['stage_id']:>5}  {s['model_stage']:<15}  {s['device']:>6}"
            f"  {_fmt(s['gpu_mem'], '.3f'):>8}  {_fmt(s['enforce_eager']):>6}"
            f"  {_fmt(s['async_scheduling']):>11}"
            f"  {_fmt(s['max_num_seqs']):>5}"
        )
    print()


def _get_stage_value(stage: dict, key: str, default: object = None) -> object:
    """Read an engine knob with the precedence ``load_deploy_config`` uses."""
    engine_args = stage.get("engine_args") or {}
    if key in engine_args:
        return engine_args[key]
    return stage.get(key, default)


def _set_stage_value(stage: dict, key: str, value: object) -> None:
    """Write an engine knob where ``load_deploy_config`` will read it."""
    engine_args = stage.get("engine_args")
    if isinstance(engine_args, dict) and key in engine_args:
        engine_args[key] = value
    else:
        stage[key] = value


def extract_stages(config: dict) -> list[dict]:
    """Extract stage info from a deploy config.

    Knobs a stage does not set are kept as ``None`` so they are not written
    back and the stage keeps its model / vLLM default.
    """
    if "stage_args" in config:
        raise ValueError(
            "This config uses the removed `stage_args` schema, which vLLM-Omni no "
            "longer loads. Use a deploy YAML with a top-level `stages:` list "
            "(see vllm_omni/deploy/)."
        )
    stages = []
    for stage in config.get("stages") or []:
        runtime = stage.get("runtime") or {}
        stages.append(
            {
                "stage_id": stage.get("stage_id", 0),
                "model_stage": _get_stage_value(stage, "model_stage", "-"),
                "device": str(runtime.get("devices", stage.get("devices", "0"))),
                "gpu_mem": _get_stage_value(stage, "gpu_memory_utilization"),
                "enforce_eager": _get_stage_value(stage, "enforce_eager"),
                "async_scheduling": _get_stage_value(stage, "async_scheduling"),
                "max_num_seqs": _get_stage_value(stage, "max_num_seqs"),
                "worker_type": _get_stage_value(stage, "worker_type"),
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

        # enforce_eager (deploy YAMLs do not record the worker type, so ask
        # for every stage unless it is known to be a generation stage)
        if s["worker_type"] in ("ar", None):
            current = s["enforce_eager"]
            hint = "no=CUDA graphs (faster), yes=eager (debug)" if not current else "yes=eager, no=CUDA graphs (faster)"
            val = input(f"    enforce_eager [{('yes' if current else 'no')}] ({hint}): ").strip().lower()
            if val in ("y", "yes", "true", "1"):
                s["enforce_eager"] = True
            elif val in ("n", "no", "false", "0"):
                s["enforce_eager"] = False

        # max_num_seqs
        val = input(f"    max_num_seqs [{_fmt(s['max_num_seqs'])}]: ").strip()
        if val:
            try:
                s["max_num_seqs"] = int(val)
            except ValueError:
                pass

        print()

    return config, stages


def apply_to_config(config: dict, stages: list[dict]) -> dict:
    """Apply stage settings back to config dict."""
    config = copy.deepcopy(config)
    for stage, s in zip(config["stages"], stages):
        runtime = stage.get("runtime")
        if isinstance(runtime, dict) and "devices" in runtime:
            runtime["devices"] = s["device"]
        else:
            stage["devices"] = s["device"]
        for key, value in (
            ("gpu_memory_utilization", s["gpu_mem"]),
            ("enforce_eager", s["enforce_eager"]),
            ("async_scheduling", s["async_scheduling"]),
            ("max_num_seqs", s["max_num_seqs"]),
        ):
            if value is not None:
                _set_stage_value(stage, key, value)
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
    try:
        stages = extract_stages(config)
    except ValueError as e:
        print(f"Error: {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
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
