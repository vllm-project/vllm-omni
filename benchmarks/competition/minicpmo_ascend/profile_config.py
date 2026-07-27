#!/usr/bin/env python3
"""Generate a MiniCPM-o deployment config with NPU profiling enabled."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def build_profile_config(
    config: dict[str, Any],
    *,
    trace_dir: Path,
    stages: set[int],
    with_stack: bool = False,
    with_memory: bool = False,
) -> dict[str, Any]:
    stage_configs = config.get("stages")
    if not isinstance(stage_configs, list):
        raise ValueError("deployment config must contain a stages list")

    available = {int(stage["stage_id"]) for stage in stage_configs}
    missing = stages - available
    if missing:
        raise ValueError(f"profile stages are not in deployment config: {sorted(missing)}")

    for stage in stage_configs:
        stage_id = int(stage["stage_id"])
        stage.pop("profiler_config", None)
        if stage_id not in stages:
            continue
        stage["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir.resolve()),
            "torch_profiler_use_gzip": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_stack": with_stack,
            "torch_profiler_with_memory": with_memory,
            "torch_profiler_with_flops": False,
            "torch_profiler_dump_cuda_time_total": False,
            "delay_iterations": 0,
            "max_iterations": 0,
            "wait_iterations": 0,
            "warmup_iterations": 0,
            "active_iterations": 1,
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--with-memory", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    generated = build_profile_config(
        config,
        trace_dir=args.trace_dir,
        stages=set(args.stages),
        with_stack=args.with_stack,
        with_memory=args.with_memory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(generated, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
