#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare WaveServe Wan ``serial`` vs ``latest`` on one Omni deploy.

Uses ``vllm_omni/deploy/waveserve_wan.yaml`` (same path as the offline example).
Only ``extra_args.chunk_schedule`` differs between regimes.

    python benchmarks/noisy_pp/waveserve_serial_vs_latest.py \
        --model /data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev \
        --repeat 3 --json-out outputs/bench_latest_kv.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

_MODEL = "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev"
_DEFAULT_DEPLOY = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy" / "waveserve_wan.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=_DEFAULT_DEPLOY)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override PP (= S*G). Default: value from deploy YAML (or denoise_steps+1).",
    )
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--denoise-steps", type=int, default=3)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=3, help="Timed requests per regime.")
    parser.add_argument("--regimes", default="serial,latest")
    parser.add_argument("--gpus-per-stage", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _dit_ms(output: Any) -> float | None:
    metrics = getattr(output, "metrics", None) or {}
    stage_metrics = metrics.get("stage_metrics") or {}
    if not stage_metrics:
        flat = metrics.get("stage_gen_time_ms")
        return float(flat) if flat is not None else None
    first = next(iter(stage_metrics.values()))
    ms = float((first or {}).get("stage_gen_time_ms") or 0.0)
    return ms if ms > 0 else None


def _resolve_deploy(args: argparse.Namespace, world: int) -> Path:
    """Return deploy path; rewrite PP/stage size only when overriding world-size."""
    base = Path(args.deploy_config).expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(f"deploy config not found: {base}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(base)
    stage0 = cfg.stages[0]
    current_pp = int(OmegaConf.select(stage0, "parallel_config.pipeline_parallel_size") or 1)
    current_s = int(
        OmegaConf.select(stage0, "model_config.ar_diffusion_stage_config.stage_parallel_size") or current_pp
    )
    stages = args.denoise_steps + 1
    layer_groups = args.gpus_per_stage
    if stages * layer_groups != world:
        raise SystemExit(f"need stages*layer_groups == world; got {stages}*{layer_groups} != {world}")

    current_history = int(
        OmegaConf.select(stage0, "model_config.ar_diffusion_stage_config.max_history_chunks") or args.history
    )
    if current_pp == world and current_s == stages and args.history == current_history:
        return base

    OmegaConf.update(stage0, "parallel_config.pipeline_parallel_size", world, force_add=True)
    OmegaConf.update(stage0, "model_config.ar_diffusion_stage_config.stage_parallel_size", stages, force_add=True)
    OmegaConf.update(
        stage0,
        "model_config.ar_diffusion_stage_config.max_history_chunks",
        max(1, args.history),
        force_add=True,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="waveserve_wan_pp_", delete=False) as fh:
        fh.write(OmegaConf.to_yaml(cfg))
        return Path(fh.name)


def _expected_slots(
    chunks: int,
    denoise_steps: int,
    stages: int,
    layer_groups: int,
    history: int,
    chunk_schedule: str,
) -> int:
    from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
        ChunkSchedule,
        Ordering,
        build_chunk_plan,
    )

    ordering = Ordering.SERIAL if chunk_schedule == "serial" else Ordering.INTERLEAVED
    schedule = ChunkSchedule(
        chunks=chunks,
        num_denoise_steps=denoise_steps,
        stages=stages,
        layer_groups=layer_groups,
        ordering=ordering,
        kv_history_chunks=max(1, history),
    )
    return build_chunk_plan(schedule).num_slots


def _run_regime(omni: Any, args: argparse.Namespace, regime: str, stages: int, layer_groups: int) -> dict:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    chunk_schedule = "serial" if regime == "serial" else "latest"
    if regime not in ("serial", "latest"):
        raise ValueError(f"unknown regime {regime!r}; expected serial or latest")

    request_ms: list[float] = []
    dit_ms: list[float] = []
    for index in range(args.repeat):
        params = OmniDiffusionSamplingParams(
            extra_args={
                "num_chunks": args.chunks,
                "num_denoise_steps": args.denoise_steps,
                "kv_history_chunks": args.history,
                "chunk_schedule": chunk_schedule,
                "reset": True,
            }
        )
        started = time.perf_counter()
        outputs = omni.generate("a cat walking on grass", sampling_params_list=[params])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not outputs:
            raise RuntimeError(f"regime={regime} produced no output")
        request_ms.append(elapsed_ms)
        gen = _dit_ms(outputs[0])
        if gen is None:
            raise RuntimeError(f"regime={regime} request {index} missing stage_gen_time_ms")
        dit_ms.append(gen)

    return {
        "regime": regime,
        "stages": stages,
        "layer_groups": layer_groups,
        "history": args.history,
        "chunk_schedule": chunk_schedule,
        "schedule_slots": _expected_slots(
            args.chunks, args.denoise_steps, stages, layer_groups, args.history, chunk_schedule
        ),
        "timed_iterations": len(request_ms),
        "request_ms": round(_median(request_ms), 3),
        "dit_ms": round(_median(dit_ms), 3),
        "min_request_ms": round(min(request_ms), 3),
        "min_dit_ms": round(min(dit_ms), 3),
        "all_request_ms": [round(x, 3) for x in request_ms],
        "all_dit_ms": [round(x, 3) for x in dit_ms],
        "performance": {
            "request_ms": round(_median(request_ms), 3),
            "dit_ms": round(_median(dit_ms), 3),
        },
    }


def main() -> None:
    args = _parse_args()
    stages = args.denoise_steps + 1
    layer_groups = args.gpus_per_stage
    world = args.world_size or (stages * layer_groups)
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]
    deploy_path = _resolve_deploy(args, world)

    from vllm_omni import Omni

    omni = Omni(model=args.model, deploy_config=str(deploy_path))
    try:
        results = [_run_regime(omni, args, regime, stages, layer_groups) for regime in regimes]
    finally:
        omni.close()

    print("REGIME       S  G  slots  request_ms    dit_ms  min_req  min_dit")
    for row in results:
        print(
            f"{row['regime']:<10} {row['stages']:>2} {row['layer_groups']:>2} "
            f"{row['schedule_slots']:>6} {row['request_ms']:>11.1f} "
            f"{row['dit_ms']:>9.1f} {row['min_request_ms']:>8.1f} {row['min_dit_ms']:>8.1f}"
        )

    by_name = {row["regime"]: row for row in results}
    comparisons: dict[str, dict[str, float]] = {}
    if "serial" in by_name and "latest" in by_name:
        serial, latest = by_name["serial"], by_name["latest"]
        comparisons["latest_vs_serial"] = {
            "request_ms_speedup": round(serial["request_ms"] / latest["request_ms"], 3),
            "dit_ms_speedup": round(serial["dit_ms"] / latest["dit_ms"], 3),
        }
        print(
            "latest vs serial: "
            f"request x{comparisons['latest_vs_serial']['request_ms_speedup']:.2f}, "
            f"dit x{comparisons['latest_vs_serial']['dit_ms_speedup']:.2f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "world": world,
            "deploy_config": str(deploy_path),
            "chunks": args.chunks,
            "denoise_steps": args.denoise_steps,
            "timed_iterations": args.repeat,
            "aggregation": "median over timed requests (no warmup)",
            "runs": results,
            "comparisons": comparisons,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"WROTE_JSON={args.json_out}")


if __name__ == "__main__":
    main()
