# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure MammothModa2 AR, transfer, DiT, and end-to-end latency."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import vllm
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_extras import (
    build_text_to_image_prompt,
    get_model_class_name,
)
from vllm_omni.transformers_utils.configs.mammoth_moda2 import (
    Mammothmoda2Config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "bytedance-research/MammothModa2-Preview"
DEFAULT_PROMPT = "A red cube on a white table"
SCENARIO_CONFIG = {
    "a": "vllm_omni/deploy/mammoth_moda2.yaml",
    "b2": "vllm_omni/deploy/mammoth_moda2_prefix_cache.yaml",
}


def build_request(
    omni: Omni,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[SamplingParams]]:
    prompt_text = " ".join([args.prompt] * args.prompt_repeat)
    request = build_text_to_image_prompt(
        model_class_name=get_model_class_name(omni),
        prompt={"prompt": prompt_text, "modalities": ["image"]},
        height=args.height,
        width=args.width,
    )
    request["additional_information"].update(
        {
            "num_inference_steps": [args.num_inference_steps],
            "text_guidance_scale": [args.text_guidance_scale],
            "cfg_range": [0.0, 1.0],
            "seed": [args.seed],
        }
    )
    info = request["additional_information"]
    ar_width = int(info["ar_width"][0])
    ar_height = int(info["ar_height"][0])
    return request, [
        SamplingParams(
            temperature=0.0,
            top_k=1,
            seed=args.seed,
            max_tokens=ar_height * (ar_width + 1) + 1,
            detokenize=False,
        ),
        SamplingParams(
            temperature=0.0,
            seed=args.seed,
            max_tokens=1,
            detokenize=False,
        ),
    ]


def _stage_metrics(output: Any, stage_id: int) -> dict[str, Any]:
    all_metrics = output.metrics["stage_metrics"]
    stage = all_metrics.get(str(stage_id), all_metrics.get(stage_id))
    if not isinstance(stage, dict):
        raise RuntimeError(f"missing stage-{stage_id} metrics in {all_metrics}")
    return stage


def run_once(
    omni: Omni,
    request: dict[str, Any],
    params: list[SamplingParams],
    hidden_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = omni.generate(
        request,
        sampling_params_list=params,
        use_tqdm=False,
    )[0]
    e2e_ms = (time.perf_counter() - started) * 1000

    ar = _stage_metrics(output, 0)
    dit = _stage_metrics(output, 1)
    durations = output.stage_durations
    ar_ms = float(ar["stage_gen_time_ms"])
    ar_ttft_ms = float(ar["vllm_ttft_ms"])
    dit_ms = float(dit["stage_gen_time_ms"])
    ar2dit_ms = float(
        durations.get(
            "ar2diffusion_ms",
            durations.get("input_processing_0_to_1_ms", 0.0),
        )
    )
    transfer_tx_ms = float(durations.get("transfer_0_to_1_ms", 0.0))
    rx_decode_ms = float(dit.get("rx_decode_time_ms") or 0.0)
    rx_in_flight_ms = float(dit.get("rx_in_flight_time_ms") or 0.0)
    transfer_total_ms = transfer_tx_ms + rx_decode_ms + rx_in_flight_ms
    hidden_state_rows = int(ar["num_tokens_in"]) + int(ar["num_tokens_out"]) - 1

    return {
        "e2e_ms": e2e_ms,
        "ar_total_ms": ar_ms,
        "ar_ttft_ms": ar_ttft_ms,
        "ar_decode_ms": max(0.0, ar_ms - ar_ttft_ms),
        "ar2dit_reconstruction_ms": ar2dit_ms,
        "transfer_tx_ms": transfer_tx_ms,
        "transfer_serialize_submit_ms": max(0.0, transfer_tx_ms - ar2dit_ms),
        "transfer_rx_decode_ms": rx_decode_ms,
        "transfer_in_flight_ms": rx_in_flight_ms,
        "transfer_total_ms": transfer_total_ms,
        "transfer_rx_bytes": int(dit.get("rx_transfer_bytes") or 0),
        "dit_total_ms": dit_ms,
        "orchestration_unattributed_ms": max(
            0.0,
            e2e_ms - ar_ms - transfer_total_ms - dit_ms,
        ),
        "ar2dit_hidden_state_rows": hidden_state_rows,
        "ar2dit_hidden_state_bytes": (hidden_state_rows * hidden_size * torch.float32.itemsize),
        "prompt_tokens": int(ar["num_tokens_in"]),
        "generated_tokens": int(ar["num_tokens_out"]),
        "image_pixels": int(dit.get("image_pixels") or 0),
    }


def validate_sample(sample: dict[str, Any]) -> None:
    required_positive = (
        "ar_total_ms",
        "ar_ttft_ms",
        "ar2dit_reconstruction_ms",
        "transfer_tx_ms",
        "dit_total_ms",
        "ar2dit_hidden_state_bytes",
    )
    missing = [key for key in required_positive if sample[key] <= 0]
    if missing:
        raise RuntimeError(f"incomplete stage attribution ({missing}): {sample}")


def summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    timing_fields = (
        "e2e_ms",
        "ar_total_ms",
        "ar_ttft_ms",
        "ar_decode_ms",
        "ar2dit_reconstruction_ms",
        "transfer_tx_ms",
        "transfer_serialize_submit_ms",
        "transfer_rx_decode_ms",
        "transfer_in_flight_ms",
        "transfer_total_ms",
        "dit_total_ms",
        "orchestration_unattributed_ms",
    )
    return {f"{field}_mean": statistics.mean(float(sample[field]) for sample in samples) for field in timing_fields}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_CONFIG,
        required=True,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--text-guidance-scale",
        type=float,
        default=9.0,
    )
    parser.add_argument("--profile-stage", type=int, choices=(0, 1))
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.prompt_repeat < 1:
        raise ValueError("--iterations and --prompt-repeat must be positive")
    if (args.profile_stage is None) != (args.profile_dir is None):
        raise ValueError("--profile-stage and --profile-dir must be supplied together")

    deploy_config = REPO_ROOT / SCENARIO_CONFIG[args.scenario]
    stage_overrides: dict[str, dict[str, Any]] = {}
    if args.profile_stage is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        stage_overrides[str(args.profile_stage)] = {
            "profiler_config": {
                "profiler": "torch",
                "torch_profiler_dir": str(args.profile_dir),
                "torch_profiler_use_gzip": False,
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
            }
        }

    config = Mammothmoda2Config.from_pretrained(args.model)
    hidden_size = int(config.get_text_config().hidden_size)
    omni = Omni(
        model=args.model,
        deploy_config=str(deploy_config),
        mode="text-to-image",
        log_stats=True,
        enable_ar_profiler=True,
        stage_overrides=stage_overrides,
    )
    try:
        request, params = build_request(omni, args)
        warmup = run_once(
            omni,
            request,
            params,
            hidden_size,
        )
        if args.profile_stage is not None:
            omni.start_profile(
                profile_prefix=(f"mammoth_full_{args.scenario}_stage{args.profile_stage}"),
                stages=[args.profile_stage],
            )
        try:
            samples = [run_once(omni, request, params, hidden_size) for _ in range(args.iterations)]
        finally:
            if args.profile_stage is not None:
                omni.stop_profile(stages=[args.profile_stage])

        for sample in samples:
            validate_sample(sample)
        trace_files = (
            sorted(str(path) for path in args.profile_dir.rglob("*.json*")) if args.profile_dir is not None else []
        )
        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "scenario": args.scenario,
            "model": args.model,
            "deploy_config": str(deploy_config.relative_to(REPO_ROOT)),
            "prompt": args.prompt,
            "prompt_repeat": args.prompt_repeat,
            "image_size": [args.height, args.width],
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "text_guidance_scale": args.text_guidance_scale,
            "iterations": args.iterations,
            "warmup": warmup,
            "samples": samples,
            "summary": summarize(samples),
            "profile_stage": args.profile_stage,
            "trace_files": trace_files,
            "environment": {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT,
                    text=True,
                ).strip(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "vllm": vllm.__version__,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
