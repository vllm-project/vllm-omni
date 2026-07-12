#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared observation-to-action offline example.

Robot / VLA policies build observation tensors, run diffusion (or AR) inference,
then post-process predicted action chunks. Model-specific hooks live in
``vllm_omni.model_extras`` (observation_builder / action_processor /
eval_context_loader / open_loop_runner).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

_WORKSPACE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WORKSPACE_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.diffusion.data import OmniDiffusionConfig  # noqa: E402
from vllm_omni.diffusion.registry import initialize_model  # noqa: E402
from vllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: E402
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args  # noqa: E402
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.model_extras import (  # noqa: E402
    build_observations,
    get_extra_body_params,
    load_eval_context,
    process_actions,
    run_open_loop,
)
from vllm_omni.model_extras.internvla_a1_dataset import select_indices  # noqa: E402


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--extra-body must be a JSON object.")
    return value


def _required_path_arg(env_name: str, cli_value: str | None, flag_name: str) -> str:
    value = cli_value or os.getenv(env_name)
    if not value:
        raise ValueError(f"Missing required path: set {flag_name} or {env_name}.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run observation-to-action offline inference.")
    parser.add_argument(
        "--model-class-name",
        default="InternVLAA1Pipeline",
        help="Pipeline class registered in model_extras (default: InternVLAA1Pipeline).",
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--enable-regional-compile", action="store_true")
    parser.add_argument("--enable-warmup", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--decode-image", action="store_true")
    parser.add_argument(
        "--extra-body",
        type=_parse_json_object,
        default=None,
        help="JSON object for declared extra_body knobs, e.g. '{\"num_steps\": 2}'.",
    )
    parser.add_argument("--output-dir", default="outputs/observation_to_action/vllm_infer")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--benchmark-forward", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    return parser.parse_args()


def _merge_extra_body(args: argparse.Namespace) -> dict[str, Any]:
    user_extra: dict[str, Any] = {}
    if args.extra_body:
        user_extra.update(args.extra_body)
    if args.num_steps is not None and "num_steps" not in user_extra:
        user_extra["num_steps"] = args.num_steps
    if args.decode_image and "decode_image" not in user_extra:
        user_extra["decode_image"] = True
    return user_extra


def build_od_config(
    args: argparse.Namespace,
    *,
    model_class_name: str,
    processor_model_name: str,
) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=str(Path(args.model_dir).resolve()),
        model_class_name=model_class_name,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        custom_pipeline_args={
            "device": args.device,
            "dtype": args.dtype,
            "compile_model": args.compile_model,
            "attn_implementation": args.attn_implementation,
            "enable_regional_compile": args.enable_regional_compile,
            "enable_warmup": args.enable_warmup,
            "strict_load": args.strict_load,
            "processor_model_name": processor_model_name,
        },
    )


def run_pipeline_forward(
    pipeline,
    *,
    model_class_name: str,
    batch_inputs: dict[str, torch.Tensor],
    noise: torch.Tensor,
    request_id: str,
    user_extra: dict[str, Any],
) -> torch.Tensor:
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={
            "batch_inputs": batch_inputs,
            "noise": noise,
        }
    )
    apply_declared_extra_args(
        sampling_params,
        get_extra_body_params(model_class_name),
        user_extra,
    )
    output = pipeline.forward(
        DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt="",
                    sampling_params=sampling_params,
                    request_id=request_id,
                )
            ]
        )
    )
    if output.error:
        raise RuntimeError(output.error)
    if output.output is None:
        raise RuntimeError(f"{model_class_name}.forward returned no output tensor.")
    return output.output


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.accelerator.synchronize()


def _latency_summary(values_ms: list[float]) -> dict[str, float]:
    sorted_values = sorted(values_ms)
    p50_index = min(len(sorted_values) - 1, round(0.50 * (len(sorted_values) - 1)))
    p90_index = min(len(sorted_values) - 1, round(0.90 * (len(sorted_values) - 1)))
    return {
        "mean_ms": float(statistics.mean(sorted_values)),
        "stdev_ms": float(statistics.pstdev(sorted_values)) if len(sorted_values) > 1 else 0.0,
        "min_ms": float(sorted_values[0]),
        "max_ms": float(sorted_values[-1]),
        "p50_ms": float(sorted_values[p50_index]),
        "p90_ms": float(sorted_values[p90_index]),
    }


def run_one_path(
    pipeline,
    *,
    model_class_name: str,
    context: dict[str, Any],
    args: argparse.Namespace,
    indices: list[int],
    user_extra: dict[str, Any],
) -> list[dict[str, object]]:
    dataset = context["dataset"]
    config = context["config"]
    results: list[dict[str, object]] = []
    for index in indices:
        obs = build_observations(
            model_class_name,
            dataset=dataset,
            config=config,
            index=index,
            seed=args.seed,
            device=args.device,
            dtype=context["torch_dtype"],
        )
        pred = run_pipeline_forward(
            pipeline,
            model_class_name=model_class_name,
            batch_inputs=obs["batch_inputs"],
            noise=obs["noise"],
            request_id=f"observation-to-action-sample-{index}",
            user_extra=user_extra,
        )
        processed = process_actions(
            model_class_name,
            pred=pred,
            dataset=dataset,
            sample=obs["sample"],
            index=index,
            seed=args.seed,
        )
        results.append(
            {
                "path": "registry",
                "index": processed["index"],
                "episode_index": processed["episode_index"],
                "task": processed["task"],
                "seed": processed["seed"],
                "shape": processed["shape"],
                "mean": processed["mean"],
                "std": processed["std"],
                "action_sha256": processed["action_sha256"],
                "first_action_prefix": processed["first_action_prefix"],
            }
        )
    return results


def benchmark_forward(
    pipeline,
    *,
    model_class_name: str,
    context: dict[str, Any],
    args: argparse.Namespace,
    index: int,
    output_dir: Path,
    user_extra: dict[str, Any],
) -> dict[str, object]:
    dataset = context["dataset"]
    config = context["config"]
    obs = build_observations(
        model_class_name,
        dataset=dataset,
        config=config,
        index=index,
        seed=args.seed,
        device=args.device,
        dtype=context["torch_dtype"],
    )
    sample = obs["sample"]

    _synchronize(args.device)
    cold_start_begin = time.perf_counter()
    pred = run_pipeline_forward(
        pipeline,
        model_class_name=model_class_name,
        batch_inputs=obs["batch_inputs"],
        noise=obs["noise"],
        request_id=f"observation-to-action-benchmark-{index}-cold",
        user_extra=user_extra,
    )
    _synchronize(args.device)
    cold_start_ms = (time.perf_counter() - cold_start_begin) * 1000.0

    warmup_ms: list[float] = []
    for iter_idx in range(args.warmup_iters):
        _synchronize(args.device)
        begin = time.perf_counter()
        _ = run_pipeline_forward(
            pipeline,
            model_class_name=model_class_name,
            batch_inputs=obs["batch_inputs"],
            noise=obs["noise"],
            request_id=f"observation-to-action-benchmark-{index}-warmup-{iter_idx}",
            user_extra=user_extra,
        )
        _synchronize(args.device)
        warmup_ms.append((time.perf_counter() - begin) * 1000.0)

    benchmark_ms: list[float] = []
    for iter_idx in range(args.benchmark_iters):
        _synchronize(args.device)
        begin = time.perf_counter()
        _ = run_pipeline_forward(
            pipeline,
            model_class_name=model_class_name,
            batch_inputs=obs["batch_inputs"],
            noise=obs["noise"],
            request_id=f"observation-to-action-benchmark-{index}-iter-{iter_idx}",
            user_extra=user_extra,
        )
        _synchronize(args.device)
        benchmark_ms.append((time.perf_counter() - begin) * 1000.0)

    processed = process_actions(
        model_class_name,
        pred=pred,
        dataset=dataset,
        sample=sample,
        index=index,
        seed=args.seed,
    )
    summary = {
        "mode": "forward_latency",
        "model_class_name": model_class_name,
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "sample_index": index,
        "episode_index": sample.episode_index,
        "task": sample.task,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "enable_regional_compile": args.enable_regional_compile,
        "num_steps": user_extra.get("num_steps"),
        "decode_image": bool(user_extra.get("decode_image", False)),
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "output_shape": processed["shape"],
        "cold_start_ms": cold_start_ms,
        "warmup_summary": _latency_summary(warmup_ms) if warmup_ms else {},
        "benchmark_summary": _latency_summary(benchmark_ms) if benchmark_ms else {},
        "benchmark_samples_ms": benchmark_ms,
    }
    with open(output_dir / "forward_latency.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    model_class_name = args.model_class_name

    # InternVLA path env defaults (other models can use --model-dir/--dataset-dir directly).
    if model_class_name == "InternVLAA1Pipeline":
        args.model_dir = _required_path_arg("INTERNVLA_A1_MODEL_DIR", args.model_dir, "--model-dir")
        args.dataset_dir = _required_path_arg("INTERNVLA_A1_DATASET_DIR", args.dataset_dir, "--dataset-dir")
    else:
        args.model_dir = _required_path_arg("OBSERVATION_TO_ACTION_MODEL_DIR", args.model_dir, "--model-dir")
        args.dataset_dir = _required_path_arg("OBSERVATION_TO_ACTION_DATASET_DIR", args.dataset_dir, "--dataset-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    user_extra = _merge_extra_body(args)

    context = load_eval_context(
        model_class_name,
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile_model,
        attn_implementation=args.attn_implementation,
        enable_regional_compile=args.enable_regional_compile,
    )
    dataset = context["dataset"]
    indices = select_indices(dataset, args.num_samples)
    od_config = build_od_config(
        args,
        model_class_name=model_class_name,
        processor_model_name=context["processor_model_name"],
    )

    eval_summaries: dict[str, object] = {}
    pipeline = initialize_model(od_config)
    if args.benchmark_forward:
        benchmark_forward(
            pipeline,
            model_class_name=model_class_name,
            context=context,
            args=args,
            index=indices[0],
            output_dir=output_dir,
            user_extra=user_extra,
        )
        return

    results = run_one_path(
        pipeline,
        model_class_name=model_class_name,
        context=context,
        args=args,
        indices=indices,
        user_extra=user_extra,
    )
    if args.num_episodes > 0:
        eval_summaries["registry"] = run_open_loop(
            model_class_name,
            policy=pipeline,
            dataset=dataset,
            config=context["config"],
            train_meta=context["train_meta"],
            run_sample_actions=lambda policy, batch_inputs, noise: run_pipeline_forward(
                policy,
                model_class_name=model_class_name,
                batch_inputs=batch_inputs,
                noise=noise,
                request_id="observation-to-action-open-loop",
                user_extra=user_extra,
            ),
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            dtype=context["torch_dtype"],
            output_dir=output_dir / "registry",
            skip_plots=args.skip_plots,
            mode="vllm_registry",
        )

    summary = {
        "mode": "registry",
        "model_class_name": model_class_name,
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "enable_regional_compile": args.enable_regional_compile,
        "num_steps": user_extra.get("num_steps"),
        "decode_image": bool(user_extra.get("decode_image", False)),
        "seed": args.seed,
        "indices": indices,
        "results": results,
        "output_dir": str(output_dir.resolve()),
        "eval_summaries": eval_summaries,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
