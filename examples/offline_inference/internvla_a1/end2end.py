#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

_WORKSPACE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WORKSPACE_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    # Keep the shared helper import clean by doing script-only path bootstrap here.
    sys.path.insert(0, str(_REPO_ROOT))

from internvla_a1_common import (  # noqa: E402
    A2DOpenLoopDataset,
    collate_open_loop_samples,
    make_shared_noise,
    run_open_loop_evaluation,
    select_indices,
    tensor_dtype,
    tensor_sha256,
)

from vllm_omni.diffusion.data import OmniDiffusionConfig  # noqa: E402
from vllm_omni.diffusion.models.internvla_a1 import (  # noqa: E402
    InternVLAA1Config,
    InternVLAA1TrainMetadata,
)
from vllm_omni.diffusion.models.internvla_a1.config import OBS_STATE  # noqa: E402
from vllm_omni.diffusion.registry import initialize_model  # noqa: E402
from vllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402


def _required_path_arg(env_name: str, cli_value: str | None) -> str:
    value = cli_value or os.getenv(env_name)
    if not value:
        raise ValueError(f"Missing required path: set --{env_name.lower().replace('_', '-')} or {env_name}.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM InternVLA-A1 inference on a few samples.")
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
    parser.add_argument("--output-dir", default="outputs/internvla_a1/vllm_infer")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--benchmark-forward", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    return parser.parse_args()


def build_od_config(args: argparse.Namespace, processor_model_name: str) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=str(Path(args.model_dir).resolve()),
        model_class_name="InternVLAA1Pipeline",
        dtype=tensor_dtype(args.dtype),
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


def build_dataset(args: argparse.Namespace) -> tuple[A2DOpenLoopDataset, InternVLAA1Config, InternVLAA1TrainMetadata]:
    model_dir = Path(args.model_dir)
    config = InternVLAA1Config.from_pretrained(model_dir)
    config.device = args.device
    config.dtype = args.dtype
    config.compile_model = args.compile_model
    config.attn_implementation = args.attn_implementation
    config.enable_regional_compile = args.enable_regional_compile

    train_meta = InternVLAA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / "stats.json", encoding="utf-8") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(
        args.dataset_dir,
        config=config,
        train_stats=train_stats,
    )
    return dataset, config, train_meta


def run_one_path(
    pipeline,
    dataset: A2DOpenLoopDataset,
    config: InternVLAA1Config,
    args: argparse.Namespace,
    indices: list[int],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index in indices:
        sample = dataset.get_sample(index)
        batch_inputs, _ = collate_open_loop_samples([sample], device=args.device, dtype=tensor_dtype(args.dtype))
        noise = make_shared_noise(
            args.seed,
            index,
            (
                batch_inputs[OBS_STATE].shape[0],
                config.chunk_size,
                config.max_action_dim,
            ),
            args.device,
        )
        pred = run_pipeline_forward(pipeline, batch_inputs, noise)
        pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
        results.append(
            {
                "path": "registry",
                "index": index,
                "episode_index": sample.episode_index,
                "task": sample.task,
                "seed": args.seed,
                "shape": list(pred.shape),
                "mean": float(pred.mean().item()),
                "std": float(pred.std().item()),
                "action_sha256": tensor_sha256(pred),
                "first_action_prefix": pred[0, 0, :8].tolist(),
            }
        )
    return results


def run_pipeline_forward(
    pipeline,
    batch_inputs: dict[str, torch.Tensor],
    noise: torch.Tensor,
) -> torch.Tensor:
    output = pipeline.forward(
        OmniDiffusionRequest(
            prompts=[""],
            sampling_params=OmniDiffusionSamplingParams(
                extra_args={
                    "batch_inputs": batch_inputs,
                    "noise": noise,
                    "decode_image": False,
                }
            ),
        )
    )
    if output.error:
        raise RuntimeError(output.error)
    if output.output is None:
        raise RuntimeError("InternVLAA1Pipeline.forward returned no output tensor.")
    return output.output


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def benchmark_forward(
    pipeline,
    dataset: A2DOpenLoopDataset,
    config: InternVLAA1Config,
    args: argparse.Namespace,
    index: int,
    output_dir: Path,
) -> dict[str, object]:
    sample = dataset.get_sample(index)
    batch_inputs, _ = collate_open_loop_samples([sample], device=args.device, dtype=tensor_dtype(args.dtype))
    noise = make_shared_noise(
        args.seed,
        index,
        (
            batch_inputs[OBS_STATE].shape[0],
            config.chunk_size,
            config.max_action_dim,
        ),
        args.device,
    )

    _synchronize(args.device)
    cold_start_begin = time.perf_counter()
    pred = run_pipeline_forward(pipeline, batch_inputs, noise)
    _synchronize(args.device)
    cold_start_ms = (time.perf_counter() - cold_start_begin) * 1000.0

    warmup_ms: list[float] = []
    for _ in range(args.warmup_iters):
        _synchronize(args.device)
        begin = time.perf_counter()
        _ = run_pipeline_forward(pipeline, batch_inputs, noise)
        _synchronize(args.device)
        warmup_ms.append((time.perf_counter() - begin) * 1000.0)

    benchmark_ms: list[float] = []
    for _ in range(args.benchmark_iters):
        _synchronize(args.device)
        begin = time.perf_counter()
        _ = run_pipeline_forward(pipeline, batch_inputs, noise)
        _synchronize(args.device)
        benchmark_ms.append((time.perf_counter() - begin) * 1000.0)

    pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
    summary = {
        "mode": "forward_latency",
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "sample_index": index,
        "episode_index": sample.episode_index,
        "task": sample.task,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "enable_regional_compile": args.enable_regional_compile,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "output_shape": list(pred.shape),
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
    args.model_dir = _required_path_arg("INTERNVLA_A1_MODEL_DIR", args.model_dir)
    args.dataset_dir = _required_path_arg("INTERNVLA_A1_DATASET_DIR", args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, config, train_meta = build_dataset(args)
    indices = select_indices(dataset, args.num_samples)
    od_config = build_od_config(args, train_meta.processor_model_name)

    eval_summaries: dict[str, object] = {}
    pipeline = initialize_model(od_config)
    if args.benchmark_forward:
        benchmark_forward(
            pipeline=pipeline,
            dataset=dataset,
            config=config,
            args=args,
            index=indices[0],
            output_dir=output_dir,
        )
        return
    results = run_one_path(
        pipeline=pipeline,
        dataset=dataset,
        config=config,
        args=args,
        indices=indices,
    )
    if args.num_episodes > 0:
        eval_summaries["registry"] = run_open_loop_evaluation(
            mode="vllm_registry",
            policy=pipeline,
            config=config,
            dataset=dataset,
            train_meta=train_meta,
            collate_samples=collate_open_loop_samples,
            run_sample_actions=lambda policy, batch_inputs, noise: run_pipeline_forward(policy, batch_inputs, noise),
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            dtype=tensor_dtype(args.dtype),
            output_dir=output_dir / "registry",
            skip_plots=args.skip_plots,
        )

    summary = {
        "mode": "registry",
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "enable_regional_compile": args.enable_regional_compile,
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
