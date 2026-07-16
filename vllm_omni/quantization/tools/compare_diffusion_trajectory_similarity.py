# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare diffusion quantization quality with final output metrics.

This tool runs a reference diffusion configuration and a candidate
configuration with the same prompt, seed, resolution, and step count. It reports
final image or video frame similarity metrics plus generation latency and memory.

The main metrics are:

* output images: the same pixel-space metrics plus PSNR
* runtime: per-run and average generation latency, and worker-reported peak
  device memory when available

Online FP8 Qwen-Image example:

    python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \\
        --model Qwen/Qwen-Image \\
        --candidate-quantization fp8 \\
        --ignored-layers img_mlp \\
        --prompt "a cup of coffee on the table" \\
        --height 1024 --width 1024 --num-inference-steps 20 --seed 42 \\
        --output-json /mnt/data4/cwq/tmp/qwen_image_fp8_similarity.json

Offline checkpoint example:

    python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \\
        --reference-model Qwen/Qwen-Image \\
        --candidate-model /path/to/quantized/qwen-image-checkpoint \\
        --candidate-quantization-config-json '{"method":"inc","weight_bits":4}' \\
        --prompt "a cup of coffee on the table" \\
        --output-json /mnt/data4/cwq/tmp/qwen_image_checkpoint_similarity.json

Offline FP8 model-id example:

    python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \\
        --model Qwen/Qwen-Image \\
        --candidate-model Qwen/Qwen-Image-FP8 \\
        --prompt "a cup of coffee on the table" \\
        --height 1024 --width 1024 --num-inference-steps 20 --seed 42 \\
        --output-json /mnt/data4/cwq/tmp/qwen_image_fp8_checkpoint_similarity.json

If the checkpoint does not carry a loadable quantization config, add
--candidate-quantization-config-json, for example '{"method":"fp8"}'.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class VariantConfig:
    label: str
    model: str
    quantization: str | None
    quantization_config: dict[str, Any] | None
    diffusion_load_format: str


@dataclass
class VariantRun:
    label: str
    result: Any
    generation_times_s: list[float]
    peak_memory_mb: list[float | None]


def metric_guidance() -> dict[str, Any]:
    """Human-readable metric notes and default heuristic thresholds."""
    return {
        "scope": (
            "Heuristic defaults for same-model comparisons with identical prompt, seed, resolution, scheduler, "
            "and inference steps. Tune thresholds per model family, task, and quantization method; visual review is "
            "still recommended for release gating."
        ),
        "descriptions": {
            "output_metrics": (
                "Pixel-space metrics computed on generated RGB images or video frames after decode. These are often "
                "the most actionable signal for image/video quality regressions."
            ),
            "cosine_similarity": (
                "Higher is better. Measures vector direction similarity and is less sensitive to scale."
            ),
            "mae": (
                "Lower is better. Mean absolute error in the tensor's native scale; uint8 output MAE is in "
                "pixel values."
            ),
            "mse": "Lower is better. Mean squared error; more sensitive to large localized differences than MAE.",
            "rmse": "Lower is better. Square root of MSE, in the same unit as the compared tensor.",
            "max_abs": (
                "Lower is better. Worst absolute element error; useful for debugging outliers, not stable as a gate."
            ),
            "l2": "Lower is better. Absolute L2 distance; shape- and scale-dependent.",
            "relative_l2": "Lower is better. L2 distance normalized by the reference L2 norm.",
            "psnr_db": "Higher is better. Pixel-space signal-to-noise ratio in dB for uint8 images/frames.",
            "generation_time_s": "Lower is better. Wall-clock time for the last measured generation run.",
            "avg_generation_time_s": "Lower is better. Mean wall-clock time across measured runs.",
            "peak_memory_mb": (
                "Lower is better. Worker-reported peak device memory for the last measured generation run."
            ),
            "max_peak_memory_mb": "Lower is better. Maximum worker-reported peak memory across measured runs.",
        },
        "recommended_thresholds": {
            "output_images_or_frames_uint8": {
                "psnr_db": {
                    "recommended_min": 20.0,
                    "good_min": 25.0,
                    "strict_min": 30.0,
                    "note": (
                        "For stochastic diffusion outputs, 20 dB is a practical smoke threshold; 25+ dB is usually "
                        "close."
                    ),
                },
                "mae": {
                    "recommended_max": 12.0,
                    "good_max": 6.0,
                    "strict_max": 3.0,
                    "note": "Measured on uint8 RGB pixels or frames.",
                },
                "cosine_similarity": {
                    "recommended_min": 0.98,
                    "strict_min": 0.995,
                    "note": "Useful as a broad image/frame similarity sanity check.",
                },
                "relative_l2": {
                    "recommended_max": 0.20,
                    "strict_max": 0.08,
                    "note": "Pixel-space relative L2 is generally more interpretable than latent-space relative L2.",
                },
            },
            "performance": {
                "avg_generation_time_ratio_candidate_over_reference": {
                    "recommended_max": 1.10,
                    "note": (
                        "A value above 1.10 suggests a latency regression unless expected for the quantization mode."
                    ),
                },
                "max_peak_memory_ratio_candidate_over_reference": {
                    "recommended_max": 1.00,
                    "target_max": 0.95,
                    "note": (
                        "Quantization should usually not increase peak memory; <=0.95 indicates a meaningful saving."
                    ),
                },
            },
        },
    }


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_json_arg(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        value = path.read_text(encoding="utf-8")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("quantization config JSON must decode to an object.")
    return parsed


def _cosine_similarity(flat_a: torch.Tensor, flat_b: torch.Tensor) -> float:
    norm_a = torch.linalg.vector_norm(flat_a).item()
    norm_b = torch.linalg.vector_norm(flat_b).item()
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cosine = float(F.cosine_similarity(flat_a, flat_b, dim=0).item())
    return max(-1.0, min(1.0, cosine))


def compute_tensor_metrics(lhs: Any, rhs: Any) -> dict[str, float]:
    lhs_tensor = torch.as_tensor(lhs).detach().cpu().double()
    rhs_tensor = torch.as_tensor(rhs).detach().cpu().double()
    if lhs_tensor.shape != rhs_tensor.shape:
        raise ValueError(f"Metric shape mismatch: {tuple(lhs_tensor.shape)} vs {tuple(rhs_tensor.shape)}")

    diff = lhs_tensor - rhs_tensor
    mse = float(diff.square().mean().item())
    rmse = float(math.sqrt(mse))
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    l2 = float(torch.linalg.vector_norm(diff).item())
    ref_l2 = float(torch.linalg.vector_norm(lhs_tensor).item())
    relative_l2 = 0.0 if ref_l2 == 0.0 and l2 == 0.0 else float("inf") if ref_l2 == 0.0 else l2 / ref_l2
    cosine = _cosine_similarity(lhs_tensor.reshape(-1), rhs_tensor.reshape(-1))
    return {
        "cosine_similarity": cosine,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs": max_abs,
        "l2": l2,
        "relative_l2": relative_l2,
    }


def compute_uint8_image_metrics(lhs: Any, rhs: Any) -> dict[str, float]:
    metrics = compute_tensor_metrics(lhs, rhs)
    mse = metrics["mse"]
    metrics["psnr_db"] = float("inf") if mse == 0.0 else 20 * math.log10(255.0) - 10 * math.log10(mse)
    return metrics


def summarize_output_frame_metrics(reference_frames: list[Any], candidate_frames: list[Any]) -> dict[str, Any]:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError(f"Output frame count mismatch: {len(reference_frames)} vs {len(candidate_frames)}")
    if not reference_frames:
        raise ValueError("No output frames available for comparison.")

    ref_stack = np.stack([np.asarray(frame.convert("RGB")) for frame in reference_frames], axis=0)
    cand_stack = np.stack([np.asarray(frame.convert("RGB")) for frame in candidate_frames], axis=0)

    frame0_metrics = compute_uint8_image_metrics(ref_stack[0], cand_stack[0])
    mid_index = len(reference_frames) // 2
    mid_metrics = compute_uint8_image_metrics(ref_stack[mid_index], cand_stack[mid_index])
    all_metrics = compute_uint8_image_metrics(ref_stack, cand_stack)

    return {
        "num_frames": len(reference_frames),
        "frame0_metrics": frame0_metrics,
        "mid_frame_index": mid_index,
        "mid_frame_metrics": mid_metrics,
        "all_frames_metrics": all_metrics,
        # Backward-compatible aliases for existing image callers.
        "num_images": len(reference_frames),
        "image0_metrics": frame0_metrics,
        "mid_image_index": mid_index,
        "mid_image_metrics": mid_metrics,
        "all_images_metrics": all_metrics,
    }


def summarize_output_image_metrics(reference_images: list[Any], candidate_images: list[Any]) -> dict[str, Any]:
    return summarize_output_frame_metrics(reference_images, candidate_images)


def _extract_inner_output(outputs: Any) -> Any:
    if isinstance(outputs, list):
        if len(outputs) != 1:
            raise ValueError(f"Expected one request output, got {len(outputs)}.")
        outputs = outputs[0]
    inner = getattr(outputs, "request_output", None)
    if inner is not None:
        return inner
    return outputs


def _request_peak_memory_mb(result: Any) -> float | None:
    diffusion_output = getattr(result, "request_output", None)
    peak_memory_mb = getattr(diffusion_output, "peak_memory_mb", None)
    if peak_memory_mb is None:
        peak_memory_mb = getattr(result, "peak_memory_mb", None)
    if peak_memory_mb is None:
        return None
    return float(peak_memory_mb)


def _build_quantization_config(
    *,
    quantization: str | None,
    quantization_config_json: str | None,
    ignored_layers: str | None,
) -> dict[str, Any] | None:
    explicit = _load_json_arg(quantization_config_json)
    if explicit is not None:
        return explicit
    if quantization is None:
        return None

    config: dict[str, Any] = {"method": quantization}
    ignored = _split_csv(ignored_layers)
    if ignored:
        config["ignored_layers"] = ignored
    return config


def _build_variant_config(args: argparse.Namespace, variant: str) -> VariantConfig:
    common_ignored_layers = args.ignored_layers if variant == "candidate" else None
    ignored_layers = getattr(args, f"{variant}_ignored_layers") or common_ignored_layers
    quantization = getattr(args, f"{variant}_quantization")
    quantization_config = _build_quantization_config(
        quantization=quantization,
        quantization_config_json=getattr(args, f"{variant}_quantization_config_json"),
        ignored_layers=ignored_layers,
    )
    return VariantConfig(
        label=variant,
        model=getattr(args, f"{variant}_model") or args.model,
        quantization=quantization if quantization_config is None else None,
        quantization_config=quantization_config,
        diffusion_load_format=getattr(args, f"{variant}_load_format"),
    )


def _build_omni_kwargs(args: argparse.Namespace, config: VariantConfig) -> dict[str, Any]:
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
    )
    kwargs: dict[str, Any] = {
        "model": config.model,
        "mode": "text-to-video" if args.task == "t2v" else "text-to-image",
        "parallel_config": parallel_config,
        "enforce_eager": args.enforce_eager,
        "enable_cpu_offload": args.enable_cpu_offload,
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "step_execution": args.step_execution if args.task == "t2i" else False,
        "diffusion_load_format": config.diffusion_load_format,
    }
    if args.model_class_name:
        kwargs["model_class_name"] = args.model_class_name
    if args.flow_shift is not None:
        kwargs["flow_shift"] = args.flow_shift
    if config.quantization_config is not None:
        kwargs["quantization_config"] = config.quantization_config
    elif config.quantization is not None:
        kwargs["quantization"] = config.quantization
    return kwargs


def _build_sampling_params(args: argparse.Namespace, seed: int):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    kwargs: dict[str, Any] = {
        "height": args.height,
        "width": args.width,
        "generator": generator,
        "seed": seed,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "num_inference_steps": args.num_inference_steps,
        "return_frames": True,
        "extra_args": {
            "timesteps_shift": args.timesteps_shift,
            "cfg_schedule": args.cfg_schedule,
            "use_norm": args.use_norm,
            "use_system_prompt": args.use_system_prompt,
            "system_prompt": args.system_prompt,
        },
    }
    if args.task == "t2v":
        kwargs["num_frames"] = args.num_frames
        kwargs["fps"] = args.fps
        kwargs["frame_rate"] = args.frame_rate
    else:
        kwargs["true_cfg_scale"] = args.cfg_scale
        kwargs["num_outputs_per_prompt"] = args.num_images_per_prompt
    return OmniDiffusionSamplingParams(**kwargs)


def _run_variant(args: argparse.Namespace, config: VariantConfig) -> VariantRun:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.platforms import current_omni_platform

    omni = Omni(**_build_omni_kwargs(args, config))
    measured_results: list[Any] = []
    generation_times: list[float] = []
    peak_memory_mb: list[float | None] = []
    try:
        for _ in range(args.warmup_runs):
            sampling_params = _build_sampling_params(args, args.seed)
            omni.generate(
                {"prompt": args.prompt, "negative_prompt": args.negative_prompt},
                sampling_params,
                use_tqdm=False,
            )

        for _ in range(args.measure_runs):
            sampling_params = _build_sampling_params(args, args.seed)
            start = time.perf_counter()
            outputs = omni.generate(
                {"prompt": args.prompt, "negative_prompt": args.negative_prompt},
                sampling_params,
                use_tqdm=False,
            )
            current_omni_platform.synchronize()
            generation_times.append(time.perf_counter() - start)
            result = _extract_inner_output(outputs)
            peak_memory_mb.append(_request_peak_memory_mb(result))
            measured_results.append(result)
    finally:
        omni.close()
        del omni
        gc.collect()
        current_omni_platform.empty_cache()

    if not measured_results:
        raise RuntimeError("No measured results were produced.")
    return VariantRun(
        label=config.label,
        result=measured_results[-1],
        generation_times_s=generation_times,
        peak_memory_mb=peak_memory_mb,
    )


def _get_output_frames(result: Any) -> list[Any]:
    images = list(getattr(result, "images", []) or [])
    if len(images) == 1 and isinstance(images[0], list):
        return list(images[0])
    return images


def _save_outputs(result: Any, output_dir: Path, label: str, task: str, fps: int) -> list[str]:
    frames = _get_output_frames(result)
    saved: list[str] = []
    if not frames:
        return saved
    variant_dir = output_dir / label
    variant_dir.mkdir(parents=True, exist_ok=True)
    if task == "t2v":
        from diffusers.utils import export_to_video

        video_path = variant_dir / "video.mp4"
        export_to_video(frames, str(video_path), fps=fps)
        saved.append(str(video_path))
        for idx in (0, len(frames) // 2, len(frames) - 1):
            path = variant_dir / f"frame_{idx}.png"
            frames[idx].save(path)
            saved.append(str(path))
        return saved
    for idx, image in enumerate(frames):
        path = variant_dir / f"image_{idx}.png"
        image.save(path)
        saved.append(str(path))
    return saved


def _run_summary(run: VariantRun) -> dict[str, Any]:
    valid_peak = [value for value in run.peak_memory_mb if value is not None]
    return {
        "generation_time_s": run.generation_times_s[-1],
        "avg_generation_time_s": sum(run.generation_times_s) / len(run.generation_times_s),
        "per_run_generation_time_s": run.generation_times_s,
        "peak_memory_mb": valid_peak[-1] if valid_peak else None,
        "avg_peak_memory_mb": sum(valid_peak) / len(valid_peak) if valid_peak else None,
        "max_peak_memory_mb": max(valid_peak) if valid_peak else None,
        "per_run_peak_memory_mb": run.peak_memory_mb,
    }


def _to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, allow_nan=True))


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0.")
    if args.measure_runs <= 0:
        raise ValueError("--measure-runs must be >= 1.")

    reference_config = _build_variant_config(args, "reference")
    candidate_config = _build_variant_config(args, "candidate")

    reference_run = _run_variant(args, reference_config)
    candidate_run = _run_variant(args, candidate_config)

    save_output_paths: dict[str, list[str]] = {}
    if args.save_output_dir:
        save_dir = Path(args.save_output_dir).expanduser().resolve()
        save_output_paths["reference"] = _save_outputs(reference_run.result, save_dir, "reference", args.task, args.fps)
        save_output_paths["candidate"] = _save_outputs(candidate_run.result, save_dir, "candidate", args.task, args.fps)

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "variant_configs": {
            "reference": reference_config.__dict__,
            "candidate": candidate_config.__dict__,
        },
        "sampling_kwargs": {
            "task": args.task,
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames if args.task == "t2v" else None,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "cfg_scale": args.cfg_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "flow_shift": args.flow_shift,
            "num_images_per_prompt": args.num_images_per_prompt,
            "step_execution": args.step_execution if args.task == "t2i" else False,
        },
        "metric_guidance": metric_guidance(),
        "reference_generation": _run_summary(reference_run),
        "candidate_generation": _run_summary(candidate_run),
        "output_metrics": summarize_output_image_metrics(
            _get_output_frames(reference_run.result),
            _get_output_frames(candidate_run.result),
        ),
    }
    if save_output_paths:
        result["saved_outputs"] = save_output_paths
    return _to_jsonable(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=["t2i", "t2v"], default="t2i")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Default model for both variants.")
    parser.add_argument("--model-class-name", help="Optional diffusion pipeline class override.")
    parser.add_argument("--reference-model", help="Reference model path or HF ID. Defaults to --model.")
    parser.add_argument("--candidate-model", help="Candidate model path or HF ID. Defaults to --model.")
    parser.add_argument("--reference-quantization", help="Reference online quantization method.")
    parser.add_argument("--candidate-quantization", help="Candidate online quantization method, e.g. fp8.")
    parser.add_argument(
        "--reference-quantization-config-json",
        help="Reference quantization_config JSON string or file.",
    )
    parser.add_argument(
        "--candidate-quantization-config-json",
        help="Candidate quantization_config JSON string or file.",
    )
    parser.add_argument("--reference-ignored-layers", help="Comma-separated reference ignored layer patterns.")
    parser.add_argument("--candidate-ignored-layers", help="Comma-separated candidate ignored layer patterns.")
    parser.add_argument("--ignored-layers", help="Alias for --candidate-ignored-layers.")
    parser.add_argument("--reference-load-format", default="default", help="Reference diffusion load format.")
    parser.add_argument("--candidate-load-format", default="default", help="Candidate diffusion load format.")
    parser.add_argument("--prompt", default="a cup of coffee on the table")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-output-dir")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames for t2v.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for saved t2v videos.")
    parser.add_argument("--frame-rate", type=float, help="Optional generation frame rate for t2v.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Qwen-Image true CFG scale.")
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measure-runs", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument(
        "--step-execution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the diffusion step scheduler for text-to-image models. Disabled by default so the tool follows the "
            "normal offline inference path unless explicitly requested."
        ),
    )
    parser.add_argument("--timesteps-shift", type=float, default=1.0)
    parser.add_argument("--cfg-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument(
        "--use-system-prompt",
        choices=[
            "None",
            "dynamic",
            "en_vanilla",
            "en_recaption",
            "en_think_recaption",
            "en_unified",
            "custom",
        ],
    )
    parser.add_argument("--system-prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    result = run_comparison(args)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    image0 = result["output_metrics"]["image0_metrics"]
    summary = {
        "output_json": str(output_json),
        "reference_avg_generation_time_s": result["reference_generation"]["avg_generation_time_s"],
        "candidate_avg_generation_time_s": result["candidate_generation"]["avg_generation_time_s"],
        "image0_psnr_db": image0["psnr_db"],
        "image0_mae": image0["mae"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
