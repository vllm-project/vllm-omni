# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Robot policy inference — shared task for robot-policy DiT models.

The script is model-agnostic: it selects an *inference mode* and an
*observation loader* declared in ``vllm_omni/model_extras/<model>.py``. To add a
new robot-policy model, register ``robot_obs_builder`` + ``action_output_processor``
+ ``robot_policy_finalizer`` — no edits here. Inference mode can be automatically
detected by checking input type.

Examples:
    # DreamZero
    python robot_policy.py --model GEAR-Dreams/DreamZero-DROID \\
        --deploy-config vllm_omni/deploy/dreamzero.yaml \\
        --data-dir outputs/dreamzero/assets \\
        --task "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.engine.stage_init_utils import _resolve_model_to_local_path
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_extras import (
    build_robot_observations,
    finalize_robot_run,
    get_extra_body_params,
    get_model_class_name,
    get_worker_extension_class,
    process_robot_actions,
)
from vllm_omni.platforms import current_omni_platform


def parse_json_object(value: str, flag_name: str = "argument") -> dict[str, Any]:
    """Parse a CLI value as a JSON object, attributing errors to ``flag_name``."""
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"{flag_name} must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must be a JSON object")
    return config


parse_profiler_config = functools.partial(parse_json_object, flag_name="--profiler-config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot policy inference, action sequence planning from an image or numpy array."
    )
    parser.add_argument(
        "--model",
        default="GEAR-Dreams/DreamZero-DROID",
        help="Diffusers Robot policy model ID or local path (Dreamzero, Internvla-a1, ...)",
    )
    parser.add_argument("--model-class-name", default=None, help="Override model class name.")
    parser.add_argument(
        "--task",
        default="",
        help=(
            "Task prompt string that controls the robot trajectory planning. "
            "Warning: this overrides the dataset-provided prompt if set."
        ),
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing organized assets needed by examples.")
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Deploy config YAML",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default="robot_policy_output.npz",
    )
    parser.add_argument(
        "--vae-use-slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization.",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offloading on DiT modules.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer), 'tea_cache' (Timestep Embedding Aware Cache). "
            "Default: None (no cache acceleration)."
        ),
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "mxfp4", "mxfp4_dualscale", "int8", "gguf"],
        help="Quantization method for the transformer. mxfp8: W8A8 MXFP8 (NPU). mxfp4: W4A4 MXFP4 (NPU). mxfp4_dualscale: W4A4 MXFP4 dual-scale + BF16 fallback mixed (NPU). fp8: online FP8 (GPU).",
    )

    # Distributed and parallel execution
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
    )
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help=("Enable Hybrid Sharded Data Parallel to shard model weights across GPUs. "),
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=-1,
        help=(
            "Number of GPUs to shard model weights across within each replica group. "
            "-1 (default) auto-calculates as world_size / replicate_size. "
        ),
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help=(
            "Number of replica groups for HSDP. Each replica holds a full sharded copy. "
            "Default 1 means pure sharding (no replication). "
        ),
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of pipeline parallel stages.",
    )
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help='JSON profiler config for torch/cuda profiling, e.g. \'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.',
    )
    parser.add_argument(
        "--extra-body",
        type=functools.partial(parse_json_object, flag_name="--extra-body"),
        default=None,
        help=(
            "Model-specific generation params as a JSON object. Keys are filtered "
            "against the model's declared extra_body_params (see vllm_omni/model_extras), "
            "unknown keys for the chosen model are silently dropped. "
            "internvla_a1 example: --extra-body '{\"decode_image\": true}' "
            'dreamzero example: --extra-body \'{"session_id": "dreamzero_1"}\''
        ),
    )
    return parser.parse_args()


def normalize_extra_body_params(
    sampling_params,
    extra_body: dict,
    declared_extra_body: dict,
):
    if declared_extra_body:
        apply_declared_extra_args(sampling_params, declared_extra_body, extra_body)
    elif extra_body:
        sampling_params.extra_args.update({k: v for k, v in extra_body.items() if v is not None})
    return sampling_params


def run_inference(
    omni: Omni, generator: torch.Generator, model_class_name, observations, extra_body, declared_extra_body
) -> list[dict[str, Any]]:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    metadata = extra_body.get("metadata", {})

    results = []
    for index, extra_args in enumerate(observations):
        prompt = extra_args.get("prompt", "")
        sp = OmniDiffusionSamplingParams(
            extra_args=extra_args,
            generator=generator,
        )
        sp = normalize_extra_body_params(sp, extra_body, declared_extra_body)
        raw = omni.generate(prompt, sampling_params_list=[sp])
        if not raw:
            raise RuntimeError(f"No output for step {index}")
        results.append(process_robot_actions(model_class_name, raw[0], **metadata))
    return results


def main() -> None:
    args = parse_args()
    model_class_name = args.model_class_name
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    model_dir = _resolve_model_to_local_path(args.model)

    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }
    elif args.cache_backend == "tea_cache":
        cache_config = {
            "rel_l1_thresh": 0.2,
        }

    profiler_enabled = args.profiler_config is not None
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
    omni_kwargs = dict(
        model=args.model,
        model_class_name=model_class_name,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        profiler_config=args.profiler_config,
    )
    if args.quantization is not None:
        omni_kwargs["quantization"] = args.quantization
    omni_kwargs["worker_extension_cls"] = get_worker_extension_class(model_class_name)
    print(f"\n{'=' * 60}")
    print(f"[Robot policy] model_class_name {model_class_name}")

    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni) or model_class_name
    print(f"[Robot Policy] Using model_class_name - {model_class_name}")

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # print task configuration
    print(f"\n{'=' * 60}")
    print("Robot policy configuration")
    print(f"  Model: {args.model}")
    print(f"  Task prompt: {args.task or '(using dataset-provided prompt)'}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Using worker extension: {omni_kwargs['worker_extension_cls']}")
    print(f"{'=' * 60}\n")

    extra_body = dict(args.extra_body or {})
    observation, metadata = build_robot_observations(
        model_class_name,
        model_dir=model_dir,
        task=args.task,
        data_dir=args.data_dir,
        **{"seed": args.seed, "device": args.device, "dtype": args.dtype, **extra_body},
    )

    if extra_body.get("metadata", {}):
        print(
            "[Warning] --extra-body key 'metadata' conflicts with "
            "builder-provided metadata; builder values take precedence."
        )
    extra_body["metadata"] = metadata
    declared_extra_body_params = get_extra_body_params(model_class_name)

    # Return type drives the mode: a single dict → single-shot,
    # any other iterable → autoregressive.
    observations = [observation] if isinstance(observation, dict) else observation
    results = run_inference(omni, generator, model_class_name, observations, extra_body, declared_extra_body_params)

    if not results:
        print("[Robot Policy] No actions produced.")
        return

    actions = [r["actions"] for r in results]
    stacked = np.stack(actions, axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, actions=stacked, num_steps=len(results))
    print(f"[Robot Policy] saved {stacked.shape} → {output_path}")

    # model-specific post-process steps
    finalize_robot_run(model_class_name, omni, results, output_path)

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, dict):
            traces = profile_results.get("traces", [])
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for rank, trace in enumerate(traces):
                print(f"\nRank {rank}:")
                if trace:
                    print(f"  • Trace: {trace}")
            if not traces:
                print("  No traces collected.")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")


if __name__ == "__main__":
    main()
