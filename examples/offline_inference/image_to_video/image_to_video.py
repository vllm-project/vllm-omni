# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Image-to-Video generation example using Wan2.2 I2V/TI2V models, LTX2/LTX-2.3,
HunyuanVideo-1.5, Cosmos3, or Wan2.1 VACE.

Supports:
- Wan2.2-I2V-A14B-Diffusers: MoE model with CLIP image encoder
- Wan2.2-TI2V-5B-Diffusers: Unified T2V+I2V model (dense 5B)
- LTX2 image-to-video pipeline
- HunyuanVideo-1.5 I2V: SigLIP + VAE dual image conditioning
- Wan2.1 VACE: first/last-frame, inpainting, and reference conditioning

Usage:
    # Wan I2V-A14B (MoE)
    python image_to_video.py --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --image input.jpg --prompt "A cat playing with yarn"

    # TI2V-5B (unified)
    python image_to_video.py --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --image input.jpg --prompt "A cat playing with yarn"

    # LTX2 image-to-video
    python image_to_video.py --model /path/to/LTX-2 \
        --model-class-name LTX2ImageToVideoPipeline \
        --image input.jpg --prompt "A cinematic dolly shot of a boat" \
        --num-frames 121 --num-inference-steps 40 --guidance-scale 4.0 \
        --frame-rate 24 --fps 24 --output ltx2_i2v.mp4

    # LTX-2.3 image-to-video
    python image_to_video.py --model diffusers/LTX-2.3-Diffusers \
        --model-class-name LTX23ImageToVideoPipeline \
        --image input.jpg --prompt "A cinematic dolly shot of a boat" \
        --height 384 --width 512 --num-frames 25 --num-inference-steps 20 \
        --frame-rate 24 --fps 24 --output ltx23_i2v.mp4

    # HunyuanVideo-1.5 I2V (480p)
    python image_to_video.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
        --image input.jpg --prompt "A cat playing with yarn" \
        --flow-shift 5.0 --guidance-scale 6.0

    # Cosmos3 I2V (image conditioning)
    python image_to_video.py --model nvidia/Cosmos3-Nano \
        --image input.jpg --prompt "The scene comes to life with smooth, natural motion." \
        --num-frames 189 --num-inference-steps 35 --guidance-scale 6.0 --fps 24 \
        --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false}'
"""

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_video_prompt,
    get_extra_body_params,
    get_model_class_name,
)
from vllm_omni.outputs import OmniRequestOutput
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
        description=(
            "Generate a video from one or more images "
            "(Wan2.2, LTX2/LTX-2.3, HunyuanVideo-1.5, Cosmos3, or Wan2.1 VACE)."
        )
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        help="Diffusers I2V model ID or local path (Wan2.2, LTX2/LTX-2.3, HunyuanVideo-1.5, Cosmos3, or Wan2.1 VACE).",
    )
    parser.add_argument(
        "--model-class-name",
        default=None,
        help="Override model class name (e.g., LTX2ImageToVideoPipeline or LTX23ImageToVideoPipeline).",
    )
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Optional deploy config YAML to use for pipeline-backed runs.",
    )
    parser.add_argument("--image", help="Path to the first-frame or source image.")
    parser.add_argument("--last-image", help="Path to a last-frame condition (used by models such as VACE).")
    parser.add_argument("--mask-image", help="Path to an inpainting mask (used by models such as VACE).")
    parser.add_argument(
        "--reference-image",
        action="append",
        default=None,
        help="Path to a reference image. Repeat to provide multiple references.",
    )
    parser.add_argument("--prompt", default="", help="Text prompt describing the desired motion.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale. Default: model-specific.")
    parser.add_argument(
        "--guidance-scale-high", type=float, default=None, help="Optional separate CFG for high-noise (MoE only)."
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Video height (auto-calculated from image if not set)."
    )
    parser.add_argument("--width", type=int, default=None, help="Video width (auto-calculated from image if not set).")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames. Default: model-specific.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Sampling steps. Default: model-specific.",
    )
    parser.add_argument("--boundary-ratio", type=float, default=0.875, help="Boundary split ratio for MoE models.")
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Optional generation frame rate (used by models like LTX2). Defaults to --fps.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Scheduler flow_shift. Default: model-specific (Wan 5.0, Cosmos3 10.0).",
    )
    parser.add_argument(
        "--sample-solver",
        type=str,
        default="unipc",
        choices=["unipc", "euler"],
        help="Sampling solver for Wan2.2 pipelines. Use 'euler' for Lightning/Distill setups.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-dtype",
        type=str,
        default=None,
        help="Diffusion attention KV cache dtype (e.g. float8_e4m3fn). Separate from vLLM --kv-cache-dtype.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-skip-steps",
        type=str,
        default=None,
        help="Diffusion KV-cache quantization skip-step selector, e.g. '0-9,20,25-30'.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-skip-layers",
        type=str,
        default=None,
        help="Diffusion KV-cache quantization skip-layer selector, e.g. '0,1,4-8'.",
    )
    parser.add_argument("--output", type=str, default="i2v_output.mp4", help="Path to save the video (mp4).")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for the output video.")
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
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for audio output when saved (default: 24000).",
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
        choices=["fp8", "mxfp8", "mxfp4", "mxfp4_dualscale", "int8"],
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
            "so unknown keys for the chosen model are silently dropped. "
            'Cosmos3 V2V example: \'{"condition_frame_indexes_vision": [0, 1], '
            '"condition_video_keep": "first", "flow_shift": 10.0, '
            '"max_sequence_length": 4096, "guardrails": false}\'.'
        ),
    )
    return parser.parse_args()


def calculate_dimensions(
    image: PIL.Image.Image,
    max_area: int = 480 * 832,
    mod_value: int = 16,
) -> tuple[int, int]:
    """Calculate output dimensions maintaining aspect ratio."""
    aspect_ratio = image.height / image.width

    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    return height, width


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    model_name = str(args.model).lower() if args.model is not None else ""
    model_class_name = args.model_class_name
    is_ltx2 = model_class_name in {"LTX2ImageToVideoPipeline", "LTX23ImageToVideoPipeline"}
    is_cosmos = "cosmos" in model_name or (model_class_name is not None and "cosmos" in model_class_name.lower())

    image = PIL.Image.open(args.image).convert("RGB") if args.image else None
    last_image = PIL.Image.open(args.last_image).convert("RGB") if args.last_image else None
    mask_image = PIL.Image.open(args.mask_image).convert("L") if args.mask_image else None
    reference_images = (
        [PIL.Image.open(path).convert("RGB") for path in args.reference_image] if args.reference_image else None
    )
    dimension_image = image or last_image or (reference_images[0] if reference_images else None)
    if dimension_image is None:
        raise ValueError("Provide --image, --last-image, or at least one --reference-image.")

    # Per-model generation defaults, applied only when the matching flag is omitted.
    # Cosmos3 would otherwise silently inherit the Wan2.2 defaults (wrong size/steps/shift).
    if is_cosmos:
        d_fps, d_guidance, d_num_frames, d_steps, d_flow_shift, d_max_area, d_mod = (
            24,
            6.0,
            189,
            35,
            10.0,
            1280 * 720,
            16,
        )
    elif is_ltx2:
        d_fps, d_guidance, d_num_frames, d_steps, d_flow_shift, d_max_area, d_mod = (
            24,
            4.0,
            121,
            40,
            5.0,
            512 * 768,
            32,
        )
    else:  # Wan2.2 / HunyuanVideo-1.5
        d_fps, d_guidance, d_num_frames, d_steps, d_flow_shift, d_max_area, d_mod = 16, 5.0, 81, 50, 5.0, 480 * 832, 16

    fps = args.fps if args.fps is not None else d_fps
    frame_rate = args.frame_rate if args.frame_rate is not None else float(fps)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else d_guidance
    num_frames = args.num_frames if args.num_frames is not None else d_num_frames
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else d_steps
    flow_shift = args.flow_shift if args.flow_shift is not None else d_flow_shift

    # Calculate dimensions if not provided (model-aware max area).
    height = args.height
    width = args.width
    if height is None or width is None:
        calc_height, calc_width = calculate_dimensions(dimension_image, max_area=d_max_area, mod_value=d_mod)
        height = height or calc_height
        width = width or calc_width

    media_inputs: dict[str, Any] = {}
    if image is not None:
        media_inputs["image"] = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    if last_image is not None:
        media_inputs["last_image"] = last_image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    if mask_image is not None:
        media_inputs["mask"] = mask_image.resize((width, height), PIL.Image.Resampling.NEAREST)
    if reference_images is not None:
        media_inputs["reference_images"] = [
            reference.resize((width, height), PIL.Image.Resampling.LANCZOS) for reference in reference_images
        ]

    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        if is_ltx2:
            cache_config = {
                "Fn_compute_blocks": 2,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 8,
                "residual_diff_threshold": 0.12,
                "max_continuous_cached_steps": 1,
                "max_cached_steps": 20,
                "enable_taylorseer": False,
                "scm_steps_mask_policy": None,
            }
        else:
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
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=flow_shift,
        diffusion_kv_cache_dtype=args.diffusion_kv_cache_dtype,
        diffusion_kv_cache_skip_steps=args.diffusion_kv_cache_skip_steps,
        diffusion_kv_cache_skip_layers=args.diffusion_kv_cache_skip_layers,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        model_class_name=model_class_name,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        profiler_config=args.profiler_config,
    )
    if args.deploy_config:
        omni_kwargs["deploy_config"] = args.deploy_config
    if args.quantization is not None:
        omni_kwargs["quantization"] = args.quantization
    # Cosmos3 loads its (gated) guardrail models at build time, so the guardrails
    # gate is an engine-level config (offline analog of the server's --no-guardrails).
    if args.extra_body and "guardrails" in args.extra_body:
        omni_kwargs["model_config"] = {"guardrails": bool(args.extra_body["guardrails"])}
    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni) or model_class_name
    declared_extra_body_params = get_extra_body_params(model_class_name)

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Frames: {num_frames}")
    print(f"  Solver: {args.sample_solver}")
    print(f"  diffusion_kv_cache_dtype(config): {args.diffusion_kv_cache_dtype}")
    print(f"  diffusion_kv_cache_skip_steps(config): {args.diffusion_kv_cache_skip_steps}")
    print(f"  diffusion_kv_cache_skip_layers(config): {args.diffusion_kv_cache_skip_layers}")
    print(
        f"  Parallel configuration: cfg_parallel_size={args.cfg_parallel_size},"
        f" tensor_parallel_size={args.tensor_parallel_size}, vae_patch_parallel_size={args.vae_patch_parallel_size},"
        f" pipeline_parallel_size={args.pipeline_parallel_size}"
    )
    print(f"  Video size: {width}x{height}")
    print(f"{'=' * 60}\n")

    prompt_dict = build_image_to_video_prompt(
        model_class_name=model_class_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        media_inputs=media_inputs,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    sampling_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        generator=generator,
        guidance_scale=guidance_scale,
        guidance_scale_2=args.guidance_scale_high,
        boundary_ratio=args.boundary_ratio,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        frame_rate=frame_rate,
        extra_args={
            "sample_solver": args.sample_solver,
            "flow_shift": flow_shift,
        },
    )

    # Route model-specific knobs through extra_body, filtered against the model's
    # declared extra_body_params. Models without a declaration only forward explicit
    # --extra-body JSON (e.g. Cosmos3 V2V's condition_frame_indexes_vision).
    extra_body = dict(args.extra_body or {})
    if declared_extra_body_params:
        apply_declared_extra_args(sampling_params, declared_extra_body_params, extra_body)
    elif extra_body:
        sampling_params.extra_args.update({k: v for k, v in extra_body.items() if v is not None})

    generation_start = time.perf_counter()
    # omni.generate() returns Generator[OmniRequestOutput, None, None]
    frames = omni.generate(prompt_dict, sampling_params)
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    audio = None
    audio_sample_rate = args.audio_sample_rate
    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.final_output_type != "image":
            raise ValueError(
                f"Unexpected output type '{frames.final_output_type}', expected 'image' for video generation."
            )
        if frames.multimodal_output and "audio" in frames.multimodal_output:
            audio = frames.multimodal_output["audio"]
            audio_sample_rate = frames.multimodal_output.get("audio_sample_rate", audio_sample_rate)
        if frames.is_pipeline_output and frames.request_output is not None:
            inner_output = frames.request_output
            if isinstance(inner_output, OmniRequestOutput):
                if inner_output.multimodal_output and "audio" in inner_output.multimodal_output:
                    audio = inner_output.multimodal_output["audio"]
                    audio_sample_rate = inner_output.multimodal_output.get("audio_sample_rate", audio_sample_rate)
                frames = inner_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if len(frames.images) == 1 and isinstance(frames.images[0], tuple) and len(frames.images[0]) == 2:
                    frames, audio = frames.images[0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    audio = frames.images[0].get("audio")
                    audio_sample_rate = frames.images[0].get("audio_sample_rate", audio_sample_rate)
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    if isinstance(frames, list) and frames:
        first_item = frames[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            frames, audio = first_item
        elif isinstance(first_item, dict):
            audio = first_item.get("audio")
            audio_sample_rate = first_item.get("audio_sample_rate", audio_sample_rate)
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio")
        audio_sample_rate = frames.get("audio_sample_rate", audio_sample_rate)
        frames = frames.get("frames") or frames.get("video")

    if frames is None:
        raise ValueError("No video frames found in output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    def _normalize_frame(frame):
        if isinstance(frame, torch.Tensor):
            frame_tensor = frame.detach().cpu()
            if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
                frame_tensor = frame_tensor[0]
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
                frame_tensor = frame_tensor.permute(1, 2, 0)
            if frame_tensor.is_floating_point():
                frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
            return frame_tensor.float().numpy()
        if isinstance(frame, np.ndarray):
            frame_array = frame
            if frame_array.ndim == 4 and frame_array.shape[0] == 1:
                frame_array = frame_array[0]
            if np.issubdtype(frame_array.dtype, np.integer):
                frame_array = frame_array.astype(np.float32) / 255.0
            return frame_array
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None and isinstance(frame, Image.Image):
            return np.asarray(frame).astype(np.float32) / 255.0
        return frame

    def _ensure_frame_list(video_array):
        if isinstance(video_array, list):
            if len(video_array) == 0:
                return video_array
            first_item = video_array[0]
            if isinstance(first_item, np.ndarray):
                if first_item.ndim == 5:
                    return list(first_item[0])
                if first_item.ndim == 4:
                    if len(video_array) == 1:
                        return list(first_item)
                    return list(first_item)
                if first_item.ndim == 3:
                    return video_array
            return video_array
        if isinstance(video_array, np.ndarray):
            if video_array.ndim == 5:
                return list(video_array[0])
            if video_array.ndim == 4:
                return list(video_array)
            if video_array.ndim == 3:
                return [video_array]
        return video_array

    # frames may be np.ndarray, torch.Tensor, or list of tensors/arrays/images
    # export_to_video expects a list of frames with values in [0, 1]
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    elif isinstance(frames, np.ndarray):
        video_array = frames
        if video_array.ndim == 5:
            video_array = video_array[0]
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
    elif isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("No video frames found in output.")
        video_array = [_normalize_frame(frame) for frame in frames]
    else:
        video_array = frames

    video_array = _ensure_frame_list(video_array)

    if audio is not None:
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        if isinstance(video_array, list):
            frames_np = np.stack(video_array, axis=0)
        elif isinstance(video_array, np.ndarray):
            frames_np = video_array
        else:
            frames_np = np.asarray(video_array)

        if frames_np.ndim == 4 and frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]

        frames_u8 = (np.clip(frames_np, 0.0, 1.0) * 255).round().clip(0, 255).astype("uint8")

        audio_np = audio
        if isinstance(audio_np, list):
            audio_np = audio_np[0] if audio_np else None
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.detach().cpu().float().numpy()
        if isinstance(audio_np, np.ndarray):
            audio_np = np.squeeze(audio_np).astype(np.float32)

        video_bytes = mux_video_audio_bytes(
            frames_u8,
            audio_np,
            fps=float(fps),
            audio_sample_rate=audio_sample_rate,
        )
        with open(str(output_path), "wb") as f:
            f.write(video_bytes)
    else:
        export_to_video(video_array, str(output_path), fps=fps)
    print(f"Saved generated video to {output_path}")

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
