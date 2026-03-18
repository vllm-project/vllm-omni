# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Image-to-Video generation example using Wan2.2 I2V/TI2V models or LTX2.

Supports:
- Wan2.2-I2V-A14B-Diffusers: MoE model with CLIP image encoder
- Wan2.2-TI2V-5B-Diffusers: Unified T2V+I2V model (dense 5B)
- LTX2 image-to-video pipeline

Usage:
    # I2V-A14B (MoE)
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
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video from an image with Wan2.2 or LTX2.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        help="Diffusers Wan2.2 I2V model ID or local path.",
    )
    parser.add_argument(
        "--model-class-name",
        default=None,
        help="Override model class name (e.g., LTX2ImageToVideoPipeline).",
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--prompt", default="", help="Text prompt describing the desired motion.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument(
        "--guidance-scale-high", type=float, default=None, help="Optional separate CFG for high-noise (MoE only)."
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Video height (auto-calculated from image if not set)."
    )
    parser.add_argument("--width", type=int, default=None, help="Video width (auto-calculated from image if not set).")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--boundary-ratio", type=float, default=0.875, help="Boundary split ratio for MoE models.")
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Optional generation frame rate (used by models like LTX2). Defaults to --fps.",
    )
    parser.add_argument(
        "--flow-shift", type=float, default=5.0, help="Scheduler flow_shift (5.0 for 720p, 12.0 for 480p)."
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
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for audio output when saved (default: 24000 for LTX2).",
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
    is_ltx2 = "ltx2" in model_name or (model_class_name and "ltx2" in model_class_name.lower())
    if model_class_name is None and is_ltx2:
        model_class_name = "LTX2ImageToVideoPipeline"

    # Load input image
    image = PIL.Image.open(args.image).convert("RGB")

    fps = args.fps if args.fps is not None else (24 if is_ltx2 else 16)
    frame_rate = args.frame_rate if args.frame_rate is not None else float(fps)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else (4.0 if is_ltx2 else 5.0)
    num_frames = args.num_frames if args.num_frames is not None else (121 if is_ltx2 else 81)
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else (40 if is_ltx2 else 50)

    # Calculate dimensions if not provided
    height = args.height
    width = args.width
    if height is None or width is None:
        # Default to 480P area for Wan2.2 I2V, 512x768 area for LTX2
        max_area = 512 * 768 if is_ltx2 else 480 * 832
        mod_value = 32 if is_ltx2 else 16
        calc_height, calc_width = calculate_dimensions(image, max_area=max_area, mod_value=mod_value)
        height = height or calc_height
        width = width or calc_width

    # Resize image to target dimensions
    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

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

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )
    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        model_class_name=model_class_name,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
    )

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Frames: {args.num_frames}")
    print(
        f"  Parallel configuration: cfg_parallel_size={args.cfg_parallel_size},"
        f" tensor_parallel_size={args.tensor_parallel_size}, vae_patch_parallel_size={args.vae_patch_parallel_size}"
    )
    print(f"  Video size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    # omni.generate() returns Generator[OmniRequestOutput, None, None]
    frames = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": image},
        },
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            generator=generator,
            guidance_scale=guidance_scale,
            guidance_scale_2=args.guidance_scale_high,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            frame_rate=frame_rate,
        ),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    audio = None
    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.final_output_type != "image":
            raise ValueError(
                f"Unexpected output type '{frames.final_output_type}', expected 'image' for video generation."
            )
        if frames.multimodal_output and "audio" in frames.multimodal_output:
            audio = frames.multimodal_output["audio"]
        if frames.is_pipeline_output and frames.request_output is not None:
            inner_output = frames.request_output
            if isinstance(inner_output, OmniRequestOutput):
                if inner_output.multimodal_output and "audio" in inner_output.multimodal_output:
                    audio = inner_output.multimodal_output["audio"]
                frames = inner_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if len(frames.images) == 1 and isinstance(frames.images[0], tuple) and len(frames.images[0]) == 2:
                    frames, audio = frames.images[0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    audio = frames.images[0].get("audio")
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
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio")
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

    use_ltx2_export = is_ltx2
    encode_video = None
    if use_ltx2_export:
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video
        except ImportError:
            encode_video = None

    if use_ltx2_export and encode_video is not None:
        if isinstance(video_array, list):
            frames_np = np.stack(video_array, axis=0)
        elif isinstance(video_array, np.ndarray):
            frames_np = video_array
        else:
            frames_np = np.asarray(video_array)

        if frames_np.ndim == 4 and frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]

        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_u8 = (frames_np * 255).round().clip(0, 255).astype("uint8")
        video_tensor = torch.from_numpy(frames_u8)

        audio_out = None
        if audio is not None:
            if isinstance(audio, list):
                audio = audio[0] if audio else None
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if isinstance(audio, torch.Tensor):
                audio_out = audio
                if audio_out.dim() > 1:
                    audio_out = audio_out[0]
                audio_out = audio_out.float().cpu()

        encode_video(
            video_tensor,
            fps=fps,
            audio=audio_out,
            audio_sample_rate=args.audio_sample_rate if audio_out is not None else None,
            output_path=str(output_path),
        )
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
