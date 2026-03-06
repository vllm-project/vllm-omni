# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Wan2.2 T2V.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers Wan2.2 model ID or local path.",
    )
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale (applied to low/high).")
    parser.add_argument("--guidance-scale-high", type=float, default=None, help="Optional separate CFG for high-noise.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames (Wan default is 81).")
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Optional generation frame rate (used by models like LTX2). Defaults to --fps.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Sampling steps.")
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=0.875,
        help="Boundary split ratio for low/high DiT. Default 0.875 uses both transformers for best quality. Set to 1.0 to load only the low-noise transformer (saves noticeable memory with good quality, recommended if memory is limited).",
    )
    parser.add_argument(
        "--flow-shift", type=float, default=5.0, help="Scheduler flow_shift (5.0 for 720p, 12.0 for 480p)."
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit"],
        help=(
            "Cache backend to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer). "
            "Default: None (no cache acceleration)."
        ),
    )
    parser.add_argument(
        "--enable-cache-dit-summary",
        action="store_true",
        help="Enable cache-dit summary logging after diffusion forward passes.",
    )
    parser.add_argument("--output", type=str, default="wan22_output.mp4", help="Path to save the video (mp4).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
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
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
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
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for audio output when saved (default: 24000 for LTX2).",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable vLLM-Omni statistics logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    frame_rate = args.frame_rate if args.frame_rate is not None else float(args.fps)

    # Wan2.2 cache-dit tuning (from cache-dit examples and cache_alignment).
    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            # DBCache parameters [cache-dit only]
            "Fn_compute_blocks": 1,  # Optimized for single-transformer models
            "Bn_compute_blocks": 0,  # Number of backward compute blocks
            "max_warmup_steps": 4,  # Maximum warmup steps (works for few-step models)
            "max_cached_steps": 20,
            "residual_diff_threshold": 0.24,  # Higher threshold for more aggressive caching
            "max_continuous_cached_steps": 3,  # Limit to prevent precision degradation
            # TaylorSeer parameters [cache-dit only]
            "enable_taylorseer": False,  # Disabled by default (not suitable for few-step models)
            "taylorseer_order": 1,  # TaylorSeer polynomial order
            # SCM (Step Computation Masking) parameters [cache-dit only]
            "scm_steps_mask_policy": None,  # SCM mask policy: None (disabled), "slow", "medium", "fast", "ultra"
            "scm_steps_policy": "dynamic",  # SCM steps policy: "dynamic" or "static"
        }
    # Configure parallel settings
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enable_cache_dit_summary=args.enable_cache_dit_summary,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        log_stats=args.log_stats,
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
        f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree},"
        f" cfg_parallel_size={args.cfg_parallel_size}, tensor_parallel_size={args.tensor_parallel_size},"
        f" vae_patch_parallel_size={args.vae_patch_parallel_size}"
    )
    print(f"  Video size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    frames = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_high,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
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
            if isinstance(inner_output, list):
                inner_output = inner_output[0] if inner_output else None
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

    use_ltx2_export = False
    if args.model and "ltx" in str(args.model).lower():
        use_ltx2_export = True
    if audio is not None:
        use_ltx2_export = True

    if use_ltx2_export:
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video
        except ImportError:
            raise ImportError("diffusers is required for LTX2 encode_video.")

        if isinstance(video_array, list):
            frames_np = np.stack(video_array, axis=0)
        elif isinstance(video_array, np.ndarray):
            frames_np = video_array
        else:
            frames_np = np.asarray(video_array)

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
            fps=args.fps,
            audio=audio_out,
            audio_sample_rate=args.audio_sample_rate if audio_out is not None else None,
            output_path=str(output_path),
        )
    else:
        export_to_video(video_array, str(output_path), fps=args.fps)
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
