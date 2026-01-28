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
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Wan2.2 T2V.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers Wan2.2 model ID or local path.",
    )
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="CFG scale (applied to low/high).")
    parser.add_argument("--guidance_scale_high", type=float, default=None, help="Optional separate CFG for high-noise.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames (Wan default is 81).")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Sampling steps.")
    parser.add_argument("--boundary_ratio", type=float, default=0.875, help="Boundary split ratio for low/high DiT.")
    parser.add_argument(
        "--flow_shift", type=float, default=5.0, help="Scheduler flow_shift (5.0 for 720p, 12.0 for 480p)."
    )
    parser.add_argument("--output", type=str, default="wan22_output.mp4", help="Path to save the video (mp4).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    # Configure parallel settings (only SP is supported for Wan)
    # Note: cfg_parallel and tensor_parallel are not implemented for Wan models
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
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
    print(f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}")
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
        ),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Extract video frames from OmniRequestOutput
    if isinstance(frames, list) and len(frames) > 0:
        first_item = frames[0]

        # Check if it's an OmniRequestOutput
        if hasattr(first_item, "final_output_type"):
            if first_item.final_output_type != "image":
                raise ValueError(
                    f"Unexpected output type '{first_item.final_output_type}', expected 'image' for video generation."
                )

            # Pipeline mode: extract from nested request_output
            if hasattr(first_item, "is_pipeline_output") and first_item.is_pipeline_output:
                if isinstance(first_item.request_output, list) and len(first_item.request_output) > 0:
                    inner_output = first_item.request_output[0]
                    if isinstance(inner_output, OmniRequestOutput) and hasattr(inner_output, "images"):
                        frames = inner_output.images[0] if inner_output.images else None
                        if frames is None:
                            raise ValueError("No video frames found in output.")
            # Diffusion mode: use direct images field
            elif hasattr(first_item, "images") and first_item.images:
                frames = first_item.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    # frames may be np.ndarray (preferred) or torch.Tensor
    # export_to_video expects a list of frames with values in [0, 1]
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            # [B, C, F, H, W] or [B, F, H, W, C]
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        # If float, assume [-1,1] and normalize to [0,1]
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    else:
        video_array = frames
        if hasattr(video_array, "shape") and video_array.ndim == 5:
            video_array = video_array[0]

    # Convert 4D array (frames, H, W, C) to list of frames for export_to_video
    if isinstance(video_array, np.ndarray) and video_array.ndim == 4:
        video_array = list(video_array)

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
