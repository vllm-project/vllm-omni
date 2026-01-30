# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Image-to-Video generation example using Wan2.2 I2V/TI2V models.

Supports:
- Wan2.2-I2V-A14B-Diffusers: MoE model with CLIP image encoder
- Wan2.2-TI2V-5B-Diffusers: Unified T2V+I2V model (dense 5B)

Usage:
    # I2V-A14B (MoE)
    python image_to_video.py --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --image input.jpg --prompt "A cat playing with yarn"

    # TI2V-5B (unified)
    python image_to_video.py --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --image input.jpg --prompt "A cat playing with yarn"
"""

import argparse
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video from an image with Wan2.2 I2V/TI2V.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        help="Diffusers Wan2.2 I2V model ID or local path.",
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--prompt", default="", help="Text prompt describing the desired motion.")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument(
        "--guidance_scale_high", type=float, default=None, help="Optional separate CFG for high-noise (MoE only)."
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Video height (auto-calculated from image if not set)."
    )
    parser.add_argument("--width", type=int, default=None, help="Video width (auto-calculated from image if not set).")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--boundary_ratio", type=float, default=0.875, help="Boundary split ratio for MoE models.")
    parser.add_argument(
        "--flow_shift", type=float, default=5.0, help="Scheduler flow_shift (5.0 for 720p, 12.0 for 480p)."
    )
    parser.add_argument("--output", type=str, default="i2v_output.mp4", help="Path to save the video (mp4).")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")
    parser.add_argument(
        "--vae_use_slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization.",
    )
    parser.add_argument(
        "--vae_use_tiling",
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
        "--layerwise-num-gpu-layers",
        type=int,
        default=1,
        help="Number of ready layers (blocks) to keep on GPU during generation.",
    )
    return parser.parse_args()


def calculate_dimensions(image: PIL.Image.Image, max_area: int = 480 * 832) -> tuple[int, int]:
    """Calculate output dimensions maintaining aspect ratio."""
    aspect_ratio = image.height / image.width
    mod_value = 16  # Must be divisible by 16

    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    return height, width


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    # Load input image
    image = PIL.Image.open(args.image).convert("RGB")

    # Calculate dimensions if not provided
    height = args.height
    width = args.width
    if height is None or width is None:
        # Default to 480P area for I2V
        calc_height, calc_width = calculate_dimensions(image, max_area=480 * 832)
        height = height or calc_height
        width = width or calc_width

    # Resize image to target dimensions
    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        layerwise_num_gpu_layers=args.layerwise_num_gpu_layers,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

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
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_high,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
        ),
    )

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
