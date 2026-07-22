# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Image-Camera to Video generation example using Lingbot World Fast

Usage example:
    python end2end.py --model path/to/lingbot-fast --image path/to/image --camera-path path/to/camera
    --width 832 --height 480 --prompt "Walk in the Great Wall of China" --output output.mp4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video from an image (Wan2.2, LTX2, HunyuanVideo-1.5).")
    parser.add_argument(
        "--model",
        default="lingbot_world/lingbot-world-base-cam/Lingbot-World-Fast",
        help="Diffusers I2V model ID or local path (Wan2.2 or HunyuanVideo-1.5).",
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--camera-path", default=None, help="Path to input camera positions")
    parser.add_argument("--prompt", default="", help="Text prompt describing the desired motion.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--height", type=int, default=None, help="Video height (auto-calculated from image if not set)."
    )
    parser.add_argument("--width", type=int, default=None, help="Video width (auto-calculated from image if not set).")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--output", type=str, default="i2v_output.mp4", help="Path to save the video (mp4).")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
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
    model_class_name = "LingbotWorldFastPipeline"

    # Load input image
    image = PIL.Image.open(args.image).convert("RGB")

    num_inference_steps = 40

    # Calculate dimensions if not provided
    height = args.height
    width = args.width
    if height is None or width is None:
        max_area = 480 * 832
        mod_value = 16
        calc_height, calc_width = calculate_dimensions(image, max_area=max_area, mod_value=mod_value)
        height = height or calc_height
        width = width or calc_width

    # Resize image to target dimensions
    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    omni = Omni(
        model=args.model,
        parallel_config=None,
        model_class_name=model_class_name,
        stage_init_timeout=6000,
        init_timeout=6000,
    )

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Video size: {width}x{height}")
    print(f"{'=' * 60}\n")

    # omni.generate() returns Generator[OmniRequestOutput, None, None]

    multi_modal_data = {"image": image}

    if args.camera_path is not None:
        poses = np.load(os.path.join(args.camera_path, "poses.npy"))
        intrinsics = np.load(os.path.join(args.camera_path, "intrinsics.npy"))

        multi_modal_data["camera"] = {"poses": poses, "intrinsics": intrinsics}

    generation_start = time.perf_counter()
    frames = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": multi_modal_data,
        },
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            generator=generator,
            num_frames=args.num_frames,
            frame_rate=args.fps,
            extra_args={"session_id": "offline_generation"},
        ),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.final_output_type != "image":
            raise ValueError(
                f"Unexpected output type '{frames.final_output_type}', expected 'image' for video generation."
            )
        if frames.is_pipeline_output and frames.request_output is not None:
            inner_output = frames.request_output
            if isinstance(inner_output, OmniRequestOutput):
                frames = inner_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if len(frames.images) == 1 and isinstance(frames.images[0], tuple) and len(frames.images[0]) == 2:
                    frames = frames.images[0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    if isinstance(frames, list) and frames:
        first_item = frames[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            frames = first_item
        elif isinstance(first_item, dict):
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames
    elif isinstance(frames, dict):
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
