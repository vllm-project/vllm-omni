# SPDX-License-Identifier: Apache-2.0
"""
Cosmos-Predict2.5 offline inference: text2world (T2W), image2world (I2W), video2world (V2W).

Note: this model requires --revision to locate weights inside the HF repo.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cosmos-Predict2.5 unified script for text2world, image2world, and video2world."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["text2world", "image2world", "video2world"],
    )
    parser.add_argument("--model", default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--revision", default="diffusers/base/post-trained")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--image", help="Input image for image2world.")
    parser.add_argument("--video", help="Input video for video2world.")
    parser.add_argument(
        "--num-latent-conditional-frames",
        type=int,
        default=None,
        help="video2world only. Diffusers Cosmos supports 1 or 2.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output", default="./output.mp4")

    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "image2world":
        if not args.image or not os.path.exists(args.image):
            raise ValueError(f"--image required and must exist for image2world: {args.image}")
    if args.mode == "video2world":
        if not args.video or not os.path.exists(args.video):
            raise ValueError(f"--video required and must exist for video2world: {args.video}")
        if args.num_latent_conditional_frames is not None and args.num_latent_conditional_frames not in (1, 2):
            raise ValueError("--num-latent-conditional-frames must be 1 or 2")


def build_extra_kwargs(args: argparse.Namespace) -> dict:
    if args.mode == "image2world":
        image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
        return {"image": image}
    if args.mode == "video2world":
        from diffusers.utils import load_video

        kwargs = {"video": load_video(args.video)}
        if args.num_latent_conditional_frames is not None:
            kwargs["num_latent_conditional_frames"] = args.num_latent_conditional_frames
        return kwargs
    return {}


def main() -> None:
    args = parse_args()
    validate_args(args)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    print(f"\n{'=' * 60}")
    print(f"Cosmos-Predict2.5 [{args.mode}] | {args.model}@{args.revision}")
    print(f"  prompt:  {args.prompt}")
    print(f"  size:    {args.width}x{args.height}, {args.num_frames} frames @ {args.fps}fps")
    print(f"  steps:   {args.num_inference_steps}, cfg={args.guidance_scale}")
    print(f"{'=' * 60}\n")

    omni = Omni(
        model=args.model,
        revision=args.revision,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        enforce_eager=args.enforce_eager,
    )

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )

    start = time.perf_counter()
    frames = omni.generate(prompt_dict, sampling, **build_extra_kwargs(args))
    print(f"Generation time: {time.perf_counter() - start:.2f}s")

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
                    frames = frames.images[0][0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    if isinstance(frames, list) and frames:
        first_item = frames[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            frames = first_item[0]
        elif isinstance(first_item, dict):
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    elif isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")

    if frames is None:
        raise ValueError("No video frames found in output.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

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
            from PIL import Image as _Image
        except ImportError:
            _Image = None
        if _Image is not None and isinstance(frame, _Image.Image):
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
    export_to_video(video_array, str(out_path), fps=args.fps)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
