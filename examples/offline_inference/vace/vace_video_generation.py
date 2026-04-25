# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""VACE video generation example.

VACE (Video All-in-one Creation Engine) supports multiple video tasks:
  - T2V:        Text-to-Video
  - I2V:        Image-to-Video (first frame conditioning)
  - V2LF:       Video-to-Last-Frame
  - FLF2V:      First-Last-Frame interpolation
  - Inpainting:  Masked region generation
  - R2V:        Reference image-guided generation

Usage examples:
  # T2V (text-to-video)
  python vace_video_generation.py --mode t2v --prompt "A robot in a warehouse"

  # I2V (image-to-video, first frame kept)
  python vace_video_generation.py --mode i2v --image input.jpg --prompt "..."

  # FLF2V (first-last frame interpolation)
  python vace_video_generation.py --mode flf2v --image first.jpg --last-image last.jpg

  # R2V (reference image guided)
  python vace_video_generation.py --mode r2v --image ref.jpg --prompt "..."
"""

import argparse
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VACE video generation.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-VACE-14B-diffusers",
        help="VACE model ID or local path.",
    )
    parser.add_argument(
        "--mode",
        default="t2v",
        choices=["t2v", "i2v", "v2lf", "flf2v", "inpaint", "r2v"],
        help="Generation mode.",
    )
    parser.add_argument("--prompt", default="A cat walking in a garden", help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--image", type=str, default=None, help="Input image path (for I2V, R2V, FLF2V, inpaint).")
    parser.add_argument("--last-image", type=str, default=None, help="Last frame image path (for FLF2V).")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory of video frames (for inpaint).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG guidance scale.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Sampling steps.")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Scheduler flow_shift.")
    parser.add_argument("--output", type=str, default="vace_output.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--vae-use-tiling", action="store_true", default=True, help="Enable VAE tiling.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses SP degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring attention degree.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def build_prompts(args):
    """Build prompt dict with multi_modal_data based on mode."""
    h, w, nf = args.height, args.width, args.num_frames

    gray = PIL.Image.new("RGB", (w, h), (128, 128, 128))
    mask_black = PIL.Image.new("L", (w, h), 0)
    mask_white = PIL.Image.new("L", (w, h), 255)

    prompt_data = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }

    if args.mode == "t2v":
        return prompt_data

    if args.mode == "r2v":
        assert args.image, "--image required for R2V mode"
        ref_img = PIL.Image.open(args.image).convert("RGB").resize((w, h))
        prompt_data["multi_modal_data"] = {"reference_images": [ref_img]}
        return prompt_data

    if args.mode == "i2v":
        assert args.image, "--image required for I2V mode"
        img = PIL.Image.open(args.image).convert("RGB").resize((w, h))
        prompt_data["multi_modal_data"] = {
            "video": [img] + [gray] * (nf - 1),
            "mask": [mask_black] + [mask_white] * (nf - 1),
        }
        return prompt_data

    if args.mode == "v2lf":
        assert args.image, "--image required for V2LF mode"
        img = PIL.Image.open(args.image).convert("RGB").resize((w, h))
        prompt_data["multi_modal_data"] = {
            "video": [gray] * (nf - 1) + [img],
            "mask": [mask_white] * (nf - 1) + [mask_black],
        }
        return prompt_data

    if args.mode == "flf2v":
        assert args.image and args.last_image, "--image and --last-image required for FLF2V"
        first = PIL.Image.open(args.image).convert("RGB").resize((w, h))
        last = PIL.Image.open(args.last_image).convert("RGB").resize((w, h))
        prompt_data["multi_modal_data"] = {
            "video": [first] + [gray] * (nf - 2) + [last],
            "mask": [mask_black] + [mask_white] * (nf - 2) + [mask_black],
        }
        return prompt_data

    if args.mode == "inpaint":
        assert args.image, "--image required for inpaint mode"
        img = PIL.Image.open(args.image).convert("RGB").resize((w, h))
        d = 80
        frames, masks = [], []
        for _ in range(nf):
            base = np.array(img).copy()
            mask = PIL.Image.new("L", (w, h), 0)
            stripe = PIL.Image.new("L", (2 * d, h), 255)
            mask.paste(stripe, (w // 2 - d, 0))
            base[np.array(mask) > 128] = 128
            frames.append(PIL.Image.fromarray(base))
            masks.append(mask)
        prompt_data["multi_modal_data"] = {"video": frames, "mask": masks}
        return prompt_data

    raise ValueError(f"Unknown mode: {args.mode}")


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
    )

    omni = Omni(
        model=args.model,
        vae_use_tiling=args.vae_use_tiling,
        flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager,
        parallel_config=parallel_config,
    )

    prompt_data = build_prompts(args)

    print(f"\n{'=' * 60}")
    print(f"VACE {args.mode.upper()} Generation")
    print(f"  Model: {args.model}")
    print(f"  Size: {args.width}x{args.height}, {args.num_frames} frames, {args.num_inference_steps} steps")
    print(f"{'=' * 60}\n")

    start = time.perf_counter()
    outputs = omni.generate(
        prompt_data,
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ),
    )
    elapsed = time.perf_counter() - start

    video = outputs[0].images
    if isinstance(video, list):
        video = video[0]
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if video.ndim == 5:
        video = video[0]
    print(f"Output shape: {video.shape}, Time: {elapsed:.1f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video

    if np.issubdtype(video.dtype, np.integer):
        video = video.astype(np.float32) / 255.0
    export_to_video(list(video), str(output_path), fps=args.fps)
    print(f"Saved to {output_path}")

    omni.close()


if __name__ == "__main__":
    main()
