#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
SkyReels-V3 Image-to-Video (R2V) Offline Inference Example.

This script demonstrates how to use the SkyReels-V3 R2V model to generate
videos from reference images using the vLLM-Omni framework.

Usage:
    python image_to_video.py --model Skywork/SkyReels-V3-R2V-14B \
                             --image path/to/image.jpg \
                             --prompt "A person walking in the park"
"""

import argparse
import os
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion


def main():
    parser = argparse.ArgumentParser(description="SkyReels-V3 Image-to-Video Generation")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/SkyReels-V3-R2V-14B",
        help="Model name or path (default: Skywork/SkyReels-V3-R2V-14B)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the reference image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic video",
        help="Text prompt describing the desired video",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (optional)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of frames to generate (default: 81)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (default: 7.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/skyreels_v3",
        help="Output directory for generated videos (default: ./outputs/skyreels_v3)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="mp4",
        choices=["mp4", "gif", "frames"],
        help="Output format: mp4, gif, or frames (default: mp4)",
    )
    
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    image = Image.open(args.image).convert("RGB")
    print(f"Loaded reference image: {args.image} ({image.size})")

    # Initialize the model
    print(f"Loading SkyReels-V3 model: {args.model}")
    model = OmniDiffusion(
        model=args.model,
        model_class_name="SkyReelsV3R2VPipeline",
        trust_remote_code=True,
    )

    # Prepare the request
    print(f"\nGenerating video with prompt: '{args.prompt}'")
    print(f"Parameters:")
    print(f"  - Resolution: {args.width}x{args.height}")
    print(f"  - Frames: {args.num_frames}")
    print(f"  - Steps: {args.num_inference_steps}")
    print(f"  - Guidance Scale: {args.guidance_scale}")
    print(f"  - Seed: {args.seed}")

    # Generate video
    outputs = model.generate(
        prompts=[
            {
                "prompt": args.prompt,
                "multi_modal_data": {"image": image},
            }
        ],
        sampling_params={
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
        },
    )

    # Save the generated video
    for idx, output in enumerate(outputs):
        video_frames = output.outputs[0]  # Get the video frames
        
        if args.output_format == "mp4":
            output_path = output_dir / f"video_{idx:04d}.mp4"
            # Save as MP4 video
            import imageio
            imageio.mimsave(output_path, video_frames, fps=24, codec="libx264")
            print(f"\nSaved video to: {output_path}")
            
        elif args.output_format == "gif":
            output_path = output_dir / f"video_{idx:04d}.gif"
            # Save as GIF
            import imageio
            imageio.mimsave(output_path, video_frames, fps=12)
            print(f"\nSaved GIF to: {output_path}")
            
        else:  # frames
            frames_dir = output_dir / f"video_{idx:04d}_frames"
            frames_dir.mkdir(exist_ok=True)
            # Save individual frames
            for frame_idx, frame in enumerate(video_frames):
                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                Image.fromarray(frame).save(frame_path)
            print(f"\nSaved {len(video_frames)} frames to: {frames_dir}")

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
