#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helios video generation client example using the /v1/videos API.

This example demonstrates how to use the generic extra_params field
to pass Helios-specific parameters without modifying the API.
"""

import argparse
import json
import time
from pathlib import Path

import requests


def create_video_job(
    api_url: str,
    prompt: str,
    model: str,
    width: int = 640,
    height: int = 384,
    num_frames: int = 99,
    guidance_scale: float = 5.0,
    seed: int = 42,
    extra_params: dict | None = None,
    input_image: Path | None = None,
) -> dict:
    """Create a video generation job."""
    data = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }

    files = {}
    if input_image:
        files["input_reference"] = open(input_image, "rb")

    if extra_params:
        data["extra_params"] = json.dumps(extra_params)

    response = requests.post(f"{api_url}/v1/videos", data=data, files=files)
    response.raise_for_status()

    if files:
        files["input_reference"].close()

    return response.json()


def poll_video_status(api_url: str, video_id: str, poll_interval: int = 2) -> dict:
    """Poll video generation status until completion."""
    print(f"Polling video job: {video_id}")

    while True:
        response = requests.get(f"{api_url}/v1/videos/{video_id}")
        response.raise_for_status()
        status_data = response.json()

        status = status_data["status"]
        progress = status_data.get("progress", 0)

        print(f"Status: {status}, Progress: {progress}%")

        if status == "completed":
            print("Video generation completed!")
            return status_data
        elif status == "failed":
            error = status_data.get("error", {})
            raise RuntimeError(f"Video generation failed: {error}")

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Helios video generation client")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API server URL",
    )
    parser.add_argument(
        "--model",
        default="BestWishYsh/Helios-Base",
        help="Model name (Helios-Base, Helios-Mid, Helios-Distilled)",
    )
    parser.add_argument(
        "--prompt",
        default="A serene lakeside sunrise with mist over the water.",
        help="Text prompt",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="Video height",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=99,
        help="Number of frames",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        help="Input image for I2V mode",
    )
    parser.add_argument(
        "--preset",
        choices=["base", "mid-stage2", "distilled"],
        default="base",
        help="Helios preset configuration",
    )

    args = parser.parse_args()

    # Define preset configurations
    presets = {
        "base": None,  # No extra params for base model
        "mid-stage2": {
            "is_enable_stage2": True,
            "pyramid_num_stages": 3,
            "pyramid_num_inference_steps_list": [20, 20, 20],
            "use_cfg_zero_star": True,
            "use_zero_init": True,
            "zero_steps": 1,
        },
        "distilled": {
            "is_enable_stage2": True,
            "pyramid_num_stages": 3,
            "pyramid_num_inference_steps_list": [2, 2, 2],
            "is_amplify_first_chunk": True,
        },
    }

    extra_params = presets[args.preset]

    print("=" * 50)
    print("Helios Video Generation")
    print("=" * 50)
    print(f"API URL: {args.api_url}")
    print(f"Model: {args.model}")
    print(f"Preset: {args.preset}")
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames}")
    print(f"Guidance Scale: {args.guidance_scale}")
    if extra_params:
        print(f"Extra Params: {json.dumps(extra_params, indent=2)}")
    print()

    # Create video job
    print("Creating video generation job...")
    job = create_video_job(
        api_url=args.api_url,
        prompt=args.prompt,
        model=args.model,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        extra_params=extra_params,
        input_image=args.input_image,
    )

    video_id = job["id"]
    print(f"Video job created: {video_id}")
    print()

    # Poll for completion
    result = poll_video_status(args.api_url, video_id)

    print()
    print("=" * 50)
    print("Generation Complete")
    print("=" * 50)
    print(f"Video ID: {result['id']}")
    print(f"File: {result.get('file_name', 'N/A')}")
    print(f"Inference Time: {result.get('inference_time_s', 'N/A')}s")
    print()


if __name__ == "__main__":
    main()
