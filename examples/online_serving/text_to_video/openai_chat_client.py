#!/usr/bin/env python3
"""
Wan2.2 OpenAI-compatible chat client for text-to-video generation.

Usage:
    python openai_chat_client.py --prompt "A cinematic shot..." --output output.mp4
    python openai_chat_client.py --num-frames 81 --fps 24 --height 720 --width 1280
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_video(
    prompt: str,
    server_url: str = "http://localhost:8093",
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    fps: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> bytes | None:
    """Generate a video using the chat completions API."""
    messages = [{"role": "user", "content": prompt}]

    extra_body = {}
    if height is not None:
        extra_body["height"] = height
    if width is not None:
        extra_body["width"] = width
    if num_frames is not None:
        extra_body["num_frames"] = num_frames
    if fps is not None:
        extra_body["fps"] = fps
    if steps is not None:
        extra_body["num_inference_steps"] = steps
    if guidance_scale is not None:
        extra_body["guidance_scale"] = guidance_scale
    if guidance_scale_2 is not None:
        extra_body["guidance_scale_2"] = guidance_scale_2
    if seed is not None:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt

    payload = {"messages": messages}
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            for item in content:
                video_url = item.get("video_url", {}).get("url", "")
                if video_url.startswith("data:video"):
                    _, b64_data = video_url.split(",", 1)
                    return base64.b64decode(b64_data)

        print(f"Unexpected response format: {content}")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 chat client")
    parser.add_argument("--prompt", "-p", default="A cinematic shot of a flying kite over the ocean.")
    parser.add_argument("--output", "-o", default="wan22_output.mp4", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8093", help="Server URL")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG scale (low noise)")
    parser.add_argument("--cfg-scale-high", type=float, default=None, help="CFG scale (high noise)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative", default="", help="Negative prompt")

    args = parser.parse_args()

    print(f"Generating video for: {args.prompt}")

    video_bytes = generate_video(
        prompt=args.prompt,
        server_url=args.server,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        steps=args.steps,
        guidance_scale=args.cfg_scale,
        guidance_scale_2=args.cfg_scale_high,
        seed=args.seed,
        negative_prompt=args.negative,
    )

    if video_bytes:
        output_path = Path(args.output)
        output_path.write_bytes(video_bytes)
        print(f"Video saved to: {output_path}")
        print(f"Size: {len(video_bytes) / 1024 / 1024:.2f} MB")
    else:
        print("Failed to generate video")
        exit(1)


if __name__ == "__main__":
    main()
