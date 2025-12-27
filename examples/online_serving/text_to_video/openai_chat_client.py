#!/usr/bin/env python3
"""
Wan2.2 OpenAI-compatible chat client for video generation.

Usage:
    python openai_chat_client.py --prompt "A cat surfing on waves" --output output.mp4
    python openai_chat_client.py --image input.jpg --prompt "A cat dancing" --output i2v.mp4
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def _encode_image_as_data_url(input_path: Path) -> str:
    image_bytes = input_path.read_bytes()
    try:
        img = Image.open(BytesIO(image_bytes))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
    except Exception:
        mime_type = "image/png"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def generate_video(
    prompt: str,
    server_url: str = "http://localhost:8093",
    image_path: str | None = None,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    fps: int | None = None,
    num_outputs_per_prompt: int | None = None,
) -> bytes | None:
    """Generate a video using the chat completions API."""
    if image_path:
        image_url = _encode_image_as_data_url(Path(image_path))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    extra_body = {}
    if height is not None:
        extra_body["height"] = height
    if width is not None:
        extra_body["width"] = width
    if num_frames is not None:
        extra_body["num_frames"] = num_frames
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
    if fps is not None:
        extra_body["fps"] = fps
    if num_outputs_per_prompt is not None:
        extra_body["num_outputs_per_prompt"] = num_outputs_per_prompt

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
                if item.get("type") == "video_url":
                    url = item.get("video_url", {}).get("url", "")
                    if url.startswith("data:video"):
                        _, b64_data = url.split(",", 1)
                        return base64.b64decode(b64_data)
        print(f"Unexpected response format: {content}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 chat client for video generation")
    parser.add_argument("--prompt", "-p", default="A cat surfing on waves", help="Text prompt")
    parser.add_argument("--image", "-i", help="Optional input image path for image-to-video")
    parser.add_argument("--output", "-o", default="wan22_output.mp4", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8093", help="Server URL")
    parser.add_argument("--height", type=int, help="Video height")
    parser.add_argument("--width", type=int, help="Video width")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--guidance-high", type=float, help="High-noise guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--num-outputs", type=int, help="Number of videos to generate")

    args = parser.parse_args()

    print(f"Generating video for: {args.prompt}")
    if args.image:
        print(f"Using image: {args.image}")

    video_bytes = generate_video(
        prompt=args.prompt,
        server_url=args.server,
        image_path=args.image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps=args.steps,
        guidance_scale=args.guidance,
        guidance_scale_2=args.guidance_high,
        seed=args.seed,
        negative_prompt=args.negative,
        fps=args.fps,
        num_outputs_per_prompt=args.num_outputs,
    )

    if video_bytes:
        output_path = Path(args.output)
        output_path.write_bytes(video_bytes)
        print(f"Video saved to: {output_path}")
        print(f"Size: {len(video_bytes) / (1024 * 1024):.1f} MB")
    else:
        print("Failed to generate video")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
