#!/usr/bin/env python3
"""
Qwen-Image OpenAI-compatible chat client for image generation.

Usage:
    python openai_chat_client.py --prompt "A beautiful landscape" --output output.png
    python openai_chat_client.py --prompt "A sunset" --height 1024 --width 1024 --steps 50 --seed 42
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8091",
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    true_cfg_scale: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    num_outputs_per_prompt: int = 1,
) -> bytes | None:
    """Generate an image using the chat completions API.

    Args:
        prompt: Text description of the image
        server_url: Server URL
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        true_cfg_scale: Qwen-Image CFG scale
        seed: Random seed
        negative_prompt: Negative prompt
        num_outputs_per_prompt: Number of images to generate

    Returns:
        Image bytes or None if failed
    """
    messages = [{"role": "user", "content": prompt}]

    # Build extra_body with generation parameters
    extra_body = {}
    if height is not None:
        extra_body["height"] = height
    if width is not None:
        extra_body["width"] = width
    if steps is not None:
        extra_body["num_inference_steps"] = steps
    if true_cfg_scale is not None:
        extra_body["true_cfg_scale"] = true_cfg_scale
    if seed is not None:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt
    extra_body["num_outputs_per_prompt"] = num_outputs_per_prompt

    # Build request payload
    payload = {"messages": messages}
    if extra_body:
        payload["extra_body"] = extra_body

    # Send request
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        # Extract image from response
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list) and len(content) > 0:
            image_url = content[0].get("image_url", {}).get("url", "")
            if image_url.startswith("data:image"):
                _, b64_data = image_url.split(",", 1)
                return base64.b64decode(b64_data)

        print(f"Unexpected response format: {content}")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image chat client")
    parser.add_argument("--prompt", "-p", default="a cup of coffee on the table", help="Text prompt")
    parser.add_argument("--output", "-o", default="qwen_image_output.png", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    print(f"Generating image for: {args.prompt}")

    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        height=args.height,
        width=args.width,
        steps=args.steps,
        true_cfg_scale=args.cfg_scale,
        seed=args.seed,
        negative_prompt=args.negative,
    )

    if image_bytes:
        output_path = Path(args.output)
        output_path.write_bytes(image_bytes)
        print(f"Image saved to: {output_path}")
        print(f"Size: {len(image_bytes) / 1024:.1f} KB")
    else:
        print("Failed to generate image")
        exit(1)


if __name__ == "__main__":
    main()
