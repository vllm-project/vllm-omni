#!/usr/bin/env python3
"""
Qwen-Image OpenAI-compatible chat client for image generation.

Usage:
    python openai_chat_client.py --prompt "A beautiful landscape" --output output.png
    python openai_chat_client.py --prompt "A sunset" --size 1024x1024 --steps 50 --seed 42
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8091",
    size: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> bytes | None:
    """Generate an image using the chat completions API.

    Args:
        prompt: Text description of the image
        server_url: Server URL
        size: Image size (e.g., "1024x1024")
        steps: Number of inference steps
        guidance: Guidance scale
        seed: Random seed
        negative_prompt: Negative prompt

    Returns:
        Image bytes or None if failed
    """
    messages = []

    # Build system message with parameters
    params = []
    if size:
        params.append(f"size={size}")
    if steps:
        params.append(f"steps={steps}")
    if guidance:
        params.append(f"guidance={guidance}")
    if seed is not None:
        params.append(f"seed={seed}")
    if negative_prompt:
        params.append(f"negative={negative_prompt}")

    if params:
        messages.append({"role": "system", "content": " ".join(params)})

    # Add user message
    messages.append({"role": "user", "content": prompt})

    # Send request
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"messages": messages},
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
    parser.add_argument("--size", default="1024x1024", help="Image size (e.g., 1024x1024)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    print(f"Generating image for: {args.prompt}")

    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        size=args.size,
        steps=args.steps,
        guidance=args.guidance,
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
