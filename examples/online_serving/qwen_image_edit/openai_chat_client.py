#!/usr/bin/env python3
"""
Qwen-Image-Edit OpenAI-compatible chat client for image editing.

Usage:
    python openai_chat_client.py --input qwen_image_output.png --prompt "Convert to watercolor style" --output output.png
    python openai_chat_client.py --input input.png --prompt "Convert to oil painting" --size 1024x1024 --seed 42
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def edit_image(
    input_image: str | Path,
    prompt: str,
    server_url: str = "http://localhost:8092",
    size: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> bytes | None:
    """Edit an image using the chat completions API.

    Args:
        input_image: Path to input image
        prompt: Text description of the edit
        server_url: Server URL
        size: Output image size (e.g., "1024x1024")
        steps: Number of inference steps
        guidance: Guidance scale
        seed: Random seed
        negative_prompt: Negative prompt

    Returns:
        Edited image bytes or None if failed
    """
    # Read and encode input image
    input_path = Path(input_image)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return None

    with open(input_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Detect image type
    img = Image.open(BytesIO(image_bytes))
    mime_type = f"image/{img.format.lower()}" if img.format else "image/png"

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

    # Add user message with text and image
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }
    )

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
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit chat client")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--prompt", "-p", required=True, help="Edit prompt")
    parser.add_argument("--output", "-o", default="output.png", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8092", help="Server URL")
    parser.add_argument("--size", help="Output image size (e.g., 1024x1024)")
    parser.add_argument("--steps", type=int, help="Inference steps")
    parser.add_argument("--guidance", type=float, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Prompt: {args.prompt}")

    image_bytes = edit_image(
        input_image=args.input,
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
        print("Failed to edit image")
        exit(1)


if __name__ == "__main__":
    main()
