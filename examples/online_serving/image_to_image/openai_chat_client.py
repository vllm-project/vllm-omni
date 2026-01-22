#!/usr/bin/env python3
"""
Qwen-Image-Edit OpenAI-compatible chat client for image editing.

Usage:
    python openai_chat_client.py --input qwen_image_output.png --prompt "Convert to watercolor style" --output output.png
    python openai_chat_client.py --input input.png --prompt "Convert to oil painting" --seed 42
    python openai_chat_client.py --input input1.png input2.png --prompt "Combine these images into a single scene"
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


def edit_image(
    input_image: str | Path | list[str | Path],
    prompt: str,
    server_url: str = "http://localhost:8092",
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> bytes | None:
    """Edit an image using the chat completions API.

    Args:
        input_image: Path(s) to input image(s). For multi-image editing, pass multiple paths.
        prompt: Text description of the edit
        server_url: Server URL
        height: Output image height in pixels
        width: Output image width in pixels
        steps: Number of inference steps
        guidance_scale: CFG guidance scale
        seed: Random seed
        negative_prompt: Negative prompt

    Returns:
        Edited image bytes or None if failed
    """
    input_images = input_image if isinstance(input_image, list) else [input_image]
    input_paths = [Path(p) for p in input_images]
    for p in input_paths:
        if not p.exists():
            print(f"Error: Input image not found: {p}")
            return None

    # Build user message with text and image
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for p in input_paths:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_as_data_url(p)}})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Build extra_body with generation parameters
    extra_body = {}
    if steps is not None:
        extra_body["num_inference_steps"] = steps
    if guidance_scale is not None:
        extra_body["guidance_scale"] = guidance_scale
    if seed is not None:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt

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
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit chat client")
    parser.add_argument("--input", "-i", required=True, nargs="+", help="Input image path(s)")
    parser.add_argument("--prompt", "-p", required=True, help="Edit prompt")
    parser.add_argument("--output", "-o", default="output.png", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8092", help="Server URL")
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    if len(args.input) == 1:
        print(f"Input: {args.input[0]}")
    else:
        print(f"Inputs ({len(args.input)}): {', '.join(args.input)}")
    print(f"Prompt: {args.prompt}")

    image_bytes = edit_image(
        input_image=args.input,
        prompt=args.prompt,
        server_url=args.server,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance,
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
