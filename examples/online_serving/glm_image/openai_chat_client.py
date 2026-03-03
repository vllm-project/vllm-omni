#!/usr/bin/env python3
"""
GLM-Image OpenAI-compatible chat client for text-to-image and image-to-image.

Usage:
    # Text-to-image
    python openai_chat_client.py --prompt "A cute cat" --output output.png

    # Image-to-image (image editing)
    python openai_chat_client.py --prompt "Convert to watercolor style" --image input.png --output output.png
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8091",
    image_path: str | None = None,
    height: int = 1024,
    width: int = 1024,
    steps: int = 50,
    guidance_scale: float = 1.5,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> bytes | None:
    """Generate or edit an image using the chat completions API.

    Args:
        prompt: Text description or editing instruction
        server_url: Server URL
        image_path: Path to input image (for image-to-image editing)
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        negative_prompt: Negative prompt

    Returns:
        Image bytes or None if failed
    """
    # Build message content
    content: list[dict] = [{"type": "text", "text": prompt}]

    if image_path:
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None
        b64_data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        suffix = img_path.suffix.lstrip(".").lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64_data}"},
            }
        )

    messages = [{"role": "user", "content": content}]

    # Build request payload
    extra_body: dict = {
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }
    if seed is not None:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt

    payload = {"messages": messages, "extra_body": extra_body}

    # Send request
    try:
        mode = "image-to-image" if image_path else "text-to-image"
        print(f"Sending {mode} request to {server_url}...")
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

        # Extract image from response
        choices = data.get("choices", [])
        for choice in choices:
            choice_content = choice.get("message", {}).get("content")
            if isinstance(choice_content, list):
                for item in choice_content:
                    if isinstance(item, dict) and "image_url" in item:
                        img_url = item["image_url"].get("url", "")
                        if img_url.startswith("data:image"):
                            _, b64 = img_url.split(",", 1)
                            return base64.b64decode(b64)

        print(f"Unexpected response format: {data}")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="GLM-Image chat client")
    parser.add_argument(
        "--prompt",
        "-p",
        default="A beautiful sunset over the ocean with sailing boats",
        help="Text prompt",
    )
    parser.add_argument("--output", "-o", default="glm_image_output.png", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")

    # Image-to-image
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Input image path (for image-to-image editing)",
    )

    # Generation parameters
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    mode = "image-to-image" if args.image else "text-to-image"
    print(f"Mode: {mode}")
    print(f"Prompt: {args.prompt}")
    if args.image:
        print(f"Input image: {args.image}")

    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        image_path=args.image,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
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
