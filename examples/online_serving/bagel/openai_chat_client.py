#!/usr/bin/env python3
"""
Bagel OpenAI-compatible chat client for image generation and multimodal tasks.

Usage:
    python openai_chat_client.py --prompt "A cute cat" --output output.png
    python openai_chat_client.py --prompt "Describe this image" --image-url https://example.com/image.png
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8091",
    image_url: str | None = None,
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    modality: str = "text2img",  # "text2img" (default), "img2img", "img2text", "text2text"
) -> bytes | str | None:
    """Generate an image or text using the chat completions API.

    Args:
        prompt: Text description or prompt
        server_url: Server URL
        image_url: URL or path to input image (for img2img/img2text)
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        seed: Random seed
        negative_prompt: Negative prompt
        modality: Task modality hint

    Returns:
        Image bytes (for image outputs) or Text string (for text outputs) or None if failed
    """

    # Construct Message Content
    content = [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}]

    if image_url:
        # Check if local file
        if Path(image_url).exists():
            with open(image_url, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
                final_image_url = f"data:image/jpeg;base64,{b64_data}"
        else:
            final_image_url = image_url

        content.append({"type": "image_url", "image_url": {"url": final_image_url}})

    messages = [{"role": "user", "content": content}]

    # Build request payload with all parameters at top level
    # Note: vLLM ignores "extra_body", so we put parameters directly in the payload
    payload = {"messages": messages}

    # Set output modalities at top level
    if modality == "text2img" or modality == "img2img":
        payload["modalities"] = ["image"]
    elif modality == "img2text" or modality == "text2text":
        payload["modalities"] = ["text"]

    # Add generation parameters directly to payload
    if height is not None:
        payload["height"] = height
    if width is not None:
        payload["width"] = width
    if steps is not None:
        payload["num_inference_steps"] = steps
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    # Send request
    try:
        print(f"Sending request to {server_url} with modality {modality}...")
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        # Extract content - check ALL choices since server may return multiple
        # (e.g., text in choices[0], image in choices[1])
        choices = data.get("choices", [])

        # First pass: look for image output in any choice
        for choice in choices:
            choice_content = choice.get("message", {}).get("content")

            # Handle Image Output
            if isinstance(choice_content, list) and len(choice_content) > 0:
                first_item = choice_content[0]
                if isinstance(first_item, dict) and "image_url" in first_item:
                    img_url_str = first_item["image_url"].get("url", "")
                    if img_url_str.startswith("data:image"):
                        _, b64_data = img_url_str.split(",", 1)
                        return base64.b64decode(b64_data)

        # Second pass: look for text output if no image found
        for choice in choices:
            choice_content = choice.get("message", {}).get("content")
            if isinstance(choice_content, str) and choice_content:
                return choice_content

        print(f"Unexpected response format: {choices}")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Bagel multimodal chat client")
    parser.add_argument("--prompt", "-p", default="<|im_start|>A cute cat<|im_end|>", help="Text prompt")
    parser.add_argument("--output", "-o", default="bagel_output.png", help="Output file (for image results)")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")

    # Modality Control
    parser.add_argument("--image-url", "-i", type=str, help="Input image URL or local path")
    parser.add_argument(
        "--modality",
        "-m",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
        help="Task modality",
    )

    # Generation Params
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    print(f"Mode: {args.modality}")
    if args.image_url:
        print(f"Input Image: {args.image_url}")

    result = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        image_url=args.image_url,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        negative_prompt=args.negative,
        modality=args.modality,
    )

    if result:
        if isinstance(result, bytes):
            # It's an image
            output_path = Path(args.output)
            output_path.write_bytes(result)
            print(f"Image saved to: {output_path}")
            print(f"Size: {len(result) / 1024:.1f} KB")
        elif isinstance(result, str):
            # It's text
            print("Response:")
            print(result)
    else:
        print("Failed to generate response")
        exit(1)


if __name__ == "__main__":
    main()
