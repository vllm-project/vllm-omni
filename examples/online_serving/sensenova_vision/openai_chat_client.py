#!/usr/bin/env python3
"""
SenseNova-Vision OpenAI-compatible chat client.

Demonstrates the SenseNova-Vision modality matrix against an
``vllm-omni serve`` endpoint.  At minimum the client exercises:

- mixed (caption_generate): image + intermediate caption text in one response
- img2text: image understanding via the OpenAI chat completions API
- text2img: image generation

Usage:
    # Text-to-image
    python openai_chat_client.py --modality text2img \
        --prompt "A cute corgi astronaut on the moon, cinematic" \
        --output sensenova_text2img.png

    # Image understanding (img2text)
    python openai_chat_client.py --modality img2text \
        --image-url /path/to/photo.jpg \
        --prompt "What are the main objects in this scene and their relationships?"

    # Mixed text + image (caption_generate)
    python openai_chat_client.py --modality mixed \
        --image-url /path/to/photo.jpg \
        --output sensenova_mixed.png

    # Image editing (img2img)
    python openai_chat_client.py --modality img2img \
        --image-url /path/to/photo.jpg \
        --prompt "Turn this image into a vibrant cartoon-style illustration." \
        --output sensenova_img2img.png

    # Text-to-text (chat)
    python openai_chat_client.py --modality text2text \
        --prompt "What is the capital of France?"
"""

import argparse
import base64
from pathlib import Path

import requests


def _encode_image(image_url: str) -> str:
    """Encode a local file or URL to a base64 data URI."""
    if Path(image_url).exists():
        with open(image_url, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    return image_url


def generate(
    prompt: str,
    server_url: str = "http://localhost:8092",
    image_url: str | None = None,
    modality: str = "text2img",
    **kwargs: object,
) -> tuple[bytes | None, str | None]:
    """Send a request to the SenseNova-Vision server.

    All keyword arguments (height, width, seed, num_inference_steps, ...) are
    forwarded as top-level fields in the request payload.  The serving layer
    maps standard keys to sampling params and forwards model-specific keys
    (cfg_text_scale, cfg_img_scale, timestep_shift, ...) to ``extra_args``
    through the model extra registry.

    Returns:
        ``(image_bytes, text)``.  Exactly one of the two is non-None for a
        single-modality request; the ``mixed`` (caption_generate) mode may
        return both.
    """
    content = [{"type": "text", "text": prompt}]

    if image_url:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(image_url)}})

    messages = [{"role": "user", "content": content}]
    payload: dict = {"messages": messages}

    if modality in ("text2img", "img2img", "mixed"):
        payload["modalities"] = ["image"]
    else:
        payload["modalities"] = ["text"]

    for key, val in kwargs.items():
        if val is not None and val is not False:
            payload[key] = val

    try:
        print(f"Sending {modality} request to {server_url}...")
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

        metrics = data.get("metrics") or {}
        think_text = metrics.get("think_text")
        if think_text:
            print(f"\n[Think]\n{think_text}\n")

        image_bytes: bytes | None = None
        text: str | None = None

        choices = data.get("choices", [])
        for choice in choices:
            choice_content = choice.get("message", {}).get("content")
            if isinstance(choice_content, list):
                for item in choice_content:
                    if isinstance(item, dict) and "image_url" in item:
                        img_url_str = item["image_url"].get("url", "")
                        if img_url_str.startswith("data:image"):
                            _, b64_data = img_url_str.split(",", 1)
                            image_bytes = base64.b64decode(b64_data)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text") or text
            elif isinstance(choice_content, str) and choice_content:
                text = choice_content

        return image_bytes, text

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="SenseNova-Vision multimodal chat client")
    parser.add_argument("--prompt", "-p", default=None, help="Text prompt (official per-mode default if omitted)")
    parser.add_argument("--output", "-o", default="sensenova_output.png", help="Output file (for image results)")
    parser.add_argument("--server", "-s", default="http://localhost:8092", help="Server URL")
    parser.add_argument("--image-url", "-i", type=str, help="Input image URL or local path")
    parser.add_argument(
        "--modality",
        "-m",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text", "mixed"],
        help="Task modality",
    )
    # Standard generation parameters
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--num-steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Model-specific parameters forwarded via the model extra registry.
    parser.add_argument("--cfg-text-scale", type=float, default=None, help="Text CFG scale")
    parser.add_argument("--cfg-img-scale", type=float, default=None, help="Image CFG scale")
    parser.add_argument("--timestep-shift", type=float, default=None, help="Timestep shift")
    parser.add_argument("--max-think-tokens", type=int, default=None, help="Max think tokens")

    args = parser.parse_args()

    default_prompts = {
        "text2img": "A cute corgi astronaut on the moon, cinematic",
        "img2img": "Turn this image into a vibrant cartoon-style illustration.",
        "img2text": "What are the main objects in this scene and their relationships?",
        "text2text": "What is the capital of France?",
        "mixed": (
            "<image> Please briefly describe the contents of the image. Please respond "
            "with interleaved segmentation masks for the corresponding parts of the answer."
        ),
    }
    prompt = args.prompt or default_prompts[args.modality]

    print(f"Mode: {args.modality}")
    if args.image_url:
        print(f"Input Image: {args.image_url}")

    extra: dict[str, object] = {
        "seed": args.seed,
    }
    if args.modality in ("text2img", "img2img", "mixed"):
        extra.update(
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
        )
    if args.cfg_text_scale is not None:
        extra["cfg_text_scale"] = args.cfg_text_scale
    if args.cfg_img_scale is not None:
        extra["cfg_img_scale"] = args.cfg_img_scale
    if args.timestep_shift is not None:
        extra["timestep_shift"] = args.timestep_shift
    if args.max_think_tokens is not None:
        extra["max_think_tokens"] = args.max_think_tokens

    image_bytes, text = generate(
        prompt=prompt,
        server_url=args.server,
        image_url=args.image_url,
        modality=args.modality,
        **extra,
    )

    saved = False
    if image_bytes:
        output_path = Path(args.output)
        output_path.write_bytes(image_bytes)
        print(f"Image saved to: {output_path}")
        print(f"Size: {len(image_bytes) / 1024:.1f} KB")
        saved = True
    if text:
        print(f"[Response]\n{text}")
        saved = True
    if not saved:
        print("Failed to generate response")
        exit(1)


if __name__ == "__main__":
    main()
