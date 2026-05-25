#!/usr/bin/env python3
"""
SenseNova-U1 OpenAI-compatible chat client for all four task modalities.

Usage:
    # Text-to-image
    python openai_chat_client.py --prompt "A cute cat" --modality text2img

    # Image-to-image editing
    python openai_chat_client.py --prompt "Turn this into an oil painting" \
        --modality img2img --image-url /path/to/image.jpg

    # Image understanding (img2text)
    python openai_chat_client.py --prompt "Describe this image in detail" \
        --modality img2text --image-url /path/to/image.jpg

    # Text-to-text (chat)
    python openai_chat_client.py --prompt "What is the capital of France?" \
        --modality text2text
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
    server_url: str = "http://localhost:8091",
    image_url: str | None = None,
    modality: str = "text2img",
    **kwargs: object,
) -> bytes | str | None:
    """Send a request to the SenseNova-U1 server.

    All keyword arguments (height, width, seed, cfg_scale, think, …) are
    forwarded as top-level fields in the request payload.  The serving layer
    maps standard keys (height, width, seed, num_inference_steps) to sampling
    params and forwards model-specific keys (think, cfg_norm, img_cfg_scale,
    …) to ``extra_args`` based on the pipeline's ``EXTRA_BODY_PARAMS``.

    Returns:
        Image bytes (for image outputs) or text string (for text outputs).
    """
    content = [{"type": "text", "text": prompt}]

    if image_url:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(image_url)}})

    messages = [{"role": "user", "content": content}]
    payload: dict = {"messages": messages}

    if modality in ("text2img", "img2img"):
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

        # Print think text if present in response metrics
        metrics = data.get("metrics") or {}
        think_text = metrics.get("think_text")
        if think_text:
            print(f"\n[Think]\n{think_text}\n")

        choices = data.get("choices", [])

        for choice in choices:
            choice_content = choice.get("message", {}).get("content")

            if isinstance(choice_content, list) and len(choice_content) > 0:
                first_item = choice_content[0]
                if isinstance(first_item, dict) and "image_url" in first_item:
                    img_url_str = first_item["image_url"].get("url", "")
                    if img_url_str.startswith("data:image"):
                        _, b64_data = img_url_str.split(",", 1)
                        return base64.b64decode(b64_data)

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
    parser = argparse.ArgumentParser(description="SenseNova-U1 multimodal chat client")
    parser.add_argument("--prompt", "-p", default="A cute cat", help="Text prompt")
    parser.add_argument("--output", "-o", default="sensenova_u1_output.png", help="Output file (for image results)")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--image-url", "-i", type=str, help="Input image URL or local path")
    parser.add_argument(
        "--modality",
        "-m",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
        help="Task modality",
    )
    # Standard generation parameters
    parser.add_argument("--height", type=int, default=2048, help="Image height")
    parser.add_argument("--width", type=int, default=2048, help="Image width")
    parser.add_argument("--num-steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Model-specific parameters (forwarded via EXTRA_BODY_PARAMS)
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--img-cfg-scale", type=float, help="Image CFG scale (img2img)")
    parser.add_argument("--cfg-norm", type=str, help="CFG normalization mode")
    parser.add_argument("--timestep-shift", type=float, help="Timestep shift")
    parser.add_argument("--think", action="store_true", help="Enable think mode")

    args = parser.parse_args()
    is_image = args.modality in ("text2img", "img2img")

    print(f"Mode: {args.modality}")
    if args.image_url:
        print(f"Input Image: {args.image_url}")

    extra: dict[str, object] = {
        "seed": args.seed,
        "think": args.think or None,
    }
    if is_image:
        extra.update(
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )
        if args.img_cfg_scale is not None:
            extra["img_cfg_scale"] = args.img_cfg_scale
        if args.cfg_norm is not None:
            extra["cfg_norm"] = args.cfg_norm
        if args.timestep_shift is not None:
            extra["timestep_shift"] = args.timestep_shift

    result = generate(
        prompt=args.prompt,
        server_url=args.server,
        image_url=args.image_url,
        modality=args.modality,
        **extra,
    )

    if result:
        if isinstance(result, bytes):
            output_path = Path(args.output)
            output_path.write_bytes(result)
            print(f"Image saved to: {output_path}")
            print(f"Size: {len(result) / 1024:.1f} KB")
        elif isinstance(result, str):
            print(f"[Response]\n{result}")
    else:
        print("Failed to generate response")
        exit(1)


if __name__ == "__main__":
    main()
