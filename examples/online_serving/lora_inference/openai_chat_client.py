#!/usr/bin/env python3
"""
OpenAI-compatible chat client for diffusion LoRA inference.

Example:
  python openai_chat_client.py \
    --server http://localhost:8091 \
    --prompt "A piece of cheesecake" \
    --lora-path /path/to/lora_adapter \
    --lora-name my_lora \
    --lora-scale 1.0 \
    --output output.png
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    *,
    prompt: str,
    server_url: str,
    height: int | None,
    width: int | None,
    num_inference_steps: int | None,
    seed: int | None,
    lora_name: str | None,
    lora_path: str | None,
    lora_scale: float | None,
    lora_int_id: int | None,
) -> bytes | None:
    messages = [{"role": "user", "content": prompt}]

    extra_body: dict = {}
    if height is not None:
        extra_body["height"] = height
    if width is not None:
        extra_body["width"] = width
    if num_inference_steps is not None:
        extra_body["num_inference_steps"] = num_inference_steps
    if seed is not None:
        extra_body["seed"] = seed

    if lora_path:
        lora_body: dict = {
            "local_path": lora_path,
            "name": lora_name or Path(lora_path).stem,
        }
        if lora_scale is not None:
            lora_body["scale"] = float(lora_scale)
        if lora_int_id is not None:
            lora_body["int_id"] = int(lora_int_id)
        extra_body["lora"] = lora_body

    payload = {"messages": messages}
    if extra_body:
        payload["extra_body"] = extra_body

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list) and content:
        image_url = content[0].get("image_url", {}).get("url", "")
        if image_url.startswith("data:image"):
            _, b64_data = image_url.split(",", 1)
            return base64.b64decode(b64_data)

    raise RuntimeError(f"Unexpected response format: {content!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion LoRA OpenAI chat client")
    parser.add_argument("--server", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--prompt", default="A piece of cheesecake", help="Text prompt")
    parser.add_argument("--output", default="lora_online_output.png", help="Output image path")

    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--lora-path", default=None, help="Server-local LoRA adapter folder (PEFT format)")
    parser.add_argument("--lora-name", default=None, help="LoRA name (optional)")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")
    parser.add_argument(
        "--lora-int-id",
        type=int,
        default=None,
        help="LoRA integer id (cache key). If omitted, the server derives a stable id from lora_path.",
    )

    args = parser.parse_args()

    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        seed=args.seed,
        lora_name=args.lora_name,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale if args.lora_path else None,
        lora_int_id=args.lora_int_id if args.lora_path else None,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_bytes)
    print(f"Saved: {out_path} ({len(image_bytes) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
