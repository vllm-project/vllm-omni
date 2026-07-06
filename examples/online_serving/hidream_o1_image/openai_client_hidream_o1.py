# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image OpenAI-compatible image generation client.

Usage:
    # Text-to-image
    python openai_client_hidream_o1.py \
        --prompt "A golden retriever in a field of sunflowers" \
        --output output.png

    # Override steps / resolution
    python openai_client_hidream_o1.py \
        --prompt "A cinematic mountain landscape" \
        --height 1024 --width 1024 \
        --steps 28 \
        --seed 42 \
        --output landscape.png
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8095",
    height: int = 1024,
    width: int = 1024,
    steps: int = 28,
    guidance_scale: float | None = None,
    seed: int = 42,
) -> bytes | None:
    payload: dict = {
        "prompt": prompt,
        "response_format": "b64_json",
        "size": f"{width}x{height}",
        "num_inference_steps": steps,
        "seed": seed,
    }
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale

    try:
        response = requests.post(
            f"{server_url}/v1/images/generations",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("data")
        if isinstance(items, list) and items:
            b64 = items[0].get("b64_json") if isinstance(items[0], dict) else None
            if isinstance(b64, str):
                return base64.b64decode(b64)
        print(f"Unexpected response format: {data}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="HiDream-O1-Image online serving client.")
    p.add_argument("--prompt", default="A golden retriever in a field of sunflowers")
    p.add_argument("--server", default="http://localhost:8095")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Omit for Dev (no CFG). Set to 5.0 for the Full variant.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="hidream_o1_output.png")
    args = p.parse_args()

    print(f"Generating: {args.prompt}")
    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    if image_bytes:
        out = Path(args.output)
        out.write_bytes(image_bytes)
        print(f"Saved to {out} ({len(image_bytes) / 1024:.1f} KB)")
    else:
        print("Failed to generate image.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
