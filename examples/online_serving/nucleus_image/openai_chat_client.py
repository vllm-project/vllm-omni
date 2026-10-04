#!/usr/bin/env python3
"""Nucleus-Image OpenAI-compatible image generation client."""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8091",
    size: str = "1024x1024",
    steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int | None = 42,
    negative_prompt: str | None = None,
    n: int = 1,
) -> bytes | None:
    payload: dict[str, object] = {
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
        "n": n,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

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
            first = items[0].get("b64_json") if isinstance(items[0], dict) else None
            if isinstance(first, str):
                return base64.b64decode(first)
        print(f"Unexpected response format: {data}")
        return None
    except Exception as exc:
        print(f"Error: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Nucleus-Image OpenAI client")
    parser.add_argument("--prompt", "-p", default="A weathered lighthouse on a rocky coastline at golden hour.")
    parser.add_argument("--output", "-o", default="nucleus_image_output.png")
    parser.add_argument("--server", "-s", default="http://localhost:8091")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative", default="blurry, low quality")
    parser.add_argument("--num-images", type=int, default=1)
    args = parser.parse_args()

    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative,
        n=args.num_images,
    )
    if not image_bytes:
        raise SystemExit(1)

    output_path = Path(args.output)
    output_path.write_bytes(image_bytes)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
