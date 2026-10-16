#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX 3 Image Klein OpenAI-compatible client with optional image conditioning.

Image modes use /v1/images/generations (T2I) or /v1/chat/completions (I2I).

Usage:
  python openai_client.py --prompt "alpine meadow at golden hour"
  python openai_client.py --prompt "restyle as watercolor" --image ref.jpg
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import requests

DEFAULT_PROMPT = "a serene alpine meadow at golden hour, wildflowers in the foreground"


def _to_data_uri(path: str) -> tuple[str, bytes]:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    data = Path(path).read_bytes()
    return mime, data


def _save_b64(b64: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def _sampling_overrides(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        key: value
        for key, value in (
            ("num_inference_steps", args.steps),
            ("guidance_scale", args.guidance_scale),
            ("flow_shift", args.flow_shift),
        )
        if value is not None
    }


def _lora_overrides(args: argparse.Namespace) -> dict:
    if not args.lora_path:
        return {}
    return {
        "lora": {
            "name": args.lora_name or Path(args.lora_path).name,
            "path": args.lora_path,
            "scale": args.lora_scale,
        }
    }


def run_t2i(args: argparse.Namespace) -> None:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "size": f"{args.width}x{args.height}",
        "n": 1,
        "response_format": "b64_json",
        "seed": args.seed,
        **_sampling_overrides(args),
        **_lora_overrides(args),
    }
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    r = requests.post(f"{args.server}/v1/images/generations", json=payload, timeout=300)
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]
    out = Path(args.output or "t2i.png")
    _save_b64(b64, out)
    print(f"[t2i] saved {out}")


def run_i2i(args: argparse.Namespace) -> None:
    image_content = []
    for path in args.image:
        mime, img_bytes = _to_data_uri(path)
        data_uri = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
        image_content.append({"type": "image_url", "image_url": {"url": data_uri}})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                *image_content,
            ],
        }
    ]
    payload = {
        "model": args.model,
        "messages": messages,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        **_sampling_overrides(args),
        **_lora_overrides(args),
    }
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    r = requests.post(f"{args.server}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    b64 = r.json()["choices"][0]["message"]["content"][0]["image_url"]["url"].split(",", 1)[1]
    out = Path(args.output or "i2i.png")
    _save_b64(b64, out)
    print(f"[i2i] saved {out}")


def main() -> int:
    p = argparse.ArgumentParser(description="FLUX 3 Image Klein OpenAI-compatible client")
    p.add_argument("--prompt", "-p", default=DEFAULT_PROMPT, help="Text prompt. Defaults to a built-in sample prompt.")
    p.add_argument("--negative-prompt", default="", help="Negative text prompt (empty string = unconditional).")
    p.add_argument(
        "--image", nargs="+", help="Optional conditioning image paths, in reference order; selects image-to-image."
    )
    p.add_argument("--output", "-o", default="", help="Output file path.")
    p.add_argument("--server", "-s", default="http://localhost:8099")
    p.add_argument("--model", default="flux-3-image-klein")
    p.add_argument("--height", type=int, default=1024, help="Output height in pixels.")
    p.add_argument("--width", type=int, default=1024, help="Output width in pixels.")
    p.add_argument("--steps", type=int, default=None, help="Sampling steps; omitted uses the model package default.")
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG scale; omitted uses the model package default. Unsupported by distilled packages.",
    )
    p.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Timestep shift (alpha); omitted uses the model package default.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-path", default=None, help="LoRA adapter path on the server; sent unchanged.")
    p.add_argument("--lora-name", default=None, help="Adapter label; defaults to the adapter path's final component.")
    p.add_argument("--lora-scale", type=float, default=1.0, help="Per-request LoRA scale (default: 1.0).")
    args = p.parse_args()
    if args.lora_name and not args.lora_path:
        p.error("--lora-name requires --lora-path")

    if args.image is not None:
        run_i2i(args)
    else:
        run_t2i(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
