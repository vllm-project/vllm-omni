#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX 3 Image Klein prompt upsampler (pup) OpenAI-compatible client.

Expands a short request into the detailed caption FLUX 3 Image Klein expects,
using black-forest-labs/flux-3-image-klein-pup served by vLLM. Without
--image the t2i system prompt is used; with one or more --image the i2i
prompt and the reference-image scaffold from the model card. --bbox switches
to the t2i_bbox / i2i_bbox prompts, which return the caption followed by a
JSON element table of bounding boxes ([y_min, x_min, y_max, x_max] on a
0-1000 grid); --boxes passes your own boxes as JSON after the request.
System prompts are resolved from the Hugging Face repo (system_prompts.json).

Usage:
  python openai_client_pup.py --prompt "A man walking with his dog, afternoon sun"
  python openai_client_pup.py --prompt "put a red bandana around the dog's neck" --image dog.jpg
  python openai_client_pup.py --prompt "A man walking with his dog, afternoon sun" --bbox
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

PUP_REPO = "black-forest-labs/flux-3-image-klein-pup"
I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {request}\n"
    "Output the caption."
)


def _to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def main() -> int:
    p = argparse.ArgumentParser(description="FLUX 3 Image Klein prompt upsampler client")
    p.add_argument("--prompt", "-p", required=True, help="Short image request to upsample.")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help="Reference image path; repeatable. Selects editing mode.",
    )
    p.add_argument("--bbox", action="store_true", help="Use the *_bbox prompts: caption followed by a JSON box table.")
    p.add_argument("--boxes", help="JSON array of boxes to honor, appended after the request (implies --bbox).")
    p.add_argument("--server", "-s", default="http://localhost:8000")
    p.add_argument("--model", default="flux-3-image-klein-pup")
    p.add_argument("--max-tokens", type=int, default=3584)
    args = p.parse_args()

    prompts = json.loads(Path(hf_hub_download(PUP_REPO, "system_prompts.json")).read_text())
    mode = ("i2i" if args.image else "t2i") + ("_bbox" if args.bbox or args.boxes else "")
    request = args.prompt if args.boxes is None else f"{args.prompt}\n{json.dumps(json.loads(args.boxes))}"
    if args.image:
        system = prompts[mode]
        content = [{"type": "image_url", "image_url": {"url": _to_data_uri(img)}} for img in args.image]
        content.append({"type": "text", "text": I2I_SCAFFOLD.format(request=request)})
    else:
        system, content = prompts[mode], request

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        # model card: presence penalty 1.5 for text-to-image, 0 for edits
        "presence_penalty": 0 if args.image else 1.5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{args.server}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    print(r.json()["choices"][0]["message"]["content"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
