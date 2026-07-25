# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
InternVL-U OpenAI-compatible chat client for all four task modalities.

Usage:
    # Text-to-image
    python openai_chat_client.py --prompt "A cute cat" --modality text2img

    # Image editing
    python openai_chat_client.py --prompt "Add a red scarf around the cat's neck" \
        --modality img2img --image-url cat.png

    # Image understanding
    python openai_chat_client.py --prompt "Describe this image in detail" \
        --modality img2text --image-url photo.jpg

    # Text chat
    python openai_chat_client.py --prompt "What is the capital of France?" \
        --modality text2text

Text-then-image (think mode) is selected by the server deployment
(``THINK=1 bash run_server.sh``); requests need no extra fields.  Chat
completions return the image only; to receive the generated chain-of-thought
text, use the REST image endpoints (``/v1/images/generations`` and
``/v1/images/edits``), which return it as a top-level ``cot_output`` field.
"""

import argparse
import base64
from pathlib import Path

import requests


def _encode_image(image_url: str) -> str:
    """Encode a local file or remote URL to a base64 data URI.

    The chat media extractor only accepts ``data:image`` URIs, so remote
    references must be downloaded client-side; passing an HTTP(S) URL through
    would silently drop the reference image and run the request as text2img.
    """
    if Path(image_url).exists():
        with open(image_url, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"
    if image_url.startswith(("http://", "https://")):
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        mime = response.headers.get("Content-Type", "")
        if not mime.startswith("image/"):
            mime = "image/png"
        b64_data = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime};base64,{b64_data}"
    if image_url.startswith("data:image"):
        return image_url
    raise ValueError(f"--image-url must be a local file, an http(s) URL, or a data URI; got {image_url!r}")


def generate(
    prompt: str,
    server_url: str,
    image_url: str | None,
    modality: str,
    **kwargs: object,
) -> bytes | str | None:
    """Send one chat completion request and return image bytes or text.

    Standard keys (height, width, seed, num_inference_steps) map to sampling
    params; InternVL-U's declared extra-body params (all_cfg_scale,
    part_cfg_scale, timestep_trunc, flow_shift) are forwarded to the
    diffusion stage.
    """
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(image_url)}})

    payload: dict = {
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image"] if modality in ("text2img", "img2img") else ["text"],
    }
    for key, value in kwargs.items():
        if value is not None:
            payload[key] = value

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=900,
    )
    response.raise_for_status()
    data = response.json()

    image_bytes: bytes | None = None
    for choice in data.get("choices", []):
        choice_content = choice.get("message", {}).get("content")
        if isinstance(choice_content, list):
            for item in choice_content:
                if not isinstance(item, dict):
                    continue
                if "image_url" in item:
                    url = item["image_url"].get("url", "")
                    if url.startswith("data:image"):
                        image_bytes = base64.b64decode(url.split(",", 1)[1])
                elif item.get("type") == "text" and item.get("text"):
                    # CoT deployments emit the generated description as text
                    # alongside the image.
                    print(f"\n[Text]\n{item['text']}\n")
        elif isinstance(choice_content, str) and choice_content:
            return choice_content
    if image_bytes is not None:
        return image_bytes
    print(f"Unexpected response format: {data.get('choices')}")
    return None


def main():
    parser = argparse.ArgumentParser(description="InternVL-U multimodal chat client")
    parser.add_argument("--prompt", "-p", default="A cute cat", help="Text prompt")
    parser.add_argument("--output", "-o", default="internvlu_output.png", help="Output file for image results")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--image-url", "-i", type=str, help="Input image URL or local path")
    parser.add_argument(
        "--modality",
        "-m",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
        help="Task modality",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--num-steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Declared model-specific extra-body params.
    parser.add_argument("--all-cfg-scale", type=float, help="Conditional guidance scale")
    parser.add_argument("--part-cfg-scale", type=float, help="Partial guidance scale")
    parser.add_argument("--timestep-trunc", type=float, help="High-timestep CFG normalization cutoff")
    parser.add_argument("--flow-shift", type=float, help="Scheduler flow shift")
    parser.add_argument("--max-tokens", type=int, help="Max tokens for text outputs")

    args = parser.parse_args()
    is_image = args.modality in ("text2img", "img2img")

    extra: dict[str, object] = {"seed": args.seed}
    if is_image:
        extra.update(
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            all_cfg_scale=args.all_cfg_scale,
            part_cfg_scale=args.part_cfg_scale,
            timestep_trunc=args.timestep_trunc,
            flow_shift=args.flow_shift,
        )
    elif args.max_tokens is not None:
        extra["max_tokens"] = args.max_tokens

    result = generate(
        args.prompt,
        server_url=args.server,
        image_url=args.image_url,
        modality=args.modality,
        **extra,
    )
    if result is None:
        raise SystemExit(1)
    if isinstance(result, bytes):
        Path(args.output).write_bytes(result)
        print(f"Saved image to {args.output}")
    else:
        print(f"[Response]\n{result}")


if __name__ == "__main__":
    main()
