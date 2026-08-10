#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for all supported Omni-Diffusion tasks."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset

TASKS = ("t2i", "vqa", "asr", "tts", "s2i", "svqa")
TEXT_TASKS = frozenset({"vqa", "asr", "svqa"})

DEFAULT_PROMPTS = {
    "t2i": (
        "A super realistic and hyper-detailed 8K fantasy night scene showing "
        "an amazing beach under the full moon, lit by dramatic lighting."
    ),
    "vqa": "Describe the image in detail.",
    "asr": "Convert the speech to text.",
    "tts": "Get the trust fund to the bank early.",
    "s2i": "Please generate an image based on the input audio.",
    "svqa": "Please respond to the input audio based on the given image.",
}


def _file_data_url(value: str | None, option: str, media_prefix: str) -> str:
    if value is None:
        path = (
            AudioAsset("mary_had_lamb").get_local_path()
            if media_prefix == "audio"
            else ImageAsset("cherry_blossom").get_path("jpg")
        )
        value = str(path)
    if value.startswith("data:"):
        return value

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{option} does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or f"{media_prefix}/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _format_task_prompt(task: str, prompt: str) -> str:
    """Add the model instruction expected by text-input generation tasks."""
    if task == "t2i":
        return f"Generate an image based on the provided text description.\n{prompt}"
    if task == "tts":
        return f"Convert the text to speech.\n{prompt}"
    return prompt


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _format_task_prompt(args.task, args.prompt or DEFAULT_PROMPTS[args.task])
    content: list[dict[str, Any]] = [] if args.task == "svqa" else [{"type": "text", "text": prompt}]
    messages: list[dict[str, Any]] = []

    if args.task in {"asr", "s2i", "svqa"}:
        content.append(
            {
                "type": "audio_url",
                "audio_url": {"url": _file_data_url(args.audio_path, "--audio-path", "audio")},
            }
        )
    if args.task in {"vqa", "svqa"}:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _file_data_url(args.image_path, "--image-path", "image")},
            }
        )
    if args.task == "svqa":
        messages.append(
            {
                "role": "system",
                "content": prompt,
            }
        )
    messages.append({"role": "user", "content": content})

    output_modality = "text" if args.task in TEXT_TASKS else "audio" if args.task == "tts" else "image"
    max_tokens = 128 if args.task in TEXT_TASKS else 50 if args.task == "tts" else 260
    return {
        "model": args.model,
        "messages": messages,
        "modalities": [output_modality],
        "max_tokens": max_tokens,
    }


def _first_image_data_url(response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"].get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            value = item.get("image_url", {}).get("url")
            if isinstance(value, str) and value.startswith("data:image"):
                return value
    raise RuntimeError("Omni-Diffusion response did not contain an image data URL.")


def save_response(task: str, response: dict[str, Any], output_path: Path) -> None:
    if "error" in response:
        raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    message = response["choices"][0]["message"]
    if task in TEXT_TASKS:
        text = str(message.get("content") or "")
        output_path.write_text(text, encoding="utf-8")
        print(text)
    elif task == "tts":
        audio_data = (message.get("audio") or {}).get("data")
        if not isinstance(audio_data, str) or not audio_data:
            raise RuntimeError("Omni-Diffusion response did not contain audio data.")
        output_path.write_bytes(base64.b64decode(audio_data))
    else:
        data_url = _first_image_data_url(response)
        output_path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))

    print(f"Saved {task.upper()} output to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", default="lijiang/Omni-Diffusion")
    parser.add_argument("--base-url", default="http://localhost:8091")
    parser.add_argument("--prompt", help="Task content; T2I and TTS instructions are added automatically.")
    parser.add_argument("--image-path")
    parser.add_argument("--audio-path")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_suffix = ".txt" if args.task in TEXT_TASKS else ".wav" if args.task == "tts" else ".png"
    output_path = Path(args.output or f"/tmp/omni_diffusion_online/{args.task}{default_suffix}")

    response = requests.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=build_payload(args),
        timeout=args.timeout,
    )
    response.raise_for_status()
    save_response(args.task, response.json(), output_path)


if __name__ == "__main__":
    main()
