#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "snu-aidas/Dynin-Omni"
DEFAULT_OUTPUT_DIR = "/tmp/dynin_online_outputs"

QUERY_CHOICES = ("t2i", "t2s", "i2i")
DEFAULT_PROMPT_BY_QUERY = {
    "t2i": "A high quality detailed living room interior photo.",
    "t2s": "Please read this sentence naturally: Hello from Dynin-Omni online serving.",
    "i2i": "Transform this image into a realistic indoor living room while preserving layout.",
}
DEFAULT_MODALITIES_BY_QUERY = {
    "t2i": ["image"],
    "t2s": ["audio"],
    "i2i": ["image"],
}
OFFLINE_PARITY_STAGE_COUNT = 3
OFFLINE_PARITY_STAGE_SAMPLING = {
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "detokenize": False,
}


def _infer_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _encode_file_as_data_url(path: Path) -> str:
    mime_type = _infer_mime_type(path)
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _to_image_url(path_or_url: str) -> str:
    value = str(path_or_url)
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return _encode_file_as_data_url(path)


def _build_user_content(query_type: str, prompt: str, image_path: str | None) -> list[dict[str, Any]]:
    if query_type == "t2i":
        return [{"type": "text", "text": f"<|t2i|> {prompt}"}]

    if query_type == "t2s":
        return [{"type": "text", "text": f"<|t2s|> {prompt}"}]

    if query_type == "i2i":
        if not image_path:
            raise ValueError("--image-path is required for query type i2i")
        return [
            {"type": "text", "text": f"<|i2i|> {prompt}"},
            {"type": "image_url", "image_url": {"url": _to_image_url(image_path)}},
        ]

    raise ValueError(f"Unsupported query_type: {query_type}")


def _collect_text_from_content(content: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            texts.append(stripped)
        return texts

    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            text_value = content.get(key)
            if isinstance(text_value, str) and text_value.strip():
                texts.append(text_value.strip())
        return texts

    if isinstance(content, list):
        for item in content:
            texts.extend(_collect_text_from_content(item))
        return texts

    content_text = getattr(content, "text", None)
    if isinstance(content_text, str) and content_text.strip():
        texts.append(content_text.strip())
    content_value = getattr(content, "content", None)
    if isinstance(content_value, str) and content_value.strip():
        texts.append(content_value.strip())
    output_text = getattr(content, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        texts.append(output_text.strip())
    return texts


def _extract_text_outputs(chat_completion: Any) -> list[str]:
    texts: list[str] = []
    for choice in getattr(chat_completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        content = getattr(message, "content", None)
        texts.extend(_collect_text_from_content(content))
        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            texts.append(reasoning_content.strip())
        choice_text = getattr(choice, "text", None)
        if isinstance(choice_text, str) and choice_text.strip():
            texts.append(choice_text.strip())
    top_level_output_text = getattr(chat_completion, "output_text", None)
    if isinstance(top_level_output_text, str) and top_level_output_text.strip():
        texts.append(top_level_output_text.strip())
    return texts


def _extract_image_data_urls(chat_completion: Any) -> list[str]:
    urls: list[str] = []
    for choice in getattr(chat_completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_url":
                continue
            image_url = (item.get("image_url") or {}).get("url")
            if isinstance(image_url, str) and image_url.startswith("data:image"):
                urls.append(image_url)
    return urls


def _extract_audio_payloads(chat_completion: Any) -> list[bytes]:
    payloads: list[bytes] = []
    for choice in getattr(chat_completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        message_audio = getattr(message, "audio", None)
        if message_audio is None:
            continue
        data_b64 = getattr(message_audio, "data", None)
        if isinstance(data_b64, str) and data_b64:
            try:
                payloads.append(base64.b64decode(data_b64))
            except Exception:
                continue
    return payloads


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    header, data = data_url.split(",", 1)
    mime_type = "image/png"
    if ";" in header and ":" in header:
        mime_type = header.split(":", 1)[1].split(";", 1)[0]
    return base64.b64decode(data), mime_type


def _image_extension_from_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "image/gif":
        return ".gif"
    return ".png"


def _save_outputs(
    *,
    query_type: str,
    chat_completion: Any,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    text_outputs = _extract_text_outputs(chat_completion)
    image_data_urls = _extract_image_data_urls(chat_completion)
    audio_payloads = _extract_audio_payloads(chat_completion)

    if text_outputs:
        text_path = output_dir / f"{query_type}_{stamp}.txt"
        text_path.write_text("\n\n".join(text_outputs) + "\n", encoding="utf-8")
        print(f"[dynin-online] text saved: {text_path}")
        print(text_outputs[0])

    for idx, image_url in enumerate(image_data_urls):
        image_bytes, mime_type = _decode_data_url(image_url)
        ext = _image_extension_from_mime(mime_type)
        image_path = output_dir / f"{query_type}_{stamp}_{idx}{ext}"
        image_path.write_bytes(image_bytes)
        print(f"[dynin-online] image saved: {image_path}")

    for idx, audio_bytes in enumerate(audio_payloads):
        audio_path = output_dir / f"{query_type}_{stamp}_{idx}.wav"
        audio_path.write_bytes(audio_bytes)
        print(f"[dynin-online] audio saved: {audio_path}")

    if not text_outputs and not image_data_urls and not audio_payloads:
        print("[dynin-online] no output extracted from response")
        raw_path = output_dir / f"{query_type}_{stamp}_raw_response.json"
        try:
            if hasattr(chat_completion, "model_dump_json"):
                serialized = chat_completion.model_dump_json(indent=2)
            else:
                if hasattr(chat_completion, "model_dump"):
                    raw_payload: Any = chat_completion.model_dump(mode="json")
                else:
                    raw_payload = chat_completion
                try:
                    serialized = json.dumps(raw_payload, ensure_ascii=False, indent=2)
                except Exception:
                    serialized = json.dumps({"repr": repr(raw_payload)}, ensure_ascii=False, indent=2)
            raw_path.write_text(serialized + "\n", encoding="utf-8")
            print(f"[dynin-online] raw response saved: {raw_path}")
        except Exception:
            pass


def _build_offline_parity_sampling_params_list() -> list[dict[str, Any]]:
    return [dict(OFFLINE_PARITY_STAGE_SAMPLING) for _ in range(OFFLINE_PARITY_STAGE_COUNT)]


def run_request(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
    )
    prompt = args.prompt.strip() if args.prompt else DEFAULT_PROMPT_BY_QUERY[args.query_type]
    user_content = _build_user_content(
        query_type=args.query_type,
        prompt=prompt,
        image_path=args.image_path,
    )
    if args.modalities:
        modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    else:
        modalities = DEFAULT_MODALITIES_BY_QUERY[args.query_type]

    extra_body = {
        "sampling_params_list": _build_offline_parity_sampling_params_list(),
    }
    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": user_content}],
        modalities=modalities,
        extra_body=extra_body,
    )
    _save_outputs(
        query_type=args.query_type,
        chat_completion=chat_completion,
        output_dir=Path(args.output_dir).expanduser(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynin-Omni online chat completion client")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="t2i",
        choices=QUERY_CHOICES,
        help="Dynin query type",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name/path",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host/IP of the vLLM Omni API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="Port of the vLLM Omni API server",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="",
        help="Custom prompt text",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Image path/URL for i2i",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="",
        help="Comma-separated output modalities override (e.g., text,image,audio)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    run_request(args)


if __name__ == "__main__":
    main()
