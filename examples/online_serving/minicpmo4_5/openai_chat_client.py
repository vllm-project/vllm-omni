#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for MiniCPM-o 4.5 online serving."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

MODEL = "openbmb/MiniCPM-o-4_5"
QueryType = Literal["text", "use_image", "use_video"]

DEFAULT_PROMPTS: dict[QueryType, str] = {
    "text": (
        "Please read this single long sentence aloud exactly once without shortening it: "
        "vLLM Omni is running a benchmark for MiniCPM speech generation, and this sentence intentionally "
        "includes enough detail about streaming text to audio generation, multimodal reasoning, "
        "stage connectors, careful benchmarking, and stable speech synthesis behavior to last well "
        "over ten seconds when spoken at a natural pace."
    ),
    "use_image": (
        "Describe the image in one single detailed spoken sentence of at least sixty words, "
        "mentioning every visible shape, its color, its approximate size, its position "
        "relative to the other shapes, the plain background, and the overall layout, and keep "
        "the answer natural but long enough to last more than ten seconds."
    ),
    "use_video": (
        "Describe the video in one single detailed spoken sentence of at least sixty words, "
        "covering the moving objects, their colors, their approximate sizes, the direction and "
        "pattern of their motion over time, the dark background, and the overall scene, and "
        "keep the answer natural but long enough to last more than ten seconds."
    ),
}


def _media_data_url(path: str, default_mime: str) -> str:
    if path.startswith(("http://", "https://", "data:")):
        return path

    media_path = Path(path).expanduser()
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    suffix = media_path.suffix.lower()
    mime = default_mime
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".mp4":
        mime = "video/mp4"
    elif suffix == ".webm":
        mime = "video/webm"
    elif suffix == ".mov":
        mime = "video/quicktime"

    encoded = base64.b64encode(media_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _system_message() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "When audio output is requested, reply with speech only "
                    "and follow any requested length constraints."
                ),
            }
        ],
    }


def _user_message(args: argparse.Namespace) -> dict[str, Any]:
    query_type: QueryType = args.query_type
    prompt = args.prompt or DEFAULT_PROMPTS[query_type]
    content: list[dict[str, Any]] = []

    if query_type == "use_image":
        image_url = _media_data_url(
            args.image_path
            or "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg",
            "image/jpeg",
        )
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    elif query_type == "use_video":
        video_url = _media_data_url(
            args.video_path
            or "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            "video/mp4",
        )
        content.append({"type": "video_url", "video_url": {"url": video_url}})

    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _parse_modalities(value: str) -> list[str]:
    modalities = [item.strip() for item in value.split(",") if item.strip()]
    if not modalities:
        raise ValueError("--modalities must include at least one modality")
    valid = {"text", "audio"}
    invalid = sorted(set(modalities) - valid)
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}. Valid: {sorted(valid)}")
    return modalities


def _load_ref_audio(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None

    import numpy as np
    import soundfile as sf

    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio file not found: {audio_path}")

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    wav_np = np.asarray(wav, dtype=np.float32)
    if wav_np.ndim == 0 or wav_np.size == 0:
        raise ValueError(f"Reference audio is empty: {audio_path}")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)
    return {
        "wav": wav_np.reshape(-1).tolist(),
        "sr": int(sr),
    }


def _output_path_for_index(path: str, index: int) -> Path:
    output = Path(path)
    if index == 0:
        return output
    return output.with_name(f"{output.stem}_{index}{output.suffix}")


def _save_non_stream_response(response: Any, output_wav: str, output_text: str) -> None:
    text_parts: list[str] = []
    audio_count = 0

    for choice in response.choices:
        message = choice.message
        content = getattr(message, "content", None)
        if content:
            if isinstance(content, str):
                text_parts.append(content)
            else:
                text_parts.append(json.dumps(content, ensure_ascii=False))

        audio = getattr(message, "audio", None)
        audio_data = getattr(audio, "data", None) if audio is not None else None
        if audio_data:
            audio_path = _output_path_for_index(output_wav, audio_count)
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(base64.b64decode(audio_data))
            print(f"Audio saved to {audio_path}")
            audio_count += 1

    if text_parts:
        text_path = Path(output_text)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("\n".join(text_parts), encoding="utf-8")
        print(f"Text saved to {text_path}")


def _save_stream_response(response: Any, output_wav: str, output_text: str) -> None:
    text_parts: list[str] = []
    audio_chunks: list[bytes] = []

    for chunk in response:
        for choice in chunk.choices:
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if not content:
                continue
            if getattr(chunk, "modality", None) == "audio":
                audio_chunks.append(base64.b64decode(content))
            else:
                text_parts.append(str(content))

    if audio_chunks:
        wav_path = Path(output_wav)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"".join(audio_chunks))
        print(f"Audio saved to {wav_path}")

    if text_parts:
        text_path = Path(output_text)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("".join(text_parts), encoding="utf-8")
        print(f"Text saved to {text_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://localhost:8091")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--query-type", choices=["text", "use_image", "use_video"], default="text")
    parser.add_argument("--prompt")
    parser.add_argument("--image-path")
    parser.add_argument("--video-path")
    parser.add_argument("--ref-audio-path", help="Optional reference audio for MiniCPM voice conditioning.")
    parser.add_argument("--modalities", default="audio")
    parser.add_argument("--output-wav", default="minicpmo45_online_output.wav")
    parser.add_argument("--output-text", default="minicpmo45_online_output.txt")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.server.rstrip('/')}/v1", api_key="EMPTY")
    extra_body = {
        "chat_template_kwargs": {
            "use_tts_template": True,
            "enable_thinking": False,
        }
    }
    ref_audio = _load_ref_audio(args.ref_audio_path)
    if ref_audio is not None:
        extra_body["additional_information"] = {"ref_audio": ref_audio}

    response = client.chat.completions.create(
        model=args.model,
        messages=[_system_message(), _user_message(args)],
        modalities=_parse_modalities(args.modalities),
        stream=args.stream,
        extra_body=extra_body,
    )

    if args.stream:
        _save_stream_response(response, args.output_wav, args.output_text)
    else:
        _save_non_stream_response(response, args.output_wav, args.output_text)


if __name__ == "__main__":
    main()
