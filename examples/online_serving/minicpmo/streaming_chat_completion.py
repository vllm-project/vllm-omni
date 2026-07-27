# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream MiniCPM-o 4.5 text and audio from /v1/chat/completions."""

from __future__ import annotations

import argparse
import base64
import io
import wave
from pathlib import Path

from openai import OpenAI


def append_wav_chunk(encoded: str, pcm_parts: list[bytes]) -> tuple[int, int, int]:
    """Decode one base64 WAV delta and append its PCM frames."""
    with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        pcm_parts.append(wav.readframes(wav.getnframes()))
    return channels, sample_width, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8099/v1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--prompt", default="Say hello, then introduce vLLM in one sentence.")
    parser.add_argument("--output", type=Path, default=Path("minicpmo_stream.wav"))
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()

    modalities = ["text"] if args.text_only else ["text", "audio"]
    stream = OpenAI(base_url=args.base_url, api_key="EMPTY").chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        modalities=modalities,
        stream=True,
        extra_body={"chat_template_kwargs": {"use_tts_template": not args.text_only}},
    )

    pcm_parts: list[bytes] = []
    audio_format: tuple[int, int, int] | None = None
    for chunk in stream:
        for choice in chunk.choices:
            content = getattr(choice.delta, "content", None)
            if not content:
                continue
            if getattr(chunk, "modality", None) == "audio":
                current_format = append_wav_chunk(content, pcm_parts)
                if audio_format is not None and current_format != audio_format:
                    raise ValueError(f"Audio format changed: {audio_format} -> {current_format}")
                audio_format = current_format
            elif getattr(chunk, "modality", None) == "text":
                print(content, end="", flush=True)

    print()
    if pcm_parts and audio_format is not None:
        channels, sample_width, sample_rate = audio_format
        with wave.open(str(args.output), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(b"".join(pcm_parts))
        print(f"Audio saved to {args.output} ({sample_rate} Hz)")


if __name__ == "__main__":
    main()
