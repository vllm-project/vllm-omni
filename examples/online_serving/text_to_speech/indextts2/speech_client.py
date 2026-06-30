# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for IndexTTS2 TTS via /v1/audio/speech endpoint.

Examples:
    # With reference audio for voice cloning
    python speech_client.py --text "你好，世界！" \
        --ref-audio /path/to/reference.wav

    # With emotion audio
    python speech_client.py --text "今天心情很好！" \
        --ref-audio /path/to/ref.wav \
        --emo-audio /path/to/happy.wav

Server setup:
    vllm serve IndexTeam/IndexTTS-2 --omni --host 0.0.0.0 --port 8092
"""

from __future__ import annotations

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8092"
DEFAULT_API_KEY = "sk-empty"


def encode_audio_to_base64(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser(description="IndexTTS2 OpenAI speech client")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--emo-audio", type=str, default=None, help="Emotion reference audio")
    parser.add_argument("--emo-text", type=str, default=None, help="Emotion description text")
    parser.add_argument(
        "--emo-vector",
        type=float,
        nargs=8,
        default=None,
        help="8-dim emotion vector: happy angry sad afraid disgusted melancholic surprised calm",
    )
    parser.add_argument("--emo-alpha", type=float, default=None, help="Emotion weight in [0, 1]")
    parser.add_argument("--use-emo-text", action="store_true", help="Infer emotion vector from emo-text or text")
    parser.add_argument("--use-random", action="store_true", help="Use random emotion prototypes")
    parser.add_argument("--model", type=str, default="IndexTeam/IndexTTS-2")
    parser.add_argument("--voice", type=str, default=None, help="Uploaded voice name to use instead of --ref-audio")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--response-format", type=str, default="wav")
    args = parser.parse_args()

    if not args.ref_audio and not args.voice:
        parser.error("IndexTTS2 requires --ref-audio or --voice for voice cloning")

    payload: dict = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }
    if args.voice:
        payload["voice"] = args.voice

    if args.ref_audio:
        ref = args.ref_audio
        if ref.startswith(("http://", "https://", "data:")):
            payload["ref_audio"] = ref
        else:
            payload["ref_audio"] = encode_audio_to_base64(ref)

    extra_params = {}
    if args.emo_audio:
        emo = args.emo_audio
        if emo.startswith(("http://", "https://", "data:")):
            extra_params["emo_audio"] = emo
        else:
            extra_params["emo_audio"] = encode_audio_to_base64(emo)
    if args.emo_text:
        extra_params["emo_text"] = args.emo_text
    if args.emo_vector is not None:
        extra_params["emo_vector"] = args.emo_vector
    if args.emo_alpha is not None:
        extra_params["emo_alpha"] = args.emo_alpha
    if args.use_emo_text:
        extra_params["use_emo_text"] = True
    if args.use_random:
        extra_params["use_random"] = True
    if extra_params:
        payload["extra_params"] = extra_params

    url = f"{args.api_base}/v1/audio/speech"
    print(f"POST {url}")
    print(f"  text: {args.text}")
    if args.ref_audio:
        print(f"  ref_audio: {args.ref_audio[:80]}...")

    with httpx.Client(timeout=300) as client:
        resp = client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {args.api_key}"},
        )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text[:500]}")
        return

    with open(args.output, "wb") as f:
        f.write(resp.content)
    print(f"Saved: {args.output} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    main()
