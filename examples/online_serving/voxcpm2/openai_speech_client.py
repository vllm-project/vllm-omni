"""OpenAI-compatible client for VoxCPM2 TTS via /v1/audio/speech endpoint.

Examples:
    # Zero-shot synthesis
    python openai_speech_client.py --text "Hello, this is VoxCPM2."

    # Voice cloning with a local reference audio file
    python openai_speech_client.py --text "Hello world" \
        --ref-audio /path/to/reference.wav

    # Voice cloning with a URL
    python openai_speech_client.py --text "Hello world" \
        --ref-audio "https://example.com/reference.wav"

Server setup:
    python -m vllm_omni.entrypoints.openai.api_server \
        --model openbmb/VoxCPM2 \
        --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm2.yaml \
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "sk-empty"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to a base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(ext, "audio/wav")

    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM2 OpenAI speech client")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (local path, URL, or data: URI)",
    )
    parser.add_argument("--model", type=str, default="voxcpm2")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--response-format", type=str, default="wav")
    args = parser.parse_args()

    # VoxCPM2 has no predefined voices. The "voice" field is required by
    # the OpenAI API schema but ignored by VoxCPM2 — use any placeholder.
    # For voice cloning, pass --ref-audio instead.
    payload: dict = {
        "model": args.model,
        "input": args.text,
        "voice": "default",
        "response_format": args.response_format,
    }

    if args.ref_audio:
        ref = args.ref_audio
        if ref.startswith(("http://", "https://", "data:")):
            payload["ref_audio"] = ref
        else:
            payload["ref_audio"] = encode_audio_to_base64(ref)

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
