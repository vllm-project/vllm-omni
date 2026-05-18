"""OpenAI-compatible client for CosyVoice3 via /v1/audio/speech.

Examples:
    python speech_client.py \
        --text "CosyVoice is undergoing a comprehensive upgrade." \
        --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \
        --ref-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
        --output cosyvoice3_output.wav

    python speech_client.py \
        --text "CosyVoice streaming over HTTP." \
        --ref-audio /path/to/reference.wav \
        --ref-text "Reference transcript." \
        --stream \
        --output cosyvoice3_output.pcm
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file as a data URL."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    mime_type = MIME_BY_SUFFIX.get(path.suffix.lower(), "audio/wav")
    audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def normalize_ref_audio(ref_audio: str) -> str:
    """Return a server-consumable reference audio URI."""
    if ref_audio.startswith(("http://", "https://", "data:", "file:")):
        return ref_audio
    return encode_audio_to_base64(ref_audio)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    response_format = "pcm" if args.stream else args.response_format
    payload: dict[str, Any] = {
        "model": args.model,
        "input": args.text,
        "ref_audio": normalize_ref_audio(args.ref_audio),
        "ref_text": args.ref_text,
        "response_format": response_format,
    }

    if args.stream:
        payload["stream"] = True
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.extra_params is not None:
        payload["extra_params"] = json.loads(args.extra_params)

    return payload


def run_tts(args: argparse.Namespace) -> None:
    try:
        import httpx
    except ImportError:
        raise SystemExit("Please install httpx: pip install httpx") from None

    payload = build_payload(args)
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Reference audio: {args.ref_audio}")
    print("Generating audio...")

    output_path = args.output or ("cosyvoice3_output.pcm" if args.stream else "cosyvoice3_output.wav")
    if args.stream:
        with httpx.Client(timeout=300.0, trust_env=False) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    raise SystemExit(f"Error {response.status_code}: {response.read().decode(errors='ignore')}")
                total_bytes = 0
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_bytes += len(chunk)
        print(f"Streamed {total_bytes} bytes to: {output_path}")
        return

    with httpx.Client(timeout=300.0, trust_env=False) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        raise SystemExit(f"Error {response.status_code}: {response.text}")

    Path(output_path).write_bytes(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CosyVoice3 OpenAI-compatible speech client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", required=True, help="Reference audio path, URL, data URL, or file URI")
    parser.add_argument("--ref-text", required=True, help="Transcript of the reference audio")
    parser.add_argument("--stream", action="store_true", help="Enable streaming PCM output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional maximum generated tokens")
    parser.add_argument(
        "--extra-params",
        default=None,
        help="Optional JSON object passed to the speech endpoint extra_params field",
    )
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format for non-streaming mode (default: wav)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
