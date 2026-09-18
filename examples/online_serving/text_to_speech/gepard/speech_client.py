# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Client for Gepard-1.0 TTS via /v1/audio/speech.

Gepard is zero-shot: omit ``voice`` or pass ``"default"``. Output is 22.05 kHz
mono. ``seed`` is optional; the packaged deploy YAML currently pins ``seed: 42``
until that default is removed, so requests without an explicit seed are still
deterministic.

Examples:
    python speech_client.py --text "Hello, this is Gepard speaking."
    python speech_client.py --text "Hello, this is Gepard speaking." --seed 7
    python speech_client.py --text "Hello, this is Gepard speaking." --stream --output output.pcm
"""

from __future__ import annotations

import argparse

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "nineninesix/gepard-1.0"


def run_tts(args) -> None:
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.stream:
        payload["stream"] = True
        payload["stream_format"] = "audio"
        if args.response_format not in ("pcm", "wav"):
            print(f"Note: streaming requires pcm/wav; overriding --response-format {args.response_format} to 'pcm'")
            payload["response_format"] = "pcm"

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("Generating audio...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    if args.stream:
        output_path = args.output or ("output.wav" if payload["response_format"] == "wav" else "output.pcm")
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    print(f"Error: {resp.status_code}")
                    print(resp.read().decode())
                    return
                total_bytes = 0
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                        total_bytes += len(chunk)
                print(f"Streamed {total_bytes} bytes to: {output_path}")
        return

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    output_path = args.output or "output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gepard-1.0 TTS client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="default", help="Voice name (only 'default' is supported)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Frame budget (1 token = 1 frame ≈ 46.4 ms at 21.5 fps)",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming (PCM output)")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm"],
        help="Audio format (default: wav). Streaming is pcm/wav only. Opus is not supported at 22.05 kHz.",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
