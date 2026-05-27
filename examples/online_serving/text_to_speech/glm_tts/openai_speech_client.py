# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for GLM-TTS via /v1/audio/speech endpoint.

GLM-TTS is a two-stage TTS system (AR + DiT) that generates audio from text
conditioned on reference speech. Each request requires ref_audio + ref_text.

Usage:
    # Voice cloning
    python openai_speech_client.py --text "你好" --ref-audio file:///path/to/ref.wav --ref-text "参考文本"

    # Streaming response, for async_chunk server mode
    python openai_speech_client.py --text "你好" --stream --ref-audio file:///path/to/ref.wav --ref-text "参考文本"

    # Specify output format
    python openai_speech_client.py --text "你好" --ref-audio file:///path/to/ref.wav \
        --ref-text "参考文本" --response-format mp3 -o output.mp3
"""

import argparse

import httpx

# Default server configuration
DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""
    if not args.ref_audio or not args.ref_text:
        raise ValueError("GLM-TTS requires --ref-audio and --ref-text for voice cloning.")

    payload = {
        "model": args.model,
        "voice": "default",
        "input": args.text,
        "response_format": args.response_format,
        "stream": bool(args.stream),
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
    }
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Voice cloning: ref_audio={args.ref_audio}, ref_text={args.ref_text}")
    print(f"Stream: {args.stream}")
    print("Generating audio...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    if args.stream:
        output_path = args.output or "tts_output.pcm"
        with httpx.Client(timeout=300.0) as client, open(output_path, "wb") as f:
            with client.stream("POST", api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    response.read()
                    print(response.text)
                    return
                for chunk in response.iter_bytes():
                    f.write(chunk)
        print(f"Streaming audio saved to: {output_path}")
    else:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(api_url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        try:
            text = response.content.decode("utf-8")
            if text.startswith('{"error"'):
                print(f"Error: {text}")
                return
        except UnicodeDecodeError:
            pass
        output_path = args.output or f"tts_output.{args.response_format}"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for GLM-TTS via /v1/audio/speech",
    )

    # Server configuration
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="glm-tts",
        help="Model name/path",
    )

    # Input text
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate (default: model default)",
    )

    # Output
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Request a streaming audio response (use with async_chunk server mode).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: tts_output.<format>)",
    )

    # Voice cloning parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio URL, file:// URI, or base64 data URL for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference audio (required with --ref-audio)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tts_generation(args)
