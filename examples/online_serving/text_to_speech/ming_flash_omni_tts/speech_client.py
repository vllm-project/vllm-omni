"""Client for Ming standalone TTS via /v1/audio/speech endpoint."""

import argparse
import json
import sys

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "Jonathan1909/Ming-flash-omni-2.0"


def run_tts(args) -> None:
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }

    instructions = args.instructions
    if args.instruction_json:
        if instructions:
            sys.exit("--instructions and --instruction-json are mutually exclusive")

        try:
            parsed = json.loads(args.instruction_json)
        except json.JSONDecodeError as exc:
            sys.exit(f"--instruction-json must be valid JSON: {exc}")
        if not isinstance(parsed, dict):
            sys.exit("--instruction-json must decode to a JSON object")
        # Re-encode with ensure_ascii=False so UTF-8 Chinese keys/values
        # arrive at the server intact rather than as \\uXXXX escapes.
        instructions = json.dumps(parsed, ensure_ascii=False)
    if instructions:
        payload["instructions"] = instructions

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print("Generating audio...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    output_path = args.output or "ming_tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ming standalone TTS speech client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name or local path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Free-form style description (mapped to caption 风格 on the server).",
    )
    parser.add_argument(
        "--instruction-json",
        default=None,
        help=(
            "Structured caption JSON forwarded as `instructions`. Accepts Ming "
            "caption keys: 方言, 风格, 语速, 基频, 音量, 情感, IP, 说话人, BGM. "
        ),
    )
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
