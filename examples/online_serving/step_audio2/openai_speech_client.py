"""OpenAI-compatible client for Step-Audio2 TTS via /v1/audio/speech endpoint.

Examples:
    # Basic TTS
    python openai_speech_client.py --text "你好世界"

    # With custom system prompt
    python openai_speech_client.py --text "Hello, how are you?" \
        --instructions "You are a friendly assistant."

    # Save to specific file
    python openai_speech_client.py --text "你好世界" -o output.wav
"""

import argparse

import httpx

DEFAULT_API_BASE = "http://localhost:8092"
DEFAULT_API_KEY = "EMPTY"


def run_tts_generation(args) -> None:
    """Run TTS generation via /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }

    if args.instructions:
        payload["instructions"] = args.instructions

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
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

    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass

    output_path = args.output or "tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step-Audio2 TTS client via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
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
        default="stepfun-ai/Step-Audio-2-mini",
        help="Model name/path",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="default",
        help="Voice name (default: default)",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="System prompt for the Thinker stage",
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "pcm"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: tts_output.wav)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tts_generation(args)
