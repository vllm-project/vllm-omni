"""Client for OmniVoice TTS via /v1/audio/speech endpoint.

Examples:
    # Basic TTS (auto voice)
    python speech_client.py --text "Hello, how are you?"

    # Specify language
    python speech_client.py --text "Bonjour, comment allez-vous?" --language French
"""

import argparse

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"


def run_tts(args) -> None:
    """Generate speech via /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": "default",
        "response_format": args.response_format,
    }

    if args.language:
        payload["language"] = args.language

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    if args.language:
        print(f"Language: {args.language}")
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

    output_path = args.output or "omnivoice_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OmniVoice TTS client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default="k2-fsa/OmniVoice", help="Model name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default=None, help="Language hint (e.g., English, Chinese, French)")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
