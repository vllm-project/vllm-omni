"""OpenAI-compatible client for Raon-Speech via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using the Raon-Speech model.

Examples:
    # Basic TTS
    python openai_speech_client.py --text "Hello, how are you?"

    # Voice cloning with a reference audio file
    python openai_speech_client.py \\
        --text "Hello, this is a cloned voice." \\
        --ref-audio /path/to/reference.wav

    # Voice cloning with a reference URL
    python openai_speech_client.py \\
        --text "Hello, this is a cloned voice." \\
        --ref-audio https://example.com/reference.wav

    # Save to a specific output file
    python openai_speech_client.py \\
        --text "Hello world" \\
        --output my_output.wav \\
        --response-format wav
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "KRAFTON/Raon-Speech-9B"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }

    # Voice cloning parameters
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens

    task = "TTS_ICL" if args.ref_audio else "TTS"
    print(f"Model: {args.model}")
    print(f"Task: {task}")
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

    # Check for JSON error response
    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass  # Binary audio data, not an error

    output_path = args.output or f"tts_output.{args.response_format}"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for Raon-Speech via /v1/audio/speech",
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
        default=DEFAULT_MODEL,
        help=f"Model name/path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )

    # Voice cloning parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file path or URL for voice cloning (TTS_ICL)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference audio (optional, improves voice cloning quality)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: tts_output.<format>)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tts_generation(args)
