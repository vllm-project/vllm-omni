"""OpenAI-compatible client for Step-Audio-Editx via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using Step-Audio-Editx models.
"""

import argparse
import base64
import os

import httpx

# Default server configuration
DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type from extension
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
        mime_type = "audio/wav"  # Default

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""

    # Build request payload
    payload = {
        "model": args.model,
        "input": args.text or "",
        "response_format": args.response_format,
    }
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        payload["seed"] = args.seed

    # Voice clone parameters (Base task)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        elif args.ref_audio.startswith("data:"):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    payload["extra_params"] = {}
    if args.edit_type is not None:
        payload["extra_params"]["edit_type"] = args.edit_type
    else:
        payload["extra_params"]["edit_type"] = "clone"

    if args.edit_info is not None:
        payload["extra_params"]["edit_info"] = args.edit_info

    print(f"Model: {args.model}")
    print(f"Task type: {args.edit_type}")
    print(f"Text: {args.text}")
    print("ref_audio prefix:", payload.get("ref_audio", "")[:80])
    print("ref_text:", payload.get("ref_text"))
    print("payload extra_params:", payload.get("extra_params"))
    print("Generating audio...")

    # Make the API call
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

    # Check for JSON error response (only if content is valid UTF-8 text)
    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass  # Binary audio data, not an error

    # Save audio response
    output_path = args.output or "tts_output.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for Step-Audio-Editx via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        default="stepfun-ai/Step-Audio-Editx",
        help="Model name/path",
    )
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        default="stepfun-ai/Step-Audio-Tokenizer",
        help="Model name/path",
    )

    # Task configuration
    parser.add_argument(
        "--edit-type",
        choices=("clone", "emotion", "paralinguistic", "style", "denoise", "vad", "speed"),
        default="clone",
        help="Task type: clone, emotion, paralinguistic, style, denoise, vad, speed",
    )
    parser.add_argument("--edit-info", type=str, default=None, help="Additional information for the edit.")
    # Input text
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize",
    )
    # Base (voice clone) parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file path, URL, or base64 for voice cloning (Base task)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference audio transcript for voice cloning (Base task)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
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
