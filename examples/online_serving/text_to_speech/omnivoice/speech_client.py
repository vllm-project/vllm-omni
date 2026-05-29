"""Client for OmniVoice TTS via /v1/audio/speech endpoint.

Examples:
    # Basic TTS (auto voice)
    python speech_client.py --text "Hello, how are you?"

    # Specify language
    python speech_client.py --text "Bonjour, comment allez-vous?" --language French

    # Use a specific uploaded/supported voice
    python speech_client.py --text "Hello" --voice my_uploaded_voice
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"


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


def run_tts(args) -> None:
    """Generate speech via /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }
    if args.seed is not None:
        payload["extra_params"] = {}
        payload["extra_params"]["seed"] = args.seed

    if args.voice:
        payload["voice"] = args.voice
    if args.language:
        payload["language"] = args.language

    if args.ref_audio:
        ref = args.ref_audio
        if ref.startswith(("http://", "https://", "data:")):
            payload["ref_audio"] = ref
        else:
            payload["ref_audio"] = encode_audio_to_base64(ref)

    if args.ref_text:
        payload["ref_text"] = args.ref_text

    if args.instructions:
        payload["instructions"] = args.instructions

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    if args.seed:
        print(f"Seed: {args.seed}")

    if args.voice:
        print(f"Voice: {args.voice}")

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
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice name (omit for auto voice; must match a supported or uploaded speaker if set)",
    )
    parser.add_argument("--language", default=None, help="Language hint (e.g., English, Chinese, French)")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (local path, URL, or data: URI)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference text for voice cloning",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Voice style/emotion instructions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation, default: None for stochastic output)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
