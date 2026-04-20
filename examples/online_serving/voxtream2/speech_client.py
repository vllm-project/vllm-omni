"""Client for Voxtream2 TTS via /v1/audio/speech.

Examples:
    python speech_client.py \
        --text "This is a Voxtream2 online serving example." \
        --ref-audio voxtream/assets/audio/english_female.wav
"""

import argparse
import base64
from pathlib import Path

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "herimor/voxtream2"


def _audio_to_data_url(audio_path: str) -> str:
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    suffix = path.suffix.lower()
    mime_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(suffix, "audio/wav")

    audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def _resolve_ref_audio(ref_audio: str, use_file_uri: bool) -> str:
    if ref_audio.startswith(("http://", "https://", "data:", "file://")):
        return ref_audio
    path = Path(ref_audio).expanduser().resolve()
    if use_file_uri:
        return path.as_uri()
    return _audio_to_data_url(str(path))


def run_tts(args) -> None:
    payload = {
        "model": args.model,
        "input": args.text,
        "ref_audio": _resolve_ref_audio(args.ref_audio, args.file_uri),
        "response_format": args.response_format,
    }

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Reference audio: {args.ref_audio}")
    print("Generating audio...")

    with httpx.Client(timeout=args.timeout) as client:
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

    output_path = args.output or "voxtream2.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Voxtream2 TTS client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", required=True, help="Reference audio path, URL, data URL, or file URI")
    parser.add_argument(
        "--file-uri",
        action="store_true",
        help="Send local --ref-audio as file:// instead of base64 data URL",
    )
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
