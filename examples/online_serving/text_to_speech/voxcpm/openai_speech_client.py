"""OpenAI-compatible client for VoxCPM via /v1/audio/speech.

Examples:
    # Basic text-to-speech
    python openai_speech_client.py --text "Hello from VoxCPM"

    # Voice cloning
    python openai_speech_client.py \
        --text "This sentence uses the cloned voice." \
        --ref-audio /path/to/reference.wav \
        --ref-text "The exact transcript spoken in the reference audio."

    # Streaming PCM output
    python openai_speech_client.py \
        --text "This is a streaming VoxCPM request." \
        --stream \
        --output output.pcm
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "OpenBMB/VoxCPM1.5"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    mime_type = mime_map.get(ext, "audio/wav")

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def build_payload(args) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": args.model,
        "input": args.text,
        "response_format": "pcm" if args.stream else args.response_format,
    }

    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://", "data:")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.stream:
        payload["stream"] = True

    return payload


def run_tts(args) -> None:
    payload = build_payload(args)
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    if args.ref_audio:
        print("Mode: voice cloning")
        print(f"Reference audio: {args.ref_audio}")
    else:
        print("Mode: text-to-speech")

    if args.stream:
        output_path = args.output or "voxcpm_output.pcm"
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(response.read().decode("utf-8", errors="ignore"))
                    return

                total_bytes = 0
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
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

    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass

    output_path = args.output or "voxcpm_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VoxCPM OpenAI-compatible speech client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", default=None, help="Reference audio path, URL, or data URL")
    parser.add_argument(
        "--ref-text",
        default=None,
        help="The exact transcript spoken in the reference audio",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming PCM output")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format for non-streaming mode (default: wav)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
