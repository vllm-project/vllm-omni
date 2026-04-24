"""Client for Fish Speech S2 Pro via /v1/audio/speech endpoint.

Examples:
    # Basic TTS
    python speech_client.py --text "Hello, how are you?"

    # Voice cloning
    python speech_client.py --text "Hello, how are you?" \
        --ref-audio ref.wav --ref-text "This is the reference transcript."

    # Streaming PCM output
    python speech_client.py --text "Hello world" --stream --output output.pcm
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts(args) -> None:
    """Generate speech via /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": "default",
        "response_format": args.response_format,
    }

    # Voice cloning parameters.
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    if args.stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    if args.ref_audio:
        print(f"Voice cloning: ref_audio={args.ref_audio}, ref_text={args.ref_text}")
    print("Generating audio...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    if args.stream:
        output_path = args.output or "output.pcm"
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

        output_path = args.output or "output.wav"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fish Speech S2 Pro TTS client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default="fishaudio/s2-pro", help="Model name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning (path or URL)")
    parser.add_argument("--ref-text", default=None, help="Transcript of reference audio")
    parser.add_argument("--stream", action="store_true", help="Enable streaming (PCM output)")
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
