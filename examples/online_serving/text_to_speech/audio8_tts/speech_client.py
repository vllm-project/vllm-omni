# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Client for Audio8 TTS Preview via the /v1/audio/speech endpoint.

Examples:
    # Basic TTS
    python speech_client.py --text "Welcome to Audio8 TTS."

    # Zero-shot voice cloning (the transcript must match the reference audio)
    python speech_client.py --text "Welcome to Audio8 TTS." \
        --ref-audio ref.wav --ref-text "The exact transcript of the reference recording."

    # Streaming PCM output
    python speech_client.py --text "Welcome to Audio8 TTS." --stream --output output.pcm
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8092"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "Audio8/Audio8-TTS-Preview-0.6b"
#: The codec runs at 44.1 kHz, so raw PCM must be played back at that rate.
PCM_SAMPLE_RATE = 44100


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file as a base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}.get(
        ext, "audio/wav"
    )
    with open(audio_path, "rb") as handle:
        audio_b64 = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def build_payload(args) -> dict:
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.stream:
        payload["stream"] = True
        payload["stream_format"] = "audio"
        payload["response_format"] = "pcm"
    return payload


def run_tts(args) -> None:
    payload = build_payload(args)
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    if args.ref_audio:
        print(f"Voice cloning: ref_audio={args.ref_audio!r} ref_text={args.ref_text!r}")

    if args.stream:
        output_path = args.output or "output.pcm"
        with (
            httpx.Client(timeout=300.0) as client,
            client.stream("POST", api_url, json=payload, headers=headers) as resp,
        ):
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.read().decode()}")
                return
            total_bytes = 0
            with open(output_path, "wb") as handle:
                for chunk in resp.iter_bytes():
                    handle.write(chunk)
                    total_bytes += len(chunk)
        seconds = total_bytes / 2 / PCM_SAMPLE_RATE
        print(f"Streamed {total_bytes} bytes (~{seconds:.2f}s of s16le @ {PCM_SAMPLE_RATE} Hz) to {output_path}")
        print(f"Play with: ffplay -f s16le -ar {PCM_SAMPLE_RATE} -ac 1 {output_path}")
        return

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
    if response.headers.get("content-type", "").startswith("application/json"):
        print(f"Error: {response.text}")
        return

    output_path = args.output or f"output.{args.response_format}"
    with open(output_path, "wb") as handle:
        handle.write(response.content)
    print(f"Audio saved to: {output_path} ({len(response.content)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio8 TTS Preview client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning (path or URL)")
    parser.add_argument("--ref-text", default=None, help="Transcript of the reference audio")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Cap the number of generated codec frames")
    parser.add_argument("--stream", action="store_true", help="Stream PCM instead of returning a whole file")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    run_tts(parser.parse_args())


if __name__ == "__main__":
    main()
