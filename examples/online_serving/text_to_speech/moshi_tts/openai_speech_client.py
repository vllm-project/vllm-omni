"""OpenAI-compatible client for Moshi TTS via /v1/audio/speech.

Demonstrates how to synthesise speech from a running Moshi TTS vLLM Omni
server using the standard OpenAI speech API.

Start the server first:
    vllm serve --omni /path/to/tts-1.6b-en_fr \\
        --deploy-config vllm_omni/deploy/moshi_tts.yaml

Then run this script:
    python openai_speech_client.py --text "Hello, this is Moshi."

With a voice prefix (local file, URL, or base64 data URI):
    python openai_speech_client.py --text "Hello!" --voice /path/to/voice.wav

Streaming (audio plays as chunks arrive):
    python openai_speech_client.py --text "Hello!" --stream
"""

import argparse
import base64
import sys
import time

import httpx

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"


def _resolve_voice(voice: str) -> str:
    """Return a value suitable for the ``ref_audio`` API field.

    * Local file path → read and encode as ``data:audio/wav;base64,...``.
    * http(s) URL → passed through; the server fetches it via MediaConnector.
    * Base64 data URI → passed through as-is.
    """
    if voice.startswith("data:"):
        return voice
    if voice.startswith(("http://", "https://")):
        return voice
    # Treat as a local file path.
    with open(voice, "rb") as fh:
        return f"data:audio/wav;base64,{base64.b64encode(fh.read()).decode()}"


def run(args) -> None:
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }
    if args.voice:
        payload["ref_audio"] = _resolve_voice(args.voice)

    output_path = args.output or f"moshi_tts_output.{args.response_format}"
    print(f"Model  : {args.model}")
    print(f"Text   : {args.text!r}")
    print(f"Output : {output_path}")

    t0 = time.perf_counter()

    if args.stream:
        print("Streaming audio chunks...")
        stem, ext = output_path.rsplit(".", 1) if "." in output_path else (output_path, args.response_format)
        chunk_idx = 0
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}: {resp.read().decode()}", file=sys.stderr)
                    sys.exit(1)
                for chunk in resp.iter_bytes(chunk_size=4096):
                    if not chunk:
                        continue
                    chunk_path = f"{stem}_chunk_{chunk_idx:03d}.{ext}"
                    with open(chunk_path, "wb") as fh:
                        fh.write(chunk)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if chunk_idx == 0:
                        print(f"  First chunk received — TTFA {elapsed_ms:.0f} ms")
                    print(f"  Chunk {chunk_idx:03d}: {len(chunk)} bytes → {chunk_path}")
                    chunk_idx += 1
        print(f"Done — {chunk_idx} chunks in {(time.perf_counter() - t0) * 1000:.0f} ms")
    else:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(api_url, json=payload, headers=headers)

        if resp.status_code != 200:
            content = resp.content
            try:
                print(f"Error {resp.status_code}: {content.decode()}", file=sys.stderr)
            except UnicodeDecodeError:
                print(f"Error {resp.status_code}: <binary>", file=sys.stderr)
            sys.exit(1)

        with open(output_path, "wb") as fh:
            fh.write(resp.content)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"Done — {len(resp.content)} bytes in {elapsed_ms:.0f} ms")

    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Moshi TTS client — /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"Server base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key (default: EMPTY)")
    parser.add_argument(
        "--model",
        "-m",
        default="moshi_tts",
        help="Model name sent in the request (default: moshi_tts)",
    )
    parser.add_argument(
        "--text",
        "-t",
        required=True,
        help="Text to synthesise",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default=None,
        help=(
            "Voice prefix for audio-conditioned models (e.g. tts-0.75b-en-public). "
            "Accepts a local file path, http(s) URL, or base64 data URI."
        ),
    )
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "mp3", "flac"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream audio as it is generated (requires server-side streaming support)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
