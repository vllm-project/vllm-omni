"""Batch speech client for Qwen3-TTS via /v1/audio/speech/batch endpoint.

This script demonstrates how to synthesize multiple texts in a single request.
A particularly useful scenario is voice cloning: set ref_audio once at the
batch level and generate many utterances in the cloned voice without repeating
the reference for each item.

Start the server (with batch-optimized config for best throughput):

    vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts_batch.yaml \
        --trust-remote-code

Examples:
    # Batch with a predefined voice
    python batch_speech_client.py \
        --texts "Hello, how are you?" "Goodbye, see you later!"

    # Voice cloning: one ref_audio, many outputs
    python batch_speech_client.py \
        --task-type Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio" \
        --texts "First cloned sentence." "Second cloned sentence." \
               "Third cloned sentence."
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

    ext = os.path.splitext(audio_path)[1].lower()
    mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg"}
    mime_type = mime_map.get(ext, "audio/wav")

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_batch(args) -> None:
    """Send a batch TTS request and save each result to a file."""
    items = [{"input": text} for text in args.texts]

    payload: dict = {
        "items": items,
        "response_format": args.response_format,
    }
    if args.voice:
        payload["voice"] = args.voice
    if args.language:
        payload["language"] = args.language
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens

    # Voice cloning parameters (shared across all items)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    print(f"Sending batch of {len(items)} item(s) to {args.api_base}")
    if args.ref_audio:
        print("Voice cloning mode — ref_audio applied to all items")

    url = f"{args.api_base}/v1/audio/speech/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return

    data = response.json()
    print(f"Total: {data['total']}  Succeeded: {data['succeeded']}  Failed: {data['failed']}")

    os.makedirs(args.output_dir, exist_ok=True)
    for result in data["results"]:
        idx = result["index"]
        if result["status"] == "success":
            audio_bytes = base64.b64decode(result["audio_data"])
            out_path = os.path.join(args.output_dir, f"batch_{idx}.{args.response_format}")
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            print(f"  [{idx}] saved {len(audio_bytes)} bytes -> {out_path}")
        else:
            print(f"  [{idx}] FAILED: {result['error']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch speech client for /v1/audio/speech/batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")

    # Texts to synthesize
    parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help="One or more texts to synthesize",
    )

    # Shared voice settings
    parser.add_argument("--voice", default="vivian", help="Speaker name (default: vivian)")
    parser.add_argument("--language", default=None, help="Language: Auto, Chinese, English, etc.")
    parser.add_argument("--instructions", default=None, help="Voice style/emotion instructions")
    parser.add_argument(
        "--task-type",
        default=None,
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type (default: CustomVoice)",
    )

    # Voice cloning (Base task)
    parser.add_argument("--ref-audio", default=None, help="Reference audio path or URL for voice cloning")
    parser.add_argument("--ref-text", default=None, help="Reference audio transcript for voice cloning")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens per item")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--output-dir", "-o", default="batch_output", help="Output directory (default: batch_output)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(args)
