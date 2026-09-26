"""
Client example for Covo-Audio-Chat with vllm-omni.

Usage
-----
# Start the server first:
#   CUDA_VISIBLE_DEVICES=0 vllm serve tencent/Covo-Audio-Chat --omni --trust-remote-code

# Audio input chat (uses default audio asset if --audio-path not provided):
python openai_chat_completion_client.py
python openai_chat_completion_client.py --audio-path /path/to/audio.wav

# Streaming mode:
python openai_chat_completion_client.py --stream
"""

import base64
import os
import sys
import time

from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
    COVO_AUDIO_SYSTEM_PROMPT,
)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:18091/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

SEED = 42


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_audio_url(audio_path: str | None) -> str:
    """Convert an audio path to a data URL for the API."""
    if not audio_path:
        return AudioAsset("mary_had_lamb").url

    if audio_path.startswith(("data:", "http://", "https://")):
        return audio_path

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
    }
    mime = mime_map.get(ext, "audio/wav")
    b64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime};base64,{b64}"


def build_audio_messages(audio_path: str | None, prompt: str) -> list[dict]:
    """Build messages for an audio chat request."""
    audio_url = get_audio_url(audio_path)
    return [
        {"role": "system", "content": COVO_AUDIO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def get_model_name(override: str | None = None) -> str:
    """Return the model name to use in API calls."""
    if override:
        return override
    # Auto-detect from the server
    models = client.models.list()
    return models.data[0].id


def run(args) -> None:
    model_name = get_model_name(args.model)
    thinker_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": 1.05,
        "stop_token_ids": [151645],
        "ignore_eos": True,
    }
    code2wav_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": SEED,
        "detokenize": False,
        "repetition_penalty": 1.05,
    }

    extra_body = {
        "sampling_params_list": [thinker_params, code2wav_params],
    }

    messages = build_audio_messages(args.audio_path, args.prompt)

    start = time.perf_counter()

    if not args.stream:
        completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            extra_body=extra_body,
            stream=False,
        )
        elapsed = time.perf_counter() - start

        for choice in completion.choices:
            if choice.message.audio:
                audio_bytes = base64.b64decode(choice.message.audio.data)
                out_path = args.output_audio_path
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"Audio saved to {out_path} ({len(audio_bytes)} bytes)")
            elif choice.message.content:
                print("Chat completion output from text:", choice.message.content)

        print(f"Elapsed: {elapsed:.2f}s", file=sys.stderr)

    else:
        stream = client.chat.completions.create(
            messages=messages,
            model=model_name,
            extra_body=extra_body,
            stream=True,
        )
        audio_idx = 0
        for chunk in stream:
            modality = getattr(chunk, "modality", None)
            elapsed = time.perf_counter() - start
            for choice in chunk.choices:
                content = getattr(choice.delta, "content", None) if hasattr(choice, "delta") else None
                if not content:
                    continue
                if modality == "audio":
                    audio_bytes = base64.b64decode(content)
                    out_dir = os.path.dirname(args.output_audio_path) or "."
                    out_path = os.path.join(out_dir, f"audio_{audio_idx}.wav")
                    os.makedirs(out_dir, exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(audio_bytes)
                    print(f"[{elapsed:.2f}s] Audio chunk saved to {out_path}", file=sys.stderr)
                    audio_idx += 1
                elif modality == "text":
                    print(content, end="", flush=True)

        print()  # trailing newline
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.2f}s", file=sys.stderr)


def parse_args():
    parser = FlexibleArgumentParser(description="Covo-Audio-Chat client example")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="请回答这段音频里的问题。",
        help="Text prompt.",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to audio file. Uses default audio asset if omitted.",
    )
    parser.add_argument(
        "--output-audio-path",
        "-o",
        type=str,
        default="./audio_0.wav",
        help="Output path for generated audio.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for the API. Auto-detected from server if omitted.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
