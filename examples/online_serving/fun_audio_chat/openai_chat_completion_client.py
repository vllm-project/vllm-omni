# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI-compatible client for Fun-Audio-Chat-8B.

Usage:
    # Start the vLLM server first:
    vllm serve FunAudioLLM/Fun-Audio-Chat-8B --omni --port 8091

    # Then run this client:
    python openai_chat_completion_client.py --query-type use_audio
    python openai_chat_completion_client.py --query-type use_audio --audio-path /path/to/audio.wav
"""

import base64
import io
import os
from typing import NamedTuple

import soundfile as sf
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

SEED = 42

# Default API settings
DEFAULT_API_BASE = "http://localhost:8091/v1"
DEFAULT_MODEL = "FunAudioLLM/Fun-Audio-Chat-8B"


class QueryResult(NamedTuple):
    messages: list[dict]
    sampling_params_list: list[dict]


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        return base64.b64encode(content).decode("utf-8")


def get_audio_url_from_path(audio_path: str | None) -> str:
    """Convert an audio path to an audio URL format for the API."""
    if not audio_path:
        return AudioAsset("mary_had_lamb").url

    if audio_path.startswith(("http://", "https://")):
        return audio_path

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect audio MIME type
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def build_sampling_params(s2s_mode: bool = False) -> list[dict]:
    """Build sampling parameters list."""
    main_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 2048,
        "detokenize": True,
        "repetition_penalty": 1.05,
        "seed": SEED,
    }

    if s2s_mode:
        crq_params = {
            "temperature": 0.9,
            "top_k": 50,
            "max_tokens": 4096,
            "seed": SEED,
            "detokenize": False,
            "repetition_penalty": 1.05,
        }
        cosyvoice_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 4096 * 16,
            "seed": SEED,
            "detokenize": True,
            "repetition_penalty": 1.1,
        }
        return [main_params, crq_params, cosyvoice_params]
    else:
        return [main_params]


def get_text_query(prompt: str | None = None) -> QueryResult:
    """Build text-only query."""
    if prompt is None:
        prompt = "Hello! How are you today?"

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful voice assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        },
    ]

    return QueryResult(
        messages=messages,
        sampling_params_list=build_sampling_params(s2s_mode=False),
    )


def get_audio_query(
    audio_path: str | None = None,
    prompt: str | None = None,
    s2s_mode: bool = False,
) -> QueryResult:
    """Build audio query."""
    if prompt is None:
        prompt = "Please respond to the audio."

    audio_url = get_audio_url_from_path(audio_path)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful voice assistant that can understand speech and respond naturally.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_url},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    return QueryResult(
        messages=messages,
        sampling_params_list=build_sampling_params(s2s_mode=s2s_mode),
    )


def main(args):
    # Initialize client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    print(f"[Info] API Base: {args.api_base}")
    print(f"[Info] Model: {args.model}")
    print(f"[Info] Query type: {args.query_type}")
    print(f"[Info] S2S mode: {args.s2s}")

    # Build query
    if args.query_type == "text":
        query = get_text_query(args.prompt)
    else:  # use_audio
        query = get_audio_query(
            audio_path=args.audio_path,
            prompt=args.prompt,
            s2s_mode=args.s2s,
        )

    print("[Info] Sending request...")

    # Send request
    try:
        chat_completion = client.chat.completions.create(
            messages=query.messages,
            model=args.model,
            extra_body={
                "sampling_params_list": query.sampling_params_list,
            },
        )

        # Process response
        print("\n" + "=" * 50)
        print("Response:")
        print("=" * 50)

        for choice in chat_completion.choices:
            if choice.message.content:
                print(f"\nText: {choice.message.content}")

            if hasattr(choice.message, "audio") and choice.message.audio:
                # Save audio output
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, "response.wav")

                audio_bytes = base64.b64decode(choice.message.audio.data)
                audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))

                sf.write(output_path, audio_np, sample_rate, format="WAV")
                print(f"\nAudio saved to: {output_path}")

        print("\n" + "=" * 50)

    except Exception as e:
        print(f"[Error] Request failed: {e}")
        raise


def parse_args():
    parser = FlexibleArgumentParser(description="OpenAI-compatible client for Fun-Audio-Chat-8B")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_audio",
        choices=["text", "use_audio"],
        help="Query type. Default: use_audio",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file or URL. Default: uses built-in test audio",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom text prompt. Default: auto-generated",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL. Default: {DEFAULT_API_BASE}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for audio files. Default: output",
    )
    parser.add_argument(
        "--s2s",
        action="store_true",
        help="Enable S2S mode (speech-to-speech output). Default: False",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
