# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI-compatible client for Step-Audio2 online serving.

Step-Audio2 is a two-stage model:
  - Stage 0 (Thinker): Audio understanding -> Text + Audio tokens
  - Stage 1 (Token2Wav): Audio tokens -> Waveform (24kHz)

Usage examples:
  # Start server first:
  vllm serve stepfun-ai/Step-Audio-2-mini --omni --port 8092

  # Audio to text (ASR)
  python openai_chat_completion_client.py --query-type audio_to_text

  # Text to audio (TTS)
  python openai_chat_completion_client.py --query-type text_to_audio --text "Hello world"

  # Audio to audio (Voice conversion)
  python openai_chat_completion_client.py --query-type audio_to_audio --audio-path input.wav
"""

import base64
import logging
import os

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_base64_content_from_url(url: str) -> str:
    """Encode content from URL to base64 format."""
    with requests.get(url) as response:
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


def get_audio_url(audio_path: str | None = None) -> str:
    """Convert audio path to URL format for the API.

    Args:
        audio_path: Local file path or URL. If None, uses default test audio.

    Returns:
        Audio URL (either original URL or base64 data URL)
    """
    if not audio_path:
        return AudioAsset("mary_had_lamb").url

    if audio_path.startswith(("http://", "https://")):
        return audio_path

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    else:
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def get_audio_to_text_query(
    audio_path: str | None = None,
    custom_prompt: str | None = None,
) -> dict:
    """Build query for Audio-to-Text (ASR) mode."""
    question = custom_prompt or "Please transcribe the audio content."
    audio_url = get_audio_url(audio_path)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "text", "text": question},
        ],
    }


def get_text_to_audio_query(
    text: str | None = None,
    custom_prompt: str | None = None,
) -> dict:
    """Build query for Text-to-Audio (TTS) mode.

    Returns a list of messages: user message + assistant message with <tts_start>.
    The assistant message triggers TTS mode and must be used with
    continue_final_message=True so the chat template does not append <|im_end|>.
    """
    text_to_speak = text or "Hello, this is a test of Step Audio 2 text to speech."

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": text_to_speak}],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
        },
    ]


def get_audio_to_audio_query(
    audio_path: str | None = None,
    custom_prompt: str | None = None,
) -> dict:
    """Build query for Audio-to-Audio (S2ST) mode.

    Returns a list of messages: user message (with audio) + assistant message
    with <tts_start>. Must be used with continue_final_message=True.
    """
    question = custom_prompt or "Please listen to this audio and repeat its content."
    audio_url = get_audio_url(audio_path)

    return [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_url}},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
        },
    ]


def get_system_prompt(query_type: str) -> dict:
    """Get system prompt based on query type."""
    if query_type == "audio_to_text":
        system_text = "You are a speech recognition assistant. Transcribe the audio accurately."
    elif query_type == "text_to_audio":
        system_text = "You are a text-to-speech assistant. Read the text aloud exactly as provided."
    else:  # audio_to_audio
        system_text = "You are an audio processing assistant. Listen and repeat the audio content."

    return {
        "role": "system",
        "content": [{"type": "text", "text": system_text}],
    }


QUERY_MAP = {
    "audio_to_text": get_audio_to_text_query,
    "text_to_audio": get_text_to_audio_query,
    "audio_to_audio": get_audio_to_audio_query,
}


def run_step_audio2(args) -> None:
    """Run Step-Audio2 inference via OpenAI-compatible API."""

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    model_name = args.model

    # Sampling parameters for each stage
    thinker_sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_tokens": args.max_tokens,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": 1.1 if args.query_type != "audio_to_text" else 1.05,
    }

    # For ASR, stop at EOS token
    if args.query_type == "audio_to_text":
        thinker_sampling_params["stop_token_ids"] = [151645]

    token2wav_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 1,
        "seed": SEED,
        "detokenize": False,
    }

    sampling_params_list = [
        thinker_sampling_params,
        token2wav_sampling_params,
    ]

    # Build query based on type
    query_func = QUERY_MAP[args.query_type]
    if args.query_type == "audio_to_text":
        user_message = query_func(audio_path=args.audio_path, custom_prompt=args.prompt)
    elif args.query_type == "text_to_audio":
        user_message = query_func(text=args.text, custom_prompt=args.prompt)
    else:  # audio_to_audio
        user_message = query_func(audio_path=args.audio_path, custom_prompt=args.prompt)

    system_message = get_system_prompt(args.query_type)

    logger.info(f"Query type: {args.query_type}")
    logger.info(f"Model: {model_name}")
    logger.info("Sending request to server...")

    # Build messages list.
    # TTS and S2ST return a list of messages (user + assistant with <tts_start>);
    # ASR returns a single user message dict.
    if isinstance(user_message, list):
        messages = [system_message] + user_message
    else:
        messages = [system_message, user_message]

    # For TTS/S2ST, use continue_final_message so <tts_start> in the
    # assistant turn is not followed by <|im_end|>.
    extra_body: dict = {"sampling_params_list": sampling_params_list}
    if args.query_type in ("text_to_audio", "audio_to_audio"):
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False

    # Send request
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        extra_body=extra_body,
    )

    # Process response
    os.makedirs(args.output_dir, exist_ok=True)

    audio_count = 0
    for choice in chat_completion.choices:
        if choice.message.audio:
            # Audio response
            audio_data = base64.b64decode(choice.message.audio.data)
            audio_file = os.path.join(args.output_dir, f"audio_{audio_count}.wav")
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            logger.info(f"Audio saved to: {audio_file}")
            audio_count += 1

        if choice.message.content:
            # Text response
            logger.info(f"Text output: {choice.message.content}")

    if audio_count == 0 and args.query_type != "audio_to_text":
        logger.warning("No audio output received. Check server logs for details.")

    logger.info("Done!")


def parse_args():
    parser = FlexibleArgumentParser(description="OpenAI-compatible client for Step-Audio2 online serving")

    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="audio_to_text",
        choices=QUERY_MAP.keys(),
        help="Query type: audio_to_text (ASR), text_to_audio (TTS), audio_to_audio",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to input audio file (local path or URL)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text to synthesize (for text_to_audio mode)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom prompt/question",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="stepfun-ai/Step-Audio-2-mini",
        help="Model name",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8092/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for Thinker stage",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output_online",
        help="Output directory for audio files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_step_audio2(args)
