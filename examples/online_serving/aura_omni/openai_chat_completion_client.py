# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for the AURA Omni pipeline."""

from __future__ import annotations

import base64
import io
import os

import soundfile as sf
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    DEFAULT_QWEN3_TTS_REF_TEXT,
    default_qwen3_tts_ref_audio_path,
)

SEED = 42
DEFAULT_MODEL = "aurateam/AURA"
DEFAULT_VIDEO_URL = "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"


def _encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _data_url(path: str, default_mime: str) -> str:
    suffix = os.path.splitext(path)[1].lower()
    mime_by_suffix = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    return f"data:{mime_by_suffix.get(suffix, default_mime)};base64,{_encode_file(path)}"


def media_url(path_or_url: str | None, *, kind: str) -> str:
    if path_or_url:
        if path_or_url.startswith(("http://", "https://", "data:")):
            return path_or_url
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"{kind} file not found: {path_or_url}")
        return _data_url(path_or_url, "audio/wav" if kind == "audio" else "video/mp4")
    if kind == "audio":
        return AudioAsset("mary_had_lamb").url
    return DEFAULT_VIDEO_URL


def sampling_params_list() -> list[dict]:
    return [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 256, "seed": SEED},
        {
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 256,
            "seed": SEED,
            "repetition_penalty": 1.0,
        },
        {
            "temperature": 0.9,
            "top_k": 50,
            "max_tokens": 4096,
            "seed": SEED,
            "detokenize": False,
            "repetition_penalty": 1.05,
            "stop_token_ids": [2150],
        },
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 65536,
            "seed": SEED,
            "repetition_penalty": 1.0,
        },
    ]


def parse_modalities(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def save_response(response, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for idx, choice in enumerate(response.choices):
        message = choice.message
        if message.content:
            out_txt = os.path.join(output_dir, f"choice_{idx}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(str(message.content).strip() + "\n")
            print(f"Text saved to {out_txt}")
            print(message.content)
        if getattr(message, "audio", None):
            audio_bytes = base64.b64decode(message.audio.data)
            audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))
            out_wav = os.path.join(output_dir, f"choice_{idx}.wav")
            sf.write(out_wav, audio_np, int(sample_rate), format="WAV")
            print(f"Audio saved to {out_wav}")


def main(args) -> None:
    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": media_url(args.audio_path, kind="audio")}},
                {"type": "video_url", "video_url": {"url": media_url(args.video_path, kind="video")}},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        modalities=parse_modalities(args.modalities),
        extra_body={
            "sampling_params_list": sampling_params_list(),
            "additional_information": {
                "aura_system_prompt": args.aura_system_prompt,
                "tts_task_type": args.tts_task_type,
                "tts_language": args.tts_language,
                "tts_speaker": args.tts_speaker,
                "tts_instruct": args.tts_instruct,
                "tts_ref_audio": args.tts_ref_audio,
                "tts_ref_text": args.tts_ref_text,
                "tts_x_vector_only_mode": args.tts_x_vector_only_mode,
                "tts_pass_token_ids": args.tts_pass_token_ids,
            },
        },
        timeout=args.timeout,
    )
    save_response(response, args.output_dir)


def parse_args():
    parser = FlexibleArgumentParser(description="AURA Omni online serving client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio-path", default=None, help="Audio file, URL, or data URL.")
    parser.add_argument("--video-path", default=None, help="Video file, URL, or data URL.")
    parser.add_argument(
        "--prompt",
        default="Use the audio and video together to decide whether a reply is needed. If needed, respond briefly in English.",
    )
    parser.add_argument("--modalities", default="text,audio")
    parser.add_argument("--output-dir", default="output_aura_omni_online")
    parser.add_argument(
        "--aura-system-prompt",
        default=(
            "You are receiving a live video stream where the final frame is the present moment. "
            "Respond only when a response is needed. Otherwise output '<|silent|>'. Respond in English."
        ),
    )
    parser.add_argument("--tts-task-type", default="Base", choices=["Base", "CustomVoice"])
    parser.add_argument("--tts-language", default="English")
    parser.add_argument("--tts-speaker", default="Vivian")
    parser.add_argument("--tts-instruct", default="")
    parser.add_argument(
        "--tts-ref-audio",
        default=default_qwen3_tts_ref_audio_path(),
        help="Base-mode reference audio path/URL visible to server.",
    )
    parser.add_argument(
        "--tts-ref-text",
        default=DEFAULT_QWEN3_TTS_REF_TEXT,
        help="Base-mode reference audio transcript.",
    )
    parser.add_argument(
        "--tts-x-vector-only-mode",
        action="store_true",
        help="Use speaker embedding only for Base mode (disable ICL ref_text conditioning).",
    )
    parser.add_argument(
        "--tts-pass-token-ids",
        action="store_true",
        help="Pass AURA-generated assistant token ids directly to Qwen3-TTS. Defaults to sending text.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
