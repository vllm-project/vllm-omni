"""Client-side orchestrator for split AURA services.

The three semantic services are:
  1. Qwen3-ASR service: audio -> transcript
  2. AURA service: transcript + video -> response text
  3. Qwen3-TTS service: response text -> audio
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    DEFAULT_QWEN3_TTS_REF_TEXT,
    SILENT_TEXT,
    default_qwen3_tts_ref_audio_path,
)

DEFAULT_VIDEO_URL = "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"
PCM_SAMPLE_RATE = 24000


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


def text_from_chat_response(response) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(parts).strip()
    return str(content or "").strip()


def run_asr(args, audio_url: str) -> str:
    client = OpenAI(base_url=f"{args.asr_base_url}/v1", api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.asr_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": args.asr_prompt},
                ],
            }
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.asr_max_tokens,
        timeout=args.timeout,
    )
    transcript = text_from_chat_response(response)
    if not transcript:
        raise RuntimeError("ASR service returned empty transcript.")
    return transcript


def run_aura(args, video_url: str, transcript: str) -> str:
    client = OpenAI(base_url=f"{args.aura_base_url}/v1", api_key=args.api_key)
    content: list[dict] = [{"type": "text", "text": transcript}]
    if video_url:
        content.insert(0, {"type": "video_url", "video_url": {"url": video_url}})

    response = client.chat.completions.create(
        model=args.aura_model,
        messages=[
            {"role": "system", "content": args.aura_system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=args.aura_temperature,
        top_p=1.0,
        max_tokens=args.aura_max_tokens,
        extra_body={"top_k": -1, "repetition_penalty": 1.0},
        timeout=args.timeout,
    )
    aura_text = text_from_chat_response(response)
    if not aura_text:
        raise RuntimeError("AURA service returned empty text.")
    return aura_text


def _tts_payload(args, text: str) -> dict:
    payload: dict = {
        "model": args.tts_model,
        "input": text,
        "task_type": args.tts_task_type,
        "language": args.tts_language,
        "response_format": "pcm" if args.tts_stream else args.tts_response_format,
        "stream": args.tts_stream,
    }
    if args.tts_task_type == "CustomVoice":
        payload["voice"] = args.tts_speaker
        if args.tts_instruct:
            payload["instructions"] = args.tts_instruct
    elif args.tts_task_type == "Base":
        payload["ref_audio"] = media_url(args.tts_ref_audio, kind="audio")
        payload["ref_text"] = args.tts_ref_text
        if args.tts_x_vector_only_mode:
            payload["x_vector_only_mode"] = True
    return payload


def run_tts(args, text: str, output_path: Path) -> None:
    payload = _tts_payload(args, text)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    api_url = f"{args.tts_base_url}/v1/audio/speech"

    with httpx.Client(timeout=args.timeout) as client:
        if not args.tts_stream:
            response = client.post(api_url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"TTS service error {response.status_code}: {response.text}")
            output_path.write_bytes(response.content)
            return

        chunks: list[bytes] = []
        with client.stream("POST", api_url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                response.read()
                raise RuntimeError(f"TTS service error {response.status_code}: {response.text}")
            for chunk in response.iter_bytes():
                if chunk:
                    chunks.append(chunk)

    pcm = b"".join(chunks)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    sf.write(output_path, audio, PCM_SAMPLE_RATE, format="WAV")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_url = media_url(args.audio_path, kind="audio")
    video_url = media_url(args.video_path, kind="video") if args.video_path or not args.no_video else ""

    transcript = run_asr(args, audio_url)
    write_text(output_dir / "asr_transcript.txt", transcript)
    print(f"ASR transcript: {transcript}")

    aura_text = run_aura(args, video_url, transcript)
    write_text(output_dir / "aura_response.txt", aura_text)
    print(f"AURA response: {aura_text}")

    if aura_text.strip() == SILENT_TEXT:
        print("AURA returned <|silent|>; skipping TTS.")
        return

    suffix = "wav" if args.tts_stream else args.tts_response_format
    audio_path = output_dir / f"tts_output.{suffix}"
    run_tts(args, aura_text, audio_path)
    print(f"TTS audio saved to {audio_path}")


def parse_args():
    parser = FlexibleArgumentParser(description="AURA split-services online client")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--asr-base-url", default="http://localhost:8661")
    parser.add_argument("--aura-base-url", default="http://localhost:8662")
    parser.add_argument("--tts-base-url", default="http://localhost:8663")
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--aura-model", default="aurateam/AURA")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--audio-path", default=None, help="Audio file, URL, or data URL.")
    parser.add_argument("--video-path", default=None, help="Video file, URL, or data URL.")
    parser.add_argument("--no-video", action="store_true", help="Do not send video to AURA.")
    parser.add_argument("--output-dir", default="output_aura_split_services")
    parser.add_argument("--asr-prompt", default="Transcribe the audio accurately.")
    parser.add_argument("--asr-max-tokens", type=int, default=256)
    parser.add_argument("--aura-system-prompt", default=DEFAULT_AURA_SYSTEM_PROMPT)
    parser.add_argument("--aura-temperature", type=float, default=0.5)
    parser.add_argument("--aura-max-tokens", type=int, default=256)
    parser.add_argument("--tts-task-type", default="CustomVoice", choices=["Base", "CustomVoice"])
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-speaker", default="Vivian")
    parser.add_argument("--tts-instruct", default="")
    parser.add_argument("--tts-ref-audio", default=default_qwen3_tts_ref_audio_path())
    parser.add_argument("--tts-ref-text", default=DEFAULT_QWEN3_TTS_REF_TEXT)
    parser.add_argument("--tts-x-vector-only-mode", action="store_true")
    parser.add_argument("--tts-response-format", default="wav", choices=["wav", "mp3", "flac", "opus", "aac", "pcm"])
    parser.add_argument("--no-tts-stream", dest="tts_stream", action="store_false")
    parser.set_defaults(tts_stream=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
