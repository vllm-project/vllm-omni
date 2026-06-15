# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for Ming-omni-tts via /v1/audio/speech.

Examples:
    python openai_speech_client.py --text "你好，世界"
    python openai_speech_client.py --text "我觉得社会企业同个人都有责任" \
        --instruction-json '{"方言":"广粤话"}' --ref-audio yue_prompt.wav --max-new-tokens 200
    python openai_speech_client.py \
        --text "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。" \
        --ref-audio 10002287-00000094.wav --ref-text "在此奉劝大家别乱打美白针。" \
        --max-new-tokens 200
    python openai_speech_client.py --text "speaker_1:你好。 speaker_2:你好。" \
        --ref-audio speaker_1.wav --ref-audio speaker_2.wav --ref-text "speaker_1:你好。 speaker_2:你好。"
    python openai_speech_client.py --text "你好，这是流式输出测试。" \
        --stream --output ming_output.pcm
"""

import argparse
import base64
import json
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "inclusionAI/Ming-omni-tts-0.5B"
EXPECTED_SPEAKER_EMBEDDING_DIM = 192


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to a base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "aac": "audio/aac",
    }
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def load_speaker_embedding(path: str) -> list[float]:
    """Load and validate a 192-d Ming speaker embedding JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("speaker_embedding file must contain a JSON list")
    if len(data) != EXPECTED_SPEAKER_EMBEDDING_DIM:
        raise ValueError(
            f"Ming dense speaker_embedding must have {EXPECTED_SPEAKER_EMBEDDING_DIM} values, got {len(data)}"
        )

    values = []
    for index, value in enumerate(data):
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"speaker_embedding[{index}] must be a number, got {value!r}") from exc
    return values


def build_instruction_payload(args) -> str | None:
    """Return a string payload for the API `instructions` field."""
    if args.instructions and args.instruction_json:
        raise ValueError("Use either --instructions or --instruction-json, not both")
    if args.instruction_json:
        parsed = json.loads(args.instruction_json)
        return json.dumps(parsed, ensure_ascii=False)
    return args.instructions


def validate_args(args) -> None:
    """Fail fast on invalid combinations before hitting the server."""
    if args.ref_text and not args.ref_audio:
        raise ValueError("--ref-audio is required when --ref-text is provided")
    if args.speaker_embedding and args.ref_audio and len(args.ref_audio) > 1:
        raise ValueError("--speaker-embedding cannot be combined with multiple --ref-audio values")


def run_tts(args) -> None:
    """Generate speech via the OpenAI-compatible /v1/audio/speech API."""
    validate_args(args)

    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }

    if args.voice:
        payload["voice"] = args.voice
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.dialect:
        payload["language"] = args.dialect

    instructions = build_instruction_payload(args)
    if instructions:
        payload["instructions"] = instructions

    if args.ref_audio:
        ref_audio = []
        for audio in args.ref_audio:
            if audio.startswith(("http://", "https://", "data:")):
                ref_audio.append(audio)
            else:
                ref_audio.append(encode_audio_to_base64(audio))
        payload["ref_audio"] = ref_audio[0] if len(ref_audio) == 1 else ref_audio
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.speaker_embedding:
        payload["speaker_embedding"] = load_speaker_embedding(args.speaker_embedding)
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Payload keys: {sorted(payload)}")

    if args.stream:
        output_path = args.output or "ming_output.pcm"
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(response.read().decode())
                    return
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
        print(f"Streamed PCM audio to: {output_path}")
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

    output_path = args.output or "ming_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible client for Ming-omni-tts via /v1/audio/speech")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--task-type",
        default=None,
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="Optional compatibility task type. Ming accepts the same field but primarily uses prompt metadata.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Maps to Ming `IP` when using built-in character voices, or to an uploaded voice sample name",
    )
    parser.add_argument("--dialect", default=None, help="Maps to Ming `方言`")
    parser.add_argument("--instructions", default=None, help="Free-form Ming instruction string")
    parser.add_argument(
        "--instruction-json",
        default=None,
        help='Structured Ming instruction JSON, for example \'{"情感":"高兴"}\'',
    )
    parser.add_argument(
        "--ref-audio",
        action="append",
        default=None,
        help="Reference audio path, URL, or data URL. Repeat for podcast-style multi-speaker prompts.",
    )
    parser.add_argument("--ref-text", default=None, help="Reference transcript for cloning")
    parser.add_argument(
        "--speaker-embedding", default=None, help="Path to a JSON file containing a 192-d speaker embedding"
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override ming_max_decode_steps")
    parser.add_argument("--stream", action="store_true", help="Enable streaming PCM output")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format when not streaming",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()
    try:
        run_tts(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
