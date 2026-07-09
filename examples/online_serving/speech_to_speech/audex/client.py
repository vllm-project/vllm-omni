# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cascaded Audex speech-to-speech client (mirrors the official web demo).

Runs the official three passes against ONE ``nemotron_labs_audex_full``
server:

  1. ASR  — /v1/chat/completions with the WAV as ``input_audio`` content
  2. chat — /v1/chat/completions on the transcript
  3. TTS  — /v1/audio/speech on the answer (optional CFG via extra_params)

Start the server first, e.g.:

    vllm-omni serve nvidia/Nemotron-Labs-Audex-2B --omni --port 8098 \\
        --trust-remote-code \\
        --stage-configs-path vllm_omni/deploy/nemotron_labs_audex_full.yaml

Then:

    python examples/online_serving/speech_to_speech/audex/client.py \\
        --audio-file question.wav --port 8098 --output answer.wav
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import requests

ASR_QUESTION = "Transcribe the input speech."


def parse_args():
    parser = argparse.ArgumentParser(description="Audex cascaded S2S client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--audio-file", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/audex_s2s_answer.wav")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def _chat(base_url: str, model: str, messages: list[dict], max_tokens: int) -> str:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _clean_transcript(text: str) -> str:
    text = text.strip()
    if "'" in text:
        parts = text.split("'")
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    audio_b64 = base64.b64encode(Path(args.audio_file).read_bytes()).decode()

    transcript = _clean_transcript(
        _chat(
            base_url,
            args.model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                        {"type": "text", "text": ASR_QUESTION},
                    ],
                }
            ],
            args.max_tokens,
        )
    )
    print(f"[1/3] transcript : {transcript}")
    if not transcript:
        raise SystemExit("ASR pass produced an empty transcript; aborting cascade")

    answer = _chat(base_url, args.model, [{"role": "user", "content": transcript}], args.max_tokens)
    print(f"[2/3] answer     : {answer}")
    if not answer:
        raise SystemExit("Chat pass produced an empty answer; aborting cascade")

    speech_payload: dict = {"model": args.model, "input": answer, "response_format": "wav"}
    if args.cfg_scale > 1.0:
        speech_payload["extra_params"] = {"cfg_scale": args.cfg_scale}
    resp = requests.post(f"{base_url}/v1/audio/speech", json=speech_payload, timeout=600)
    resp.raise_for_status()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"[3/3] speech     : {len(resp.content)} bytes -> {out_path}")


if __name__ == "__main__":
    main()
