# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex online client for all four deployment modes.

Pick ``--mode`` to match the server's deploy yaml (see run_server.sh):

  tts           /v1/audio/speech: text -> speech WAV
  tta           /v1/audio/speech: caption -> general-audio WAV
  thinker_only  /v1/chat/completions: audio (+ question) -> text
  s2s           the official three-pass cascade against one audex_s2s
                server: ASR (chat, input_audio) -> chat -> TTS

Audio-input modes fall back to vLLM's public ``mary_had_lamb`` asset when
``--audio-file`` is omitted.

Examples:

    python examples/online_serving/audex/client.py --mode tts \\
        --text "Hello there." --output hello.wav
    python examples/online_serving/audex/client.py --mode tta \\
        --caption "Heavy rain falling on a tin roof." --output rain.wav
    python examples/online_serving/audex/client.py --mode thinker_only
    python examples/online_serving/audex/client.py --mode s2s \\
        --audio-file question.wav --output answer.wav
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import requests

ASR_QUESTION = "Transcribe the input speech."
DEFAULT_TEXT = "Hello there! This is the Audex text to speech pipeline."
DEFAULT_CAPTION = "Heavy rain falling on a tin roof."
# Official quality settings; 1.0 disables guidance.
DEFAULT_CFG = {"tts": 1.5, "tta": 3.0, "s2s": 1.5}


def parse_args():
    parser = argparse.ArgumentParser(description="Audex online client (tts / tta / thinker_only / s2s)")
    parser.add_argument("--mode", choices=("tts", "tta", "thinker_only", "s2s"), default="s2s")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="TTS input text (mode=tts).")
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION, help="TTA caption (mode=tta).")
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Input WAV for thinker_only/s2s. Defaults to vLLM's mary_had_lamb asset.",
    )
    parser.add_argument("--question", type=str, default=ASR_QUESTION, help="Instruction (mode=thinker_only).")
    parser.add_argument("--output", type=str, default="results/audex_online.wav")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="CFG strength; defaults per mode (tts/s2s 1.5, tta 3.0). 1.0 disables.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def _audio_b64(audio_file: str | None) -> str:
    if audio_file:
        return base64.b64encode(Path(audio_file).read_bytes()).decode()
    # Fall back to vLLM's public asset, re-encoded as WAV for the chat API.
    import io

    import numpy as np
    import soundfile as sf
    from vllm.assets.audio import AudioAsset

    audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype="float32"), int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _chat(base_url: str, model: str, messages: list[dict], max_tokens: int) -> str:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            # Text-final passes must say so explicitly: without "modalities"
            # the server falls back to the deployment's configured output
            # modalities, which for the s2s pipeline include the audio final
            # stage — routing ASR/chat through code2wav instead of stopping
            # at stage 0.
            "modalities": ["text"],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _speech(base_url: str, model: str, text: str, cfg_scale: float, out_path: Path) -> None:
    payload: dict = {"model": model, "input": text, "response_format": "wav"}
    if cfg_scale > 1.0:
        payload["extra_params"] = {"cfg_scale": cfg_scale}
    resp = requests.post(f"{base_url}/v1/audio/speech", json=payload, timeout=600)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)


def _audio_question(audio_b64: str, question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "text", "text": question},
            ],
        }
    ]


def _strip_think(text: str) -> str:
    # The chat template lets the model reason before answering; keep only
    # the part after the closing think tag so the TTS pass speaks the
    # actual answer, not the chain of thought.
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _clean_transcript(text: str) -> str:
    # The full checkpoint prefixes transcripts with a language-tag sentence
    # and quotes the content; extract the quoted payload when present.
    text = text.strip()
    if "'" in text:
        parts = text.split("'")
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else DEFAULT_CFG.get(args.mode, 1.0)
    out_path = Path(args.output)

    if args.mode == "tts":
        _speech(base_url, args.model, args.text, cfg_scale, out_path)
        print(f"speech: {out_path.stat().st_size} bytes -> {out_path}")
        return

    if args.mode == "tta":
        _speech(base_url, args.model, args.caption, cfg_scale, out_path)
        print(f"audio : {out_path.stat().st_size} bytes -> {out_path}")
        return

    audio_b64 = _audio_b64(args.audio_file)

    if args.mode == "thinker_only":
        answer = _strip_think(_chat(base_url, args.model, _audio_question(audio_b64, args.question), args.max_tokens))
        print(f"answer: {answer}")
        return

    # s2s — the official three-pass cascade.
    transcript = _clean_transcript(
        _chat(base_url, args.model, _audio_question(audio_b64, ASR_QUESTION), args.max_tokens)
    )
    print(f"[1/3] transcript : {transcript}")
    if not transcript:
        raise SystemExit("ASR pass produced an empty transcript; aborting cascade")

    answer = _strip_think(_chat(base_url, args.model, [{"role": "user", "content": transcript}], args.max_tokens))
    print(f"[2/3] answer     : {answer}")
    if not answer:
        raise SystemExit("Chat pass produced an empty answer; aborting cascade")

    _speech(base_url, args.model, answer, cfg_scale, out_path)
    print(f"[3/3] speech     : {out_path.stat().st_size} bytes -> {out_path}")


if __name__ == "__main__":
    main()
