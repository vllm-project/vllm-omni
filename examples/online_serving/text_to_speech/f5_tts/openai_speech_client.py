# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI-compatible client for F5-TTS via /v1/audio/speech endpoint.

F5-TTS is a flow-matching diffusion TTS model served as a diffusion pipeline
through vLLM-Omni's --omni flag.

Start the server:
  vllm serve SWivid/F5-TTS/F5TTS_v1_Base --omni --trust-remote-code \
    --enforce-eager --port 8091

  # With cache acceleration:
  vllm serve SWivid/F5-TTS/F5TTS_v1_Base --omni --trust-remote-code \
    --enforce-eager --port 8091 --cache-backend cache_dit

Examples:
  python openai_speech_client.py --text "Hello, this is F5-TTS speaking."
  python openai_speech_client.py --text "Hello." --ref-audio ref.wav --ref-text "Reference."
  python openai_speech_client.py --text "Hello." --num-inference-steps 16 --seed 42
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
REF_AUDIO_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
)
REF_TEXT = "Some call me nature, others call me mother nature."


def encode_audio_to_base64(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.rsplit(".", 1)[-1].lower()
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def run(args) -> bool:
    payload: dict = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
        "stream": args.stream,
    }

    if args.ref_audio:
        if os.path.exists(args.ref_audio):
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
        else:
            payload["ref_audio"] = args.ref_audio
    else:
        payload["ref_audio"] = REF_AUDIO_URL

    payload["ref_text"] = args.ref_text or REF_TEXT

    if args.num_inference_steps:
        payload["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        payload["guidance_scale"] = args.guidance_scale
    if args.seed is not None:
        payload["seed"] = args.seed

    url = f"{args.api_base}/v1/audio/speech"
    print(f"POST {url}")
    print(f"  text: {args.text[:80]}...")
    print(f"  steps: {args.num_inference_steps}, cfg: {args.guidance_scale}")

    try:
        with httpx.Client(timeout=300) as client:
            resp = client.post(url, json=payload, headers={
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
            })
        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text}")
            return False
        with open(args.output, "wb") as f:
            f.write(resp.content)
        print(f"Saved: {args.output} ({len(resp.content) / 1024:.1f} KB)")
        return True
    except httpx.ConnectError:
        print(f"Connection failed: {args.api_base}")
        return False


def parse_args():
    p = argparse.ArgumentParser(description="F5-TTS OpenAI speech client")
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="SWivid/F5-TTS/F5TTS_v1_Base")
    p.add_argument("--text", default=(
        "I don't really care what you call me. "
        "I've been a silent spectator, watching species evolve, "
        "empires rise and fall. But always, I am here."
    ))
    p.add_argument("--ref-audio", default=None, help="Path/URL to reference audio.")
    p.add_argument("--ref-text", default=None, help="Transcript of reference audio.")
    p.add_argument("--num-inference-steps", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--response-format", default="wav", choices=["wav", "mp3", "flac", "pcm"])
    p.add_argument("--stream", action="store_true")
    p.add_argument("--output", default="f5_tts_output.wav")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(0 if run(args) else 1)
