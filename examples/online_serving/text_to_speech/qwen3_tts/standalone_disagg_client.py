# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Disaggregated Qwen3-TTS client: chains standalone stages over HTTP.

Reference coordinator that calls talker (stage 0) then forwards codec tokens
to code2wav (stage 1) to produce audio.

Start the servers first (see docs/features/standalone_disaggregation.md):
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni --standalone --stage-id 0 --port 8000
    CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni --standalone --stage-id 1 --port 8001

Examples:
    python standalone_disagg_client.py --text "Hello, how are you?" --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    python standalone_disagg_client.py --text "Hello" --voice vivian -o hello.wav --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
"""

import argparse
import sys
import time

import httpx

DEFAULT_TALKER_URL = "http://localhost:8000"
DEFAULT_CODE2WAV_URL = "http://localhost:8001"


def _post(client, url, body, label):
    resp = client.post(url, json=body)
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        print(f"{label} failed ({resp.status_code}): {detail}", file=sys.stderr)
        sys.exit(1)
    return resp


def main():
    parser = argparse.ArgumentParser(description="Disaggregated TTS via standalone stages")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="", help="Speaker voice name")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", help="Model name")
    parser.add_argument("--talker-url", default=DEFAULT_TALKER_URL, help="Talker stage URL")
    parser.add_argument("--code2wav-url", default=DEFAULT_CODE2WAV_URL, help="Code2wav stage URL")
    parser.add_argument("--response-format", default="wav", help="Audio format (wav, mp3, flac, pcm, opus)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.25-4.0)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output path")
    parser.add_argument("--timeout", type=int, default=120, help="Per-stage timeout (seconds)")
    args = parser.parse_args()

    with httpx.Client(timeout=args.timeout) as client:
        # Step 1: talker (entry mode)
        print(f"[1/2] Talker @ {args.talker_url} ...")
        t0 = time.time()
        resp = _post(
            client,
            f"{args.talker_url}/v1/stage/run",
            {
                "model": args.model,
                "input": args.text,
                "voice": args.voice,
                "response_format": args.response_format,
                "speed": args.speed,
            },
            "Talker",
        )
        talker_result = resp.json()
        t1 = time.time()

        stage_output = talker_result.get("stage_output")
        if stage_output is None:
            print(f"Talker returned no stage_output: {talker_result}", file=sys.stderr)
            sys.exit(1)

        codec_frames = len(stage_output.get("codes", {}).get("audio", []))
        print(f"      {t1 - t0:.1f}s, {codec_frames} codec frames")

        # Step 2: code2wav (downstream mode)
        print(f"[2/2] Code2wav @ {args.code2wav_url} ...")
        t2 = time.time()
        downstream_body = {
            "stage_output": stage_output,
            "request_id": talker_result.get("request_id", "disagg"),
            "response_format": args.response_format,
            "speed": args.speed,
        }
        resp = _post(
            client,
            f"{args.code2wav_url}/v1/stage/run",
            downstream_body,
            "Code2wav",
        )
        t3 = time.time()

        with open(args.output, "wb") as f:
            f.write(resp.content)

        print(f"      {t3 - t2:.1f}s, {len(resp.content)} bytes")
        print(f"Total {t3 - t0:.1f}s, saved to {args.output}")


if __name__ == "__main__":
    main()
