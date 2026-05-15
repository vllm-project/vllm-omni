# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch client for the higgs-audio v2 online server.

Sends a fixed list of prompts to ``/v1/audio/speech`` and saves the returned
WAV files (or raw PCM bytes when ``--format pcm``) into ``--output-dir``.

Usage:

  python examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py \
      --base-url http://localhost:8094 \
      --output-dir /tmp/higgs_audio_v2_batch \
      --prompts "Hello world." "The quick brown fox jumps over the lazy dog."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_PROMPTS = (
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "It was the night before my birthday.",
    "Innovation distinguishes between a leader and a follower.",
)


def _slug(text: str) -> str:
    import re

    s = re.sub(r"\s+", "_", text.strip().lower())
    return re.sub(r"[^a-z0-9_]+", "", s)[:32] or "prompt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8094")
    parser.add_argument("--model", default="higgs_audio_v2")
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/higgs_audio_v2_batch"))
    parser.add_argument("--format", choices=("wav", "pcm"), default="wav")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print(
            "this client needs `httpx`. Install with `pip install httpx`.",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = args.base_url.rstrip("/") + "/v1/audio/speech"
    failures = 0
    with httpx.Client(timeout=args.timeout_s) as client:
        for prompt in args.prompts:
            payload = {
                "model": args.model,
                "input": prompt,
                "response_format": args.format,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            }
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                print(f"[FAIL] {prompt!r} -> {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
                failures += 1
                continue
            suffix = ".wav" if args.format == "wav" else ".pcm"
            out = args.output_dir / f"{_slug(prompt)}{suffix}"
            out.write_bytes(resp.content)
            print(f"[ ok ] {prompt!r} -> {out} ({len(resp.content)} bytes)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
