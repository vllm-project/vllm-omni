# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch client for the higgs-audio v3 online server.

Sends prompts to ``/v1/audio/speech`` and saves the returned WAV files.

Usage (plain text -> speech):

  python examples/online_serving/text_to_speech/higgs_audio_v3/batch_speech_client.py \
      --base-url http://localhost:8095 \
      --output-dir /tmp/higgs_v3_batch \
      --prompts "Hello world." "The quick brown fox jumps over the lazy dog."

Usage (voice clone):

  python examples/online_serving/text_to_speech/higgs_audio_v3/batch_speech_client.py \
      --base-url http://localhost:8095 \
      --output-dir /tmp/higgs_v3_clone \
      --ref-audio path/to/reference.wav \
      --ref-text "the transcript of the reference clip" \
      --prompts "Hello world."
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

DEFAULT_PROMPTS = (
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Today is a beautiful day for a walk in the park.",
    "Innovation distinguishes between a leader and a follower.",
)


def _slug(text: str) -> str:
    import re

    s = re.sub(r"\s+", "_", text.strip().lower())
    return re.sub(r"[^a-z0-9_]+", "", s)[:32] or "prompt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8095")
    parser.add_argument("--model", default="higgs_audio_v3")
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/higgs_v3_batch"))
    parser.add_argument("--format", choices=("wav", "pcm"), default="wav")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help="Reference clip for voice clone (WAV/FLAC/MP3 path). Pair with --ref-text.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference clip. Optional but improves fidelity.",
    )
    args = parser.parse_args()

    ref_audio_data_url: str | None = None
    if args.ref_audio is not None:
        if not args.ref_audio.exists():
            print(f"ref-audio file not found: {args.ref_audio}", file=sys.stderr)
            return 2
        mime = "audio/wav" if args.ref_audio.suffix.lower() == ".wav" else "audio/mpeg"
        ref_b64 = base64.b64encode(args.ref_audio.read_bytes()).decode("ascii")
        ref_audio_data_url = f"data:{mime};base64,{ref_b64}"

    try:
        import httpx
    except ImportError:
        print("this client needs `httpx`. Install with `pip install httpx`.", file=sys.stderr)
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
            if ref_audio_data_url is not None:
                payload["ref_audio"] = ref_audio_data_url
                if args.ref_text:
                    payload["ref_text"] = args.ref_text
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
