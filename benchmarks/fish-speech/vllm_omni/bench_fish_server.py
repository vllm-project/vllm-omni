"""Benchmark Fish Speech S2 Pro via vllm-omni /v1/audio/speech endpoint.

Thin wrapper around the shared TTS benchmark infrastructure, providing
fish-speech-specific payload construction and audio parameters.

Usage:
    python bench_fish_server.py \
        --host 127.0.0.1 --port 8091 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --result-dir results/

    # With voice cloning
    python bench_fish_server.py \
        --port 8091 \
        --ref-audio https://example.com/ref.wav \
        --ref-text "Reference transcript" \
        --result-dir results/
"""

import argparse
import asyncio
import sys
from functools import partial
from pathlib import Path

# Allow imports from benchmarks/fish-speech/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fish_bench_utils import run_benchmark_sweep  # noqa: E402

# Fish Speech S2 Pro: DAC decoder outputs 44.1 kHz 16-bit mono PCM by default.
DEFAULT_SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DEFAULT_CHANNELS = 1
REQUEST_DEFAULTS = {
    # Match the stage config's AR generation limit explicitly so benchmark
    # requests do not depend on server-side defaults.
    "max_new_tokens": 2048,
}


def create_payload(
    prompt: str,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> dict:
    """Build a vllm-omni /v1/audio/speech request for Fish Speech."""
    payload: dict = {
        "input": prompt,
        "voice": "default",
        "stream": True,
        "response_format": "pcm",
    }
    payload.update(REQUEST_DEFAULTS)
    if ref_audio:
        payload["ref_audio"] = ref_audio
    if ref_text:
        payload["ref_text"] = ref_text
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Fish Speech S2 Pro Benchmark (vllm-omni)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 10],
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--config-name", type=str, default="vllm_omni")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


async def main():
    args = parse_args()
    payload_fn = partial(
        create_payload,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )
    await run_benchmark_sweep(
        host=args.host,
        port=args.port,
        num_prompts=args.num_prompts,
        concurrency_levels=args.max_concurrency,
        create_payload_fn=payload_fn,
        sample_rate=args.sample_rate,
        sample_width=SAMPLE_WIDTH,
        sample_channels=args.channels,
        num_warmups=args.num_warmups,
        request_timeout_s=args.request_timeout,
        config_name=args.config_name,
        result_dir=args.result_dir,
        result_timestamp=args.timestamp,
    )


if __name__ == "__main__":
    asyncio.run(main())
