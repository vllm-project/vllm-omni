"""Benchmark Fish Speech voice cache: inline ref_audio vs uploaded voice.

Measures TTFP improvement from DAC-code caching when using uploaded voices.

Setup:
  1. Start vllm-omni with Fish Speech S2 Pro (use our feat branch)
  2. Provide a reference audio file for voice cloning

Usage:
    python bench_voice_cache.py \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio." \
        --num-prompts 20 \
        --port 8091

The script runs two rounds:
  A) Inline ref_audio: every request sends base64 audio (no cache)
  B) Uploaded voice: upload once, then use voice name (cache hits after 1st)
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

# Allow imports from benchmarks/fish-speech/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fish_bench_utils import (  # noqa: E402
    BenchmarkResult,
    RequestResult,
    compute_stats,
    print_benchmark_results,
    send_streaming_request,
)

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here.",
    "Please remember to bring your identification documents tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time?",
    "The restaurant on the corner serves the best pasta I have ever tasted.",
    "After the meeting, we should discuss the quarterly results.",
    "Learning a new language takes patience and genuine curiosity.",
    "The train leaves at half past seven, so we need to arrive early.",
    "Could you please turn down the music, I'm trying to concentrate.",
    "It was a dark and stormy night when the keeper heard a knock.",
]


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


async def upload_voice(
    host: str,
    port: int,
    audio_path: str,
    ref_text: str,
    voice_name: str = "bench_voice",
) -> dict:
    """Upload a voice via POST /v1/audio/voices."""
    url = f"http://{host}:{port}/v1/audio/voices"
    data = aiohttp.FormData()
    data.add_field("name", voice_name)
    data.add_field("consent", "true")
    if ref_text:
        data.add_field("ref_text", ref_text)
    data.add_field(
        "audio_sample",
        open(audio_path, "rb"),
        filename=os.path.basename(audio_path),
        content_type="audio/wav",
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as resp:
            result = await resp.json()
            print(f"  Upload response ({resp.status}): {json.dumps(result, indent=2)}")
            return result


async def delete_voice(host: str, port: int, voice_name: str) -> None:
    """Delete an uploaded voice."""
    url = f"http://{host}:{port}/v1/audio/voices/{voice_name}"
    async with aiohttp.ClientSession() as session:
        async with session.delete(url) as resp:
            if resp.status == 200:
                print(f"  Deleted voice '{voice_name}'")


async def run_round(
    host: str,
    port: int,
    num_prompts: int,
    create_payload_fn,
    label: str,
    num_warmups: int = 2,
    timeout_s: float = 120.0,
) -> BenchmarkResult:
    """Run one benchmark round and return results."""
    api_url = f"http://{host}:{port}/v1/audio/speech"
    connector = aiohttp.TCPConnector(limit=1, limit_per_host=1)
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    )

    try:
        # Warmup.
        if num_warmups > 0:
            print(f"  [{label}] Warming up ({num_warmups} requests)...")
            for i in range(num_warmups):
                payload = create_payload_fn(PROMPTS[i % len(PROMPTS)])
                r = await send_streaming_request(
                    session,
                    api_url,
                    payload,
                    SAMPLE_RATE,
                    SAMPLE_WIDTH,
                )
                status = "OK" if r.success else f"FAIL: {r.error[:80]}"
                print(f"    warmup {i + 1}: ttfp={r.ttfp * 1000:.0f}ms  {status}")

        # Benchmark.
        print(f"  [{label}] Running {num_prompts} requests (concurrency=1)...")
        results: list[RequestResult] = []
        start = time.perf_counter()
        for i in range(num_prompts):
            prompt = PROMPTS[i % len(PROMPTS)]
            payload = create_payload_fn(prompt)
            r = await send_streaming_request(
                session,
                api_url,
                payload,
                SAMPLE_RATE,
                SAMPLE_WIDTH,
            )
            results.append(r)
            tag = "HIT" if i > 0 and label == "uploaded_voice" else ""
            print(
                f"    req {i + 1:3d}: ttfp={r.ttfp * 1000:7.1f}ms  "
                f"e2e={r.e2e * 1000:7.1f}ms  "
                f"{'OK' if r.success else 'FAIL'} {tag}"
            )
        wall_time = time.perf_counter() - start
    finally:
        await session.close()

    bench = compute_stats(results, wall_time)
    bench.concurrency = 1
    bench.num_prompts = num_prompts
    bench.config_name = label
    return bench


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Fish Speech voice cache (inline vs uploaded)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--ref-audio", required=True, help="Path to reference audio file")
    parser.add_argument("--ref-text", required=True, help="Transcript of reference audio")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--voice-name", default="bench_voice")
    args = parser.parse_args()

    if not os.path.exists(args.ref_audio):
        print(f"Error: ref_audio not found: {args.ref_audio}")
        sys.exit(1)

    ref_audio_b64 = encode_audio_to_base64(args.ref_audio)
    print(f"Reference audio: {args.ref_audio} ({len(ref_audio_b64) // 1024}KB base64)")

    # ---- Round A: Inline ref_audio (no cache) ----
    print(f"\n{'=' * 60}")
    print("Round A: INLINE ref_audio (every request sends full audio)")
    print(f"{'=' * 60}")

    def make_inline_payload(prompt: str) -> dict:
        return {
            "input": prompt,
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
            "ref_audio": ref_audio_b64,
            "ref_text": args.ref_text,
            "max_new_tokens": 2048,
        }

    bench_inline = await run_round(
        args.host,
        args.port,
        args.num_prompts,
        make_inline_payload,
        "inline_ref_audio",
        num_warmups=args.num_warmups,
    )
    print_benchmark_results(bench_inline)

    # ---- Upload voice ----
    print(f"\n{'=' * 60}")
    print("Uploading voice for cache test...")
    print(f"{'=' * 60}")
    await delete_voice(args.host, args.port, args.voice_name)
    await upload_voice(
        args.host,
        args.port,
        args.ref_audio,
        args.ref_text,
        args.voice_name,
    )

    # ---- Round B: Uploaded voice (cache hits after 1st request) ----
    print(f"\n{'=' * 60}")
    print("Round B: UPLOADED VOICE (cache hits after 1st request)")
    print(f"{'=' * 60}")

    def make_uploaded_payload(prompt: str) -> dict:
        return {
            "input": prompt,
            "voice": args.voice_name,
            "stream": True,
            "response_format": "pcm",
            "ref_text": args.ref_text,
            "max_new_tokens": 2048,
        }

    bench_cached = await run_round(
        args.host,
        args.port,
        args.num_prompts,
        make_uploaded_payload,
        "uploaded_voice",
        num_warmups=args.num_warmups,
    )
    print_benchmark_results(bench_cached)

    # ---- Comparison ----
    print(f"\n{'=' * 60}")
    print("COMPARISON: Inline ref_audio vs Uploaded voice (cached)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'Inline':>12} {'Cached':>12} {'Speedup':>10}")
    print(f"{'-' * 64}")

    def fmt_speedup(inline_val: float, cached_val: float) -> str:
        if cached_val > 0 and inline_val > 0:
            ratio = inline_val / cached_val
            return f"{ratio:.2f}x"
        return "N/A"

    rows = [
        ("Mean TTFP (ms)", bench_inline.mean_ttfp_ms, bench_cached.mean_ttfp_ms),
        ("Median TTFP (ms)", bench_inline.median_ttfp_ms, bench_cached.median_ttfp_ms),
        ("P99 TTFP (ms)", bench_inline.p99_ttfp_ms, bench_cached.p99_ttfp_ms),
        ("Mean E2E (ms)", bench_inline.mean_e2e_ms, bench_cached.mean_e2e_ms),
        ("Median E2E (ms)", bench_inline.median_e2e_ms, bench_cached.median_e2e_ms),
        ("Mean RTF", bench_inline.mean_rtf, bench_cached.mean_rtf),
    ]
    for label, a, b in rows:
        print(f"{label:<30} {a:>12.1f} {b:>12.1f} {fmt_speedup(a, b):>10}")

    print("\nNote: Round B request #1 is a cache MISS (cold start).")
    print("      Requests #2+ are cache HITs (skip DAC encoding).")

    # Cleanup.
    await delete_voice(args.host, args.port, args.voice_name)


if __name__ == "__main__":
    asyncio.run(main())
