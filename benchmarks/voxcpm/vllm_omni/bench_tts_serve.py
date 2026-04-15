"""Benchmark VoxCPM via /v1/audio/speech.

Reports TTFP (time to first packet), E2E latency, and RTF (real-time factor).

By default, runs a voice cache comparison using the Qwen reference audio:
  A) Inline ref_audio: every request sends base64 audio (re-encoded each time)
  B) Uploaded voice:   upload once, then use voice name (cache hits after 1st)

Usage:
    # Voice cache comparison (uses default Qwen reference audio):
    python bench_tts_serve.py --port 8000

    # Plain TTS only (no voice cloning):
    python bench_tts_serve.py --port 8000 --no-voice-cache

    # Custom reference audio:
    python bench_tts_serve.py --port 8000 \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio."
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

DEFAULT_MODEL = "openbmb/VoxCPM2"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
PROMPTS = [
    "Hello, welcome to the VoxCPM speech benchmark.",
    "This is a short benchmark prompt for online text-to-speech generation.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please remember to bring your identification documents tomorrow morning.",
    "Learning a new language takes patience, practice, and curiosity.",
    "This benchmark reports TTFP and RTF for the VoxCPM online serving path.",
]


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0
    e2e: float = 0.0
    audio_bytes: int = 0
    audio_duration: float = 0.0
    rtf: float = 0.0
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    label: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    p95_rtf: float = 0.0
    total_audio_duration_s: float = 0.0
    request_throughput: float = 0.0
    per_request: list[dict[str, float | str]] = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = DEFAULT_SAMPLE_RATE, sample_width: int = 2) -> float:
    num_samples = num_bytes / sample_width
    return num_samples / sample_rate


def encode_audio_to_base64(audio_path: str) -> str:
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def download_ref_audio(url: str, dest_dir: str | None = None) -> str:
    """Download reference audio to a local temp file, returning the path."""
    if dest_dir is None:
        dest_dir = tempfile.gettempdir()
    filename = url.rsplit("/", 1)[-1]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"  Using cached ref audio: {dest}")
        return dest
    print(f"  Downloading ref audio: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Voice registration helpers
# ---------------------------------------------------------------------------

async def upload_voice(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    audio_path: str,
    ref_text: str | None,
    voice_name: str = "bench_voice",
) -> dict:
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
    async with session.post(url, data=data) as resp:
        result = await resp.json()
        print(f"  Upload response ({resp.status}): {json.dumps(result, indent=2)}")
        return result


async def delete_voice(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    voice_name: str,
) -> None:
    url = f"http://{host}:{port}/v1/audio/voices/{voice_name}"
    async with session.delete(url) as resp:
        if resp.status == 200:
            print(f"  Deleted voice '{voice_name}'")


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, object],
    *,
    pbar: tqdm | None = None,
) -> RequestResult:
    result = RequestResult(prompt=str(payload.get("input", "")))
    started_at = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                return result

            first_chunk = True
            total_bytes = 0
            async for chunk in response.content.iter_any():
                if not chunk:
                    continue
                if first_chunk:
                    result.ttfp = time.perf_counter() - started_at
                    first_chunk = False
                total_bytes += len(chunk)

            result.e2e = time.perf_counter() - started_at
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)
            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True
    except Exception as e:
        result.error = str(e)
        result.e2e = time.perf_counter() - started_at

    if pbar is not None:
        pbar.update(1)
    return result


# ---------------------------------------------------------------------------
# Run one benchmark round
# ---------------------------------------------------------------------------

async def run_benchmark(
    *,
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int,
    make_payload: callable,
    label: str,
) -> BenchmarkResult:
    api_url = f"http://{host}:{port}/v1/audio/speech"
    connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency, keepalive_timeout=60)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if num_warmups > 0:
            print(f"  [{label}] Warming up with {num_warmups} requests...")
            warmup_tasks = [
                send_tts_request(session, api_url, make_payload(PROMPTS[i % len(PROMPTS)]))
                for i in range(num_warmups)
            ]
            warmup_results = await asyncio.gather(*warmup_tasks)
            for i, r in enumerate(warmup_results):
                status = "OK" if r.success else f"FAIL: {r.error[:80]}"
                print(f"    warmup {i + 1}: ttfp={r.ttfp * 1000:.0f}ms  {status}")
            print("  Warmup done.")

        request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
        semaphore = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=num_prompts, desc=f"  [{label}] c={max_concurrency}")

        async def limited_request(prompt: str) -> RequestResult:
            async with semaphore:
                return await send_tts_request(session, api_url, make_payload(prompt), pbar=pbar)

        started_at = time.perf_counter()
        results = await asyncio.gather(*[asyncio.create_task(limited_request(p)) for p in request_prompts])
        duration = time.perf_counter() - started_at
        pbar.close()

    succeeded = [r for r in results if r.success]
    bench = BenchmarkResult(
        label=label,
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(succeeded),
        failed=len(results) - len(succeeded),
        duration_s=duration,
    )

    if not succeeded:
        return bench

    ttfps = np.array([r.ttfp * 1000 for r in succeeded], dtype=np.float64)
    e2es = np.array([r.e2e * 1000 for r in succeeded], dtype=np.float64)
    rtfs = np.array([r.rtf for r in succeeded], dtype=np.float64)
    audio_durations = np.array([r.audio_duration for r in succeeded], dtype=np.float64)

    bench.mean_ttfp_ms = float(np.mean(ttfps))
    bench.median_ttfp_ms = float(np.median(ttfps))
    bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
    bench.mean_e2e_ms = float(np.mean(e2es))
    bench.median_e2e_ms = float(np.median(e2es))
    bench.p95_e2e_ms = float(np.percentile(e2es, 95))
    bench.mean_rtf = float(np.mean(rtfs))
    bench.median_rtf = float(np.median(rtfs))
    bench.p95_rtf = float(np.percentile(rtfs, 95))
    bench.total_audio_duration_s = float(np.sum(audio_durations))
    bench.request_throughput = len(succeeded) / duration if duration > 0 else 0.0
    bench.per_request = [
        {
            "prompt": r.prompt,
            "ttfp_ms": r.ttfp * 1000,
            "e2e_ms": r.e2e * 1000,
            "rtf": r.rtf,
            "audio_duration_s": r.audio_duration,
        }
        for r in succeeded
    ]

    return bench


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary(result: BenchmarkResult) -> None:
    width = 58
    title = result.label or "VoxCPM Serving Benchmark"
    print("")
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    print(f"concurrency         : {result.concurrency}")
    print(f"requests            : {result.completed}/{result.num_prompts} succeeded")
    print(f"wall time (s)       : {result.duration_s:.3f}")
    print(f"mean TTFP (ms)      : {result.mean_ttfp_ms:.2f}")
    print(f"median TTFP (ms)    : {result.median_ttfp_ms:.2f}")
    print(f"p95 TTFP (ms)       : {result.p95_ttfp_ms:.2f}")
    print(f"mean E2E (ms)       : {result.mean_e2e_ms:.2f}")
    print(f"median E2E (ms)     : {result.median_e2e_ms:.2f}")
    print(f"p95 E2E (ms)        : {result.p95_e2e_ms:.2f}")
    print(f"mean RTF            : {result.mean_rtf:.3f}")
    print(f"median RTF          : {result.median_rtf:.3f}")
    print(f"p95 RTF             : {result.p95_rtf:.3f}")
    print(f"request throughput  : {result.request_throughput:.2f} req/s")
    print("=" * width)


def print_comparison(bench_a: BenchmarkResult, bench_b: BenchmarkResult) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"{'COMPARISON: Inline ref_audio vs Uploaded voice (cached)':^{width}}")
    print(f"{'=' * width}")
    print(f"{'Metric':<30} {'Inline':>12} {'Cached':>12} {'Speedup':>10}")
    print(f"{'-' * width}")

    def fmt_speedup(a: float, b: float) -> str:
        if a > 0 and b > 0:
            return f"{a / b:.2f}x"
        return "N/A"

    rows = [
        ("Mean TTFP (ms)", bench_a.mean_ttfp_ms, bench_b.mean_ttfp_ms),
        ("Median TTFP (ms)", bench_a.median_ttfp_ms, bench_b.median_ttfp_ms),
        ("P95 TTFP (ms)", bench_a.p95_ttfp_ms, bench_b.p95_ttfp_ms),
        ("Mean E2E (ms)", bench_a.mean_e2e_ms, bench_b.mean_e2e_ms),
        ("Median E2E (ms)", bench_a.median_e2e_ms, bench_b.median_e2e_ms),
        ("P95 E2E (ms)", bench_a.p95_e2e_ms, bench_b.p95_e2e_ms),
        ("Mean RTF", bench_a.mean_rtf, bench_b.mean_rtf),
        ("Throughput (req/s)", bench_a.request_throughput, bench_b.request_throughput),
    ]
    for label, a, b in rows:
        print(f"{label:<30} {a:>12.1f} {b:>12.1f} {fmt_speedup(a, b):>10}")

    print(f"\nNote: Uploaded voice request #1 is a cache MISS (cold start).")
    print(f"      Requests #2+ are cache HITs (skip audio re-encoding).")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []

    # Resolve reference audio: explicit path, default URL, or none
    ref_audio_path = args.ref_audio
    ref_text = args.ref_text
    if ref_audio_path is None and not args.no_voice_cache:
        ref_audio_path = download_ref_audio(DEFAULT_REF_AUDIO_URL)
        if ref_text is None:
            ref_text = DEFAULT_REF_TEXT

    if ref_audio_path is not None and not os.path.exists(ref_audio_path):
        print(f"Error: ref audio file not found: {ref_audio_path}")
        return

    if ref_audio_path is None:
        # Plain TTS benchmark (no voice cloning)
        def make_plain_payload(prompt: str) -> dict:
            return {
                "model": args.model,
                "input": prompt,
                "stream": True,
                "response_format": "pcm",
            }

        for concurrency in args.max_concurrency:
            result = await run_benchmark(
                host=args.host,
                port=args.port,
                num_prompts=args.num_prompts,
                max_concurrency=concurrency,
                num_warmups=args.num_warmups,
                make_payload=make_plain_payload,
                label=f"plain_tts (c={concurrency})",
            )
            print_summary(result)
            all_results.append(result)
    else:
        ref_audio_b64 = encode_audio_to_base64(ref_audio_path)
        print(f"Reference audio: {ref_audio_path} ({len(ref_audio_b64) // 1024}KB base64)")

        for concurrency in args.max_concurrency:
            # ---- Round A: Inline ref_audio (no cache) ----
            print(f"\n{'=' * 60}")
            print(f"Round A: INLINE ref_audio (c={concurrency})")
            print(f"{'=' * 60}")

            def make_inline_payload(prompt: str) -> dict:
                return {
                    "model": args.model,
                    "input": prompt,
                    "stream": True,
                    "response_format": "pcm",
                    "ref_audio": ref_audio_b64,
                    "ref_text": ref_text,
                }

            bench_inline = await run_benchmark(
                host=args.host,
                port=args.port,
                num_prompts=args.num_prompts,
                max_concurrency=concurrency,
                num_warmups=args.num_warmups,
                make_payload=make_inline_payload,
                label=f"inline_ref_audio (c={concurrency})",
            )
            print_summary(bench_inline)
            all_results.append(bench_inline)

            # ---- Upload voice ----
            print(f"\n{'=' * 60}")
            print("Uploading voice for cache test...")
            print(f"{'=' * 60}")

            connector = aiohttp.TCPConnector(limit=1)
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60)) as session:
                await delete_voice(session, args.host, args.port, args.voice_name)
                await upload_voice(session, args.host, args.port, ref_audio_path, ref_text, args.voice_name)

            # ---- Round B: Uploaded voice (cache hits after 1st request) ----
            print(f"\n{'=' * 60}")
            print(f"Round B: UPLOADED VOICE (c={concurrency})")
            print(f"{'=' * 60}")

            def make_uploaded_payload(prompt: str) -> dict:
                return {
                    "model": args.model,
                    "input": prompt,
                    "voice": args.voice_name,
                    "stream": True,
                    "response_format": "pcm",
                }

            bench_cached = await run_benchmark(
                host=args.host,
                port=args.port,
                num_prompts=args.num_prompts,
                max_concurrency=concurrency,
                num_warmups=args.num_warmups,
                make_payload=make_uploaded_payload,
                label=f"uploaded_voice (c={concurrency})",
            )
            print_summary(bench_cached)
            all_results.append(bench_cached)

            print_comparison(bench_inline, bench_cached)

            # Cleanup
            connector = aiohttp.TCPConnector(limit=1)
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60)) as session:
                await delete_voice(session, args.host, args.port, args.voice_name)

    payload = {
        "model": args.model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "results": [asdict(r) for r in all_results],
    }
    result_path = result_dir / "bench_tts_serve.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VoxCPM via /v1/audio/speech (with voice cache comparison)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8091, help="Server port")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts per round")
    parser.add_argument("--max-concurrency", type=int, nargs="+", default=[1], help="Concurrency levels to benchmark")
    parser.add_argument("--num-warmups", type=int, default=3, help="Warmup request count per round")
    parser.add_argument(
        "--ref-audio", default=None,
        help="Path to reference audio file (default: downloads Qwen clone_2.wav)",
    )
    parser.add_argument(
        "--ref-text", default=None,
        help="Transcript of the reference audio (default: Qwen clone_2 transcript)",
    )
    parser.add_argument("--voice-name", default="bench_voice", help="Name for the uploaded voice")
    parser.add_argument(
        "--no-voice-cache", action="store_true",
        help="Skip voice cache comparison; run plain TTS only",
    )
    parser.add_argument("--result-dir", default="results", help="Directory to save benchmark JSON")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
