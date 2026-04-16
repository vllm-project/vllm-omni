"""Benchmark VoxCPM via /v1/audio/speech.

Reports TTFP (time to first packet), E2E latency, and RTF (real-time factor).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

DEFAULT_MODEL = "OpenBMB/VoxCPM1.5"
DEFAULT_SAMPLE_RATE = 24000
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


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    *,
    model: str,
    prompt: str,
    ref_audio: str | None,
    ref_text: str | None,
    pbar: tqdm | None = None,
) -> RequestResult:
    payload: dict[str, object] = {
        "model": model,
        "input": prompt,
        "stream": True,
        "response_format": "pcm",
    }
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
    if ref_text is not None:
        payload["ref_text"] = ref_text

    result = RequestResult(prompt=prompt)
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


async def run_benchmark(
    *,
    host: str,
    port: int,
    model: str,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int,
    ref_audio: str | None,
    ref_text: str | None,
) -> BenchmarkResult:
    api_url = f"http://{host}:{port}/v1/audio/speech"
    connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency, keepalive_timeout=60)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if num_warmups > 0:
            print(f"  Warming up with {num_warmups} requests...")
            warmup_tasks = [
                send_tts_request(
                    session,
                    api_url,
                    model=model,
                    prompt=PROMPTS[i % len(PROMPTS)],
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                for i in range(num_warmups)
            ]
            await asyncio.gather(*warmup_tasks)
            print("  Warmup done.")

        request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
        semaphore = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

        async def limited_request(prompt: str) -> RequestResult:
            async with semaphore:
                return await send_tts_request(
                    session,
                    api_url,
                    model=model,
                    prompt=prompt,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    pbar=pbar,
                )

        started_at = time.perf_counter()
        results = await asyncio.gather(*[asyncio.create_task(limited_request(prompt)) for prompt in request_prompts])
        duration = time.perf_counter() - started_at
        pbar.close()

    succeeded = [result for result in results if result.success]
    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(succeeded),
        failed=len(results) - len(succeeded),
        duration_s=duration,
    )

    if not succeeded:
        return bench

    ttfps = np.array([result.ttfp * 1000 for result in succeeded], dtype=np.float64)
    e2es = np.array([result.e2e * 1000 for result in succeeded], dtype=np.float64)
    rtfs = np.array([result.rtf for result in succeeded], dtype=np.float64)
    audio_durations = np.array([result.audio_duration for result in succeeded], dtype=np.float64)

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
            "prompt": result.prompt,
            "ttfp_ms": result.ttfp * 1000,
            "e2e_ms": result.e2e * 1000,
            "rtf": result.rtf,
            "audio_duration_s": result.audio_duration,
        }
        for result in succeeded
    ]

    return bench


def print_summary(result: BenchmarkResult) -> None:
    width = 54
    print("")
    print("=" * width)
    print(f"{'VoxCPM Serving Benchmark':^{width}}")
    print("=" * width)
    print(f"concurrency         : {result.concurrency}")
    print(f"requests            : {result.completed}/{result.num_prompts} succeeded")
    print(f"wall time (s)       : {result.duration_s:.3f}")
    print(f"mean TTFP (ms)      : {result.mean_ttfp_ms:.2f}")
    print(f"p95 TTFP (ms)       : {result.p95_ttfp_ms:.2f}")
    print(f"mean E2E (ms)       : {result.mean_e2e_ms:.2f}")
    print(f"p95 E2E (ms)        : {result.p95_e2e_ms:.2f}")
    print(f"mean RTF            : {result.mean_rtf:.3f}")
    print(f"p95 RTF             : {result.p95_rtf:.3f}")
    print(f"request throughput  : {result.request_throughput:.2f} req/s")
    print("=" * width)


async def main_async(args) -> None:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []
    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            model=args.model,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            num_warmups=args.num_warmups,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
        print_summary(result)
        all_results.append(result)

    payload = {
        "model": args.model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "results": [asdict(result) for result in all_results],
    }
    result_path = result_dir / "bench_tts_serve.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VoxCPM via /v1/audio/speech")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8091, help="Server port")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts to send")
    parser.add_argument("--max-concurrency", type=int, nargs="+", default=[1], help="Concurrency levels to benchmark")
    parser.add_argument("--num-warmups", type=int, default=3, help="Warmup request count")
    parser.add_argument("--ref-audio", default=None, help="Reference audio URL or data URL for voice cloning")
    parser.add_argument("--ref-text", default=None, help="Reference audio transcript for voice cloning")
    parser.add_argument("--result-dir", default="results", help="Directory to save benchmark JSON")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
