"""Benchmark comparing async_chunk on vs off for Qwen3-TTS.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF across
concurrency levels for both async_chunk modes. Saves results as JSON.

Usage:
    # Run against a server already serving with a given config:
    python bench_async_chunk.py \
        --host 127.0.0.1 --port 8000 \
        --config-name async_chunk_on \
        --num-prompts 50 \
        --max-concurrency 1 10 \
        --result-dir results/
"""

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

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
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
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = 24000, sample_width: int = 2) -> float:
    return num_bytes / sample_width / sample_rate


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    voice: str = "vivian",
    language: str = "English",
    stream: bool = True,
    pbar: tqdm | None = None,
) -> RequestResult:
    payload = {
        "input": prompt,
        "voice": voice,
        "language": language,
        "stream": stream,
        "response_format": "pcm",
    }

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                return result

            first_chunk = True
            total_bytes = 0

            async for chunk in response.content.iter_any():
                if first_chunk and len(chunk) > 0:
                    result.ttfp = time.perf_counter() - st
                    first_chunk = False
                total_bytes += len(chunk)

            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)
            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int = 3,
    voice: str = "vivian",
    language: str = "English",
    stream: bool = True,
) -> BenchmarkResult:
    api_url = f"http://{host}:{port}/v1/audio/speech"

    connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency, keepalive_timeout=60)
    session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=600))

    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = [
            send_tts_request(session, api_url, PROMPTS[i % len(PROMPTS)], voice, language, stream)
            for i in range(num_warmups)
        ]
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited_request(prompt):
        async with semaphore:
            return await send_tts_request(session, api_url, prompt, voice, language, stream, pbar)

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    await session.close()

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfps = [r.ttfp * 1000 for r in successful]
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful]
        audio_durs = [r.audio_duration for r in successful]

        bench.mean_ttfp_ms = float(np.mean(ttfps))
        bench.median_ttfp_ms = float(np.median(ttfps))
        bench.std_ttfp_ms = float(np.std(ttfps))
        bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
        bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
        bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))

        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.std_rtf = float(np.std(rtfs))

        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

        bench.per_request = [
            {
                "ttfp_ms": r.ttfp * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "prompt": r.prompt,
            }
            for r in successful
        ]

    print(f"\n{'=' * 60}")
    print(f"  Concurrency: {max_concurrency}  |  Completed: {bench.completed}  |  Failed: {bench.failed}")
    print(f"  Duration: {duration:.2f}s  |  Throughput: {bench.request_throughput:.2f} req/s")
    print(
        f"  TTFP (ms):  mean={bench.mean_ttfp_ms:.1f}  median={bench.median_ttfp_ms:.1f}"
        f"  p90={bench.p90_ttfp_ms:.1f}  p99={bench.p99_ttfp_ms:.1f}"
    )
    print(
        f"  E2E (ms):   mean={bench.mean_e2e_ms:.1f}  median={bench.median_e2e_ms:.1f}"
        f"  p90={bench.p90_e2e_ms:.1f}  p99={bench.p99_e2e_ms:.1f}"
    )
    print(f"  RTF:        mean={bench.mean_rtf:.3f}  median={bench.median_rtf:.3f}")
    print(f"  Throughput: {bench.audio_throughput:.2f} audio-sec/wall-sec")
    print(f"{'=' * 60}\n")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def main(args):
    all_results = []

    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            num_warmups=args.num_warmups,
            voice=args.voice,
            language=args.language,
            stream=args.stream,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS async_chunk benchmark client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--voice", type=str, default="vivian")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--config-name", type=str, default="async_chunk_on")
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
