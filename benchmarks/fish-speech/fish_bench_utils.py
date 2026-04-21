"""Shared benchmark infrastructure for Fish Speech serving benchmarks.

Provides common dataclasses, metrics computation, streaming HTTP client,
and result formatting used by model-specific benchmark scripts.

Model-specific scripts supply a ``create_payload_fn(prompt) -> dict``
callback and audio parameters; everything else is handled here.
"""

import asyncio
import base64
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# ---------------------------------------------------------------------------
# Shared test prompts (varying length for realistic workload)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0  # Total audio bytes received
    audio_duration: float = 0.0  # Audio duration in seconds
    rtf: float = 0.0  # Real-time factor = e2e / audio_duration
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
    # TTFP stats (ms)
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0  # audio_duration / wall_time
    request_throughput: float = 0.0  # requests / second
    # Per-request details
    per_request: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def pcm_bytes_to_duration(
    num_bytes: int,
    sample_rate: int = 24000,
    sample_width: int = 2,
) -> float:
    """Convert raw PCM byte count to duration in seconds."""
    return num_bytes / sample_width / sample_rate


def _is_sse_response(response: aiohttp.ClientResponse) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    return "text/event-stream" in content_type


async def _read_raw_audio_stream(
    response: aiohttp.ClientResponse,
    *,
    start_time: float,
) -> tuple[int, float]:
    first_audio_at = 0.0
    total_bytes = 0

    async for chunk in response.content.iter_any():
        if chunk and first_audio_at <= 0:
            first_audio_at = time.perf_counter() - start_time
        total_bytes += len(chunk)

    return total_bytes, first_audio_at


def _extract_sse_payload(raw_event: bytes) -> bytes | None:
    data_lines: list[bytes] = []
    for raw_line in raw_event.splitlines():
        line = raw_line.rstrip(b"\r")
        if line.startswith(b"data: "):
            data_lines.append(line[6:])
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())

    if not data_lines:
        return None
    return b"\n".join(data_lines).strip()


async def _read_sse_audio_stream(
    response: aiohttp.ClientResponse,
    *,
    start_time: float,
) -> tuple[int, float]:
    """Decode SSE events and count raw audio bytes from base64 payloads."""
    first_audio_at = 0.0
    total_bytes = 0
    pending = b""

    async for chunk in response.content.iter_any():
        if not chunk:
            continue
        pending += chunk
        pending = pending.replace(b"\r\n", b"\n")

        while b"\n\n" in pending:
            raw_event, pending = pending.split(b"\n\n", 1)
            payload_bytes = _extract_sse_payload(raw_event)
            if payload_bytes is None:
                continue
            if payload_bytes == b"[DONE]":
                return total_bytes, first_audio_at

            try:
                payload = json.loads(payload_bytes)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid SSE JSON payload: {exc}") from exc

            audio = payload.get("audio")
            if not isinstance(audio, dict):
                continue

            audio_b64 = audio.get("data")
            if not audio_b64:
                continue

            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception as exc:
                raise ValueError(f"Invalid base64 audio chunk: {exc}") from exc

            if audio_bytes and first_audio_at <= 0:
                first_audio_at = time.perf_counter() - start_time
            total_bytes += len(audio_bytes)

    return total_bytes, first_audio_at


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_stats(
    results: list[RequestResult],
    wall_time: float,
) -> BenchmarkResult:
    """Compute aggregate statistics from per-request results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        completed=len(successful),
        failed=len(failed),
        duration_s=wall_time,
    )

    if not successful:
        return bench

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
    bench.p99_rtf = float(np.percentile(rtfs, 99))

    bench.mean_audio_duration_s = float(np.mean(audio_durs))
    bench.total_audio_duration_s = float(np.sum(audio_durs))
    bench.audio_throughput = bench.total_audio_duration_s / wall_time
    bench.request_throughput = len(successful) / wall_time

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

    return bench


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_benchmark_results(bench: BenchmarkResult) -> None:
    """Print benchmark results in standardized format."""
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{bench.concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{bench.duration_s:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")


def save_results(
    all_results: list[dict],
    result_dir: str,
    config_name: str,
) -> Path:
    """Save benchmark results as JSON and return the file path."""
    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = out / f"bench_{config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")
    return result_file


# ---------------------------------------------------------------------------
# Streaming HTTP client
# ---------------------------------------------------------------------------
async def send_streaming_request(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict,
    sample_rate: int,
    sample_width: int,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming TTS request and measure latency metrics."""
    result = RequestResult(prompt=payload.get("input", ""))
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
            else:
                if _is_sse_response(response):
                    total_bytes, result.ttfp = await _read_sse_audio_stream(
                        response,
                        start_time=st,
                    )
                else:
                    total_bytes, result.ttfp = await _read_raw_audio_stream(
                        response,
                        start_time=st,
                    )

                result.e2e = time.perf_counter() - st
                result.audio_bytes = total_bytes
                result.audio_duration = pcm_bytes_to_duration(total_bytes, sample_rate, sample_width)

                if total_bytes <= 0 or result.ttfp <= 0:
                    result.error = "HTTP 200 but no audio bytes were received"
                else:
                    if result.audio_duration > 0:
                        result.rtf = result.e2e / result.audio_duration
                    result.success = True

    except Exception as e:
        result.error = str(e)
        result.e2e = time.perf_counter() - st

    finally:
        if pbar:
            pbar.update(1)
    return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    create_payload_fn: Callable[[str], dict],
    sample_rate: int,
    sample_width: int = 2,
    num_warmups: int = 3,
    request_timeout_s: float = 120.0,
) -> BenchmarkResult:
    """Run a TTS streaming benchmark at a given concurrency level.

    Args:
        create_payload_fn: Model-specific function that takes a prompt string
            and returns the request JSON payload dict.
        sample_rate: PCM sample rate for audio duration calculation.
        sample_width: PCM sample width in bytes (default 2 for 16-bit).
    """
    api_url = f"http://{host}:{port}/v1/audio/speech"

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=request_timeout_s,
            connect=min(10.0, request_timeout_s),
            sock_connect=min(10.0, request_timeout_s),
            sock_read=request_timeout_s,
        ),
    )

    try:
        # Warmup
        if num_warmups > 0:
            print(f"  Warming up with {num_warmups} requests...")
            warmup_tasks = [
                send_streaming_request(
                    session,
                    api_url,
                    create_payload_fn(PROMPTS[i % len(PROMPTS)]),
                    sample_rate,
                    sample_width,
                )
                for i in range(num_warmups)
            ]
            warmup_results = await asyncio.gather(*warmup_tasks)
            warmup_ok = sum(1 for r in warmup_results if r.success)
            if warmup_ok == 0:
                print("  WARNING: All warmup requests failed!")
                for r in warmup_results:
                    if r.error:
                        print(f"    {r.error[:200]}")
            print(f"  Warmup done ({warmup_ok}/{num_warmups} succeeded).")

        # Build request list
        request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

        # Run
        print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
        semaphore = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

        async def limited_request(prompt: str) -> RequestResult:
            async with semaphore:
                return await send_streaming_request(
                    session,
                    api_url,
                    create_payload_fn(prompt),
                    sample_rate,
                    sample_width,
                    pbar,
                )

        start_time = time.perf_counter()
        tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
        results: list[RequestResult] = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_time
        pbar.close()

    finally:
        await session.close()

    # Compute stats
    bench = compute_stats(results, wall_time)
    bench.concurrency = max_concurrency
    bench.num_prompts = num_prompts

    print_benchmark_results(bench)

    # Print sample errors
    failed = [r for r in results if not r.success]
    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def run_benchmark_sweep(
    host: str,
    port: int,
    num_prompts: int,
    concurrency_levels: list[int],
    create_payload_fn: Callable[[str], dict],
    sample_rate: int,
    sample_width: int = 2,
    num_warmups: int = 3,
    request_timeout_s: float = 120.0,
    config_name: str = "benchmark",
    result_dir: str = "results",
) -> list[dict]:
    """Run benchmarks across multiple concurrency levels and save results."""
    all_results = []

    for concurrency in concurrency_levels:
        result = await run_benchmark(
            host=host,
            port=port,
            num_prompts=num_prompts,
            max_concurrency=concurrency,
            create_payload_fn=create_payload_fn,
            sample_rate=sample_rate,
            sample_width=sample_width,
            num_warmups=num_warmups,
            request_timeout_s=request_timeout_s,
        )
        result.config_name = config_name
        all_results.append(asdict(result))

    save_results(all_results, result_dir, config_name)
    return all_results
