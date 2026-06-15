# SPDX-License-Identifier: Apache-2.0
"""
E2E tests for Step-Audio2 online serving with concurrent requests.

Tests:
1. Single request via API
2. Concurrent throughput (1/2/4/8 concurrent requests)
3. Latency percentiles (p50/p95/p99)
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import pytest
from vllm.utils.network_utils import get_open_port

MODEL = "stepfun-ai/Step-Audio-2-mini"
STAGE_CONFIG = str(Path(__file__).parent / "stage_configs" / "step_audio2_ci.yaml")

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        print(f"Server ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def create_dummy_audio_base64(duration_sec: float = 5.0, sample_rate: int = 16000) -> str:
    """Create dummy audio as base64."""
    import base64
    import io
    import wave

    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_s2st_request(audio_base64: str) -> dict:
    """Create S2ST request payload."""
    # Use audio_url format with base64 data URL
    audio_url = f"data:audio/wav;base64,{audio_base64}"
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "请仔细聆听这段语音，然后复述其内容。"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                    },
                    {
                        "type": "text",
                        "text": "<tts_start>",
                    },
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }


async def send_request(session: aiohttp.ClientSession, url: str, payload: dict) -> tuple[float, int, str]:
    """Send single request and return (latency, num_tokens, response_text)."""
    start = time.perf_counter()
    async with session.post(url, json=payload) as response:
        result = await response.json()
    latency = time.perf_counter() - start

    if "choices" in result and len(result["choices"]) > 0:
        text = result["choices"][0].get("message", {}).get("content", "")
        tokens = result.get("usage", {}).get("completion_tokens", len(text.split()))
    else:
        text = str(result)
        tokens = 0

    return latency, tokens, text


async def benchmark_concurrent(
    url: str,
    payload: dict,
    num_concurrent: int,
    num_requests: int,
    warmup: int = 2,
) -> dict:
    """Run concurrent benchmark with true sustained concurrency.

    Uses semaphore to maintain exactly num_concurrent requests in flight at all times,
    rather than batch-and-wait approach. Measures real wall clock time.
    """
    semaphore = asyncio.Semaphore(num_concurrent)

    async def bounded_request(session: aiohttp.ClientSession):
        async with semaphore:
            return await send_request(session, url, payload)

    async with aiohttp.ClientSession() as session:
        if warmup > 0:
            print(f"    Warming up ({warmup} requests)...", end=" ", flush=True)
            for _ in range(warmup):
                try:
                    await send_request(session, url, payload)
                except Exception as e:
                    print(f"Warmup failed: {e}")
            print("done")

        print(f"    Running {num_requests} requests with concurrency={num_concurrent}...", end=" ", flush=True)
        wall_start = time.perf_counter()

        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        wall_time = time.perf_counter() - wall_start
        print(f"done ({wall_time:.2f}s)")

    all_latencies = []
    all_tokens = []
    errors = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"    Request failed: {result}")
            errors += 1
            continue
        latency, tokens, _ = result
        all_latencies.append(latency)
        all_tokens.append(tokens)

    if not all_latencies:
        return {"error": "All requests failed"}

    latencies = np.array(all_latencies)
    total_tokens = sum(all_tokens)
    successful_requests = len(all_latencies)

    return {
        "num_concurrent": num_concurrent,
        "num_requests": num_requests,
        "successful_requests": successful_requests,
        "errors": errors,
        "wall_time": wall_time,
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "p99_latency": float(np.percentile(latencies, 99)),
        "mean_latency": float(np.mean(latencies)),
        "min_latency": float(np.min(latencies)),
        "max_latency": float(np.max(latencies)),
        "throughput_req_per_sec": successful_requests / wall_time,
        "throughput_tokens_per_sec": total_tokens / wall_time,
    }


class TestStepAudio2OnlineServing:
    """Online serving tests for Step-Audio2."""

    @pytest.fixture(scope="class")
    def server(self):
        """Start vLLM-Omni server."""
        with OmniServer(MODEL, ["--stage-configs-path", STAGE_CONFIG]) as omni_server:
            yield f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_single_request(self, server):
        """Test single request works."""
        audio_base64 = create_dummy_audio_base64(duration_sec=2.0)
        payload = create_s2st_request(audio_base64)

        async with aiohttp.ClientSession() as session:
            latency, tokens, text = await send_request(session, server, payload)

        assert latency > 0
        assert tokens >= 0
        print(f"Single request: latency={latency:.2f}s, tokens={tokens}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_concurrent", [1, 2, 4])
    async def test_concurrent_throughput(self, server, num_concurrent):
        """Test concurrent throughput at different concurrency levels."""
        audio_base64 = create_dummy_audio_base64(duration_sec=5.0)
        payload = create_s2st_request(audio_base64)

        num_requests = num_concurrent * 3

        stats = await benchmark_concurrent(server, payload, num_concurrent, num_requests)

        assert "error" not in stats
        print(f"\nConcurrency={num_concurrent}:")
        print(f"  p50={stats['p50_latency']:.2f}s, p95={stats['p95_latency']:.2f}s")
        print(f"  throughput={stats['throughput_req_per_sec']:.2f} req/s")


async def run_throughput_benchmark(
    server_url: str,
    audio_duration: float = 5.0,
    concurrency_levels: list[int] = [1, 2, 4, 8],
    requests_per_level: int = 10,
    warmup: int = 2,
):
    """Run full throughput benchmark."""
    audio_base64 = create_dummy_audio_base64(duration_sec=audio_duration)
    payload = create_s2st_request(audio_base64)

    print(f"{'=' * 70}")
    print("Step-Audio2 Online Serving Throughput Benchmark")
    print(f"{'=' * 70}")
    print(f"Server: {server_url}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Requests per level: {requests_per_level}")
    print(f"Warmup requests: {warmup}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"{'=' * 70}")

    results = []
    for num_concurrent in concurrency_levels:
        print(f"\n[Concurrency={num_concurrent}]")
        stats = await benchmark_concurrent(server_url, payload, num_concurrent, requests_per_level, warmup)
        results.append(stats)

        if "error" not in stats:
            print(
                f"    Result: p50={stats['p50_latency']:.2f}s, "
                f"p95={stats['p95_latency']:.2f}s, "
                f"throughput={stats['throughput_req_per_sec']:.2f} req/s, "
                f"{stats['throughput_tokens_per_sec']:.1f} tok/s"
            )

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Conc':<6} {'Reqs':<6} {'Err':<5} {'Wall(s)':<9} {'p50':<8} {'p95':<8} {'p99':<8} {'Req/s':<8} {'Tok/s':<8}"
    )
    print("-" * 80)

    for stats in results:
        if "error" in stats:
            print(f"{stats.get('num_concurrent', '?'):<6} ERROR: {stats['error']}")
            continue
        print(
            f"{stats['num_concurrent']:<6} "
            f"{stats['successful_requests']:<6} "
            f"{stats['errors']:<5} "
            f"{stats['wall_time']:<9.2f} "
            f"{stats['p50_latency']:<8.2f} "
            f"{stats['p95_latency']:<8.2f} "
            f"{stats['p99_latency']:<8.2f} "
            f"{stats['throughput_req_per_sec']:<8.2f} "
            f"{stats['throughput_tokens_per_sec']:<8.1f}"
        )

    print("=" * 80)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step-Audio2 Online Serving Benchmark")
    parser.add_argument(
        "--server-url", type=str, default="http://localhost:8000/v1/chat/completions", help="vLLM server URL"
    )
    parser.add_argument("--audio-duration", type=float, default=5.0, help="Input audio duration in seconds")
    parser.add_argument("--concurrency", type=str, default="1,2,4,8", help="Comma-separated concurrency levels to test")
    parser.add_argument("--requests-per-level", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup requests before each benchmark")

    args = parser.parse_args()
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]

    asyncio.run(
        run_throughput_benchmark(
            args.server_url,
            args.audio_duration,
            concurrency_levels,
            args.requests_per_level,
            args.warmup,
        )
    )
