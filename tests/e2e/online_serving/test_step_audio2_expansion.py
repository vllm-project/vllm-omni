# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E expansion tests for Step-Audio2 online serving (nightly CI).

Tests speech-to-speech translation via /v1/chat/completions with concurrent requests.
"""

import asyncio
import os
import time
from pathlib import Path

import aiohttp
import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

MODEL = "stepfun-ai/Step-Audio-2-mini"
STAGE_CONFIG = str(Path(__file__).parent / "stage_configs" / "step_audio2_ci.yaml")

TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG,
            env_dict={"VLLM_IMAGE_FETCH_TIMEOUT": "60"},
        ),
        id="step_audio2",
    )
]


SAMPLE_RATE = 16000


def _synthetic_audio_base64(duration_sec: int = 2) -> str:
    return generate_synthetic_audio(duration_sec, 1, SAMPLE_RATE)["base64"]


def create_s2st_request(audio_base64: str) -> dict:
    """Create S2ST request payload."""
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
    warmup: int = 1,
) -> dict:
    """Run concurrent benchmark with sustained concurrency."""
    semaphore = asyncio.Semaphore(num_concurrent)

    async def bounded_request(session: aiohttp.ClientSession):
        async with semaphore:
            return await send_request(session, url, payload)

    async with aiohttp.ClientSession() as session:
        for _ in range(warmup):
            try:
                await send_request(session, url, payload)
            except Exception:
                pass

        wall_start = time.perf_counter()
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.perf_counter() - wall_start

    latencies = []
    errors = 0
    for result in results:
        if isinstance(result, Exception):
            errors += 1
            continue
        latency, _, _ = result
        latencies.append(latency)

    if not latencies:
        return {"error": "All requests failed"}

    latencies_arr = np.array(latencies)
    successful_requests = len(latencies)

    return {
        "num_concurrent": num_concurrent,
        "successful_requests": successful_requests,
        "errors": errors,
        "wall_time": wall_time,
        "p50_latency": float(np.percentile(latencies_arr, 50)),
        "throughput_req_per_sec": successful_requests / wall_time,
    }


def _chat_completions_url(omni_server) -> str:
    return f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
@pytest.mark.asyncio
async def test_single_s2st_request(omni_server) -> None:
    """Test single speech-to-speech request via chat completions API."""
    url = _chat_completions_url(omni_server)
    payload = create_s2st_request(_synthetic_audio_base64(duration_sec=2))

    async with aiohttp.ClientSession() as session:
        latency, tokens, _ = await send_request(session, url, payload)

    assert latency > 0
    assert tokens >= 0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("num_concurrent", [1, 2])
@pytest.mark.asyncio
async def test_concurrent_s2st_throughput(omni_server, num_concurrent: int) -> None:
    """Test concurrent speech-to-speech throughput at different concurrency levels."""
    url = _chat_completions_url(omni_server)
    payload = create_s2st_request(_synthetic_audio_base64(duration_sec=5))
    num_requests = num_concurrent * 2

    stats = await benchmark_concurrent(url, payload, num_concurrent, num_requests)

    assert "error" not in stats
    assert stats["successful_requests"] > 0
    assert stats["errors"] == 0
