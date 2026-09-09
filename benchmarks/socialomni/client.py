# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import aiohttp

T = TypeVar("T")


@dataclass
class RequestResult:
    request_id: str = ""
    text: str = ""
    is_success: bool = False
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


async def run_phase(
    items: Sequence[T],
    send: Callable[[aiohttp.ClientSession, T], Awaitable[RequestResult]],
    *,
    max_concurrency: int,
    timeout_s: float,
    warmup: int | None = None,
) -> tuple[list[RequestResult], float]:
    """Run ordered requests with bounded concurrency, excluding warmup time."""
    if max_concurrency < 1 or timeout_s <= 0 or (warmup is not None and warmup < 0):
        raise ValueError("concurrency and timeout must be positive; warmup must be non-negative")
    if not items:
        return [], 0.0
    semaphore = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s), trust_env=True) as session:

        async def bounded(item: T) -> RequestResult:
            async with semaphore:
                return await send(session, item)

        count = max_concurrency if warmup is None else warmup
        await asyncio.gather(*(bounded(items[i % len(items)]) for i in range(count)))
        started = time.perf_counter()
        results = await asyncio.gather(*(bounded(item) for item in items))
        return list(results), time.perf_counter() - started


def request_metrics(results: Sequence[RequestResult], wall_clock_s: float) -> dict[str, int | float]:
    successful = [result for result in results if result.is_success]
    return {
        "requests": len(results),
        "successful_requests": len(successful),
        "wall_clock_s": wall_clock_s,
        "mean_latency_s": sum(result.latency_s for result in successful) / len(successful) if successful else 0.0,
        "requests_per_second": len(successful) / wall_clock_s if wall_clock_s else 0.0,
        "prompt_tokens": sum(result.prompt_tokens for result in successful),
        "completion_tokens": sum(result.completion_tokens for result in successful),
    }
