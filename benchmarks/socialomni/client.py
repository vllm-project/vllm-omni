# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import aiohttp

T = TypeVar("T")


class Progress:
    """Report phase progress to stderr without changing result output."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.completed = 0
        self.failed = 0
        self.started = self.last_report = time.perf_counter()
        print(f"{label}: 0/{total}", file=sys.stderr, flush=True)

    def update(self, success: bool = True) -> None:
        self.completed += 1
        self.failed += not success
        now = time.perf_counter()
        if self.completed == self.total or now - self.last_report >= 5:
            print(
                f"{self.label}: {self.completed}/{self.total}, {self.failed} failed, {now - self.started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            self.last_report = now


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
    description: str | None = None,
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
        if description and count:
            print(f"{description}: warming up {count} requests", file=sys.stderr, flush=True)
        await asyncio.gather(*(bounded(items[i % len(items)]) for i in range(count)))
        progress = Progress(description, len(items)) if description else None
        started = time.perf_counter()

        async def measured(item: T) -> RequestResult:
            result = await bounded(item)
            if progress:
                progress.update(result.is_success)
            return result

        results = await asyncio.gather(*(measured(item) for item in items))
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
