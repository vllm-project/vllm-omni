# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for diffusion serving benchmarks."""

import asyncio
import random
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from backends import RequestFuncInput, RequestFuncOutput


async def iter_requests(
    input: list[RequestFuncInput] | int,
    request_rate: float,
) -> AsyncGenerator[RequestFuncInput | int, None]:
    """Yield requests (or indices) using a Poisson arrival process.

    When *input* is a list, yields items from that list (used by
    ``diffusion_benchmark_serving``).  When *input* is an int, yields
    ``range(input)`` indices (used by ``benchmark_glm_image``).

    If *request_rate* is ``inf``, all items are yielded immediately.
    Otherwise inter-arrival times follow an exponential distribution.
    """
    if request_rate != float("inf"):
        if request_rate <= 0:
            raise ValueError(f"request_rate must be positive or inf, got {request_rate}.")

    items: range | list = range(input) if isinstance(input, int) else input
    for i, item in enumerate(items):
        if request_rate != float("inf") and i > 0:
            interval_s = random.expovariate(request_rate)
            await asyncio.sleep(interval_s)
        yield item


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    total_duration: float,
    requests_list: list[RequestFuncInput] | None = None,
    slo_enabled: bool = False,
    slo_scale: float = 3.0,
) -> dict[str, Any]:
    """Aggregate latency, throughput, memory, and stage-duration metrics."""
    success_outputs = [o for o in outputs if o.success]
    error_outputs = [o for o in outputs if not o.success]

    num_success = len(success_outputs)
    latencies = [o.latency for o in success_outputs]
    peak_memories = [o.peak_memory_mb for o in success_outputs if o.peak_memory_mb > 0]

    stage_duration_lists: dict[str, list[float]] = {}
    for o in success_outputs:
        for stage, duration in (o.stage_durations or {}).items():
            stage_duration_lists.setdefault(stage, []).append(duration)
    stage_durations_mean = {s: float(np.mean(v)) for s, v in stage_duration_lists.items()}
    stage_durations_p50 = {s: float(np.percentile(v, 50)) for s, v in stage_duration_lists.items()}
    stage_durations_p99 = {s: float(np.percentile(v, 99)) for s, v in stage_duration_lists.items()}

    metrics: dict[str, Any] = {
        "duration": total_duration,
        "completed_requests": num_success,
        "failed_requests": len(error_outputs),
        "throughput_qps": num_success / total_duration if total_duration > 0 else 0,
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_median": np.median(latencies) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "latency_p95": np.percentile(latencies, 95) if latencies else 0,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
        "peak_memory_mb_max": max(peak_memories) if peak_memories else 0,
        "peak_memory_mb_mean": np.mean(peak_memories) if peak_memories else 0,
        "peak_memory_mb_median": np.median(peak_memories) if peak_memories else 0,
        "stage_durations_mean": stage_durations_mean,
        "stage_durations_p50": stage_durations_p50,
        "stage_durations_p99": stage_durations_p99,
    }

    if slo_enabled and requests_list is not None:
        slo_defined_total = 0
        slo_met_success = 0

        for req, out in zip(requests_list, outputs):
            if req.slo_ms is None:
                continue
            slo_defined_total += 1
            if out.slo_achieved is None:
                continue
            if out.slo_achieved:
                slo_met_success += 1

        slo_attain_all = (slo_met_success / slo_defined_total) if slo_defined_total > 0 else 0.0

        metrics.update(
            {
                "slo_attainment_rate": slo_attain_all,
                "slo_met_success": slo_met_success,
                "slo_scale": slo_scale,
            }
        )

    return metrics
