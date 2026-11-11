# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request coverage checks independent of model accuracy and latency averages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class BenchmarkOutput(Protocol):
    success: bool


def positive_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def request_coverage(outputs: Sequence[BenchmarkOutput]) -> dict[str, int]:
    """Keep counts even when the CLI drops per-request details from its artifact."""
    return {
        "request_count": len(outputs),
        "successful_audio_requests": sum(
            o.success
            and positive_finite(getattr(o, "audio_duration", None))
            and positive_finite(getattr(o, "audio_ttfp", None))
            and positive_finite(getattr(o, "audio_rtf", None))
            for o in outputs
        ),
    }


def coverage_errors(result: dict, expected: int, *, require_audio: bool = False) -> list[str]:
    """Missing fields fail closed; an average never establishes full coverage."""
    errors = []
    for key in ("request_count", "completed"):
        if result.get(key) != expected:
            errors.append(f"{key}={result.get(key)!r}, expected {expected}")
    for key in ("failed", "timed_out_requests", "cancelled_after_timeout_requests"):
        if result.get(key, 0) != 0:
            errors.append(f"{key}={result[key]!r}, expected 0")
    if require_audio and result.get("successful_audio_requests") != expected:
        errors.append(f"successful_audio_requests={result.get('successful_audio_requests')!r}, expected {expected}")
    return errors
