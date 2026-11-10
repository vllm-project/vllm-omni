# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare a perf result with the hardware baseline for the machine that ran it.

``--assert-baseline`` turns this on. A metric fails when it is more than 20%
worse than the baseline for the current hardware bucket. Throughput and QPS are
worse when they fall. Latency, memory, and real-time factor are worse when they
rise. A bucket for some other device is recorded and not compared.
"""

import math
from collections.abc import Mapping
from typing import Any

BASELINE_REGRESSION_TOLERANCE = 0.20

# Substrings, not a closed metric list. ``throughput_qps`` matches the higher
# group only; ``mean_audio_rtf`` and ``peak_memory_mb_mean`` match the lower group.
_LOWER_IS_BETTER_TOKENS = ("latency", "ttft", "tpot", "itl", "e2el", "ttfp", "rtf", "memory")
_HIGHER_IS_BETTER_TOKENS = ("throughput", "qps")


def metric_lower_is_better(name: str) -> bool:
    """Return whether a smaller measured value is the better one."""
    key = name.lower().replace("-", "_")
    lower = any(token in key for token in _LOWER_IS_BETTER_TOKENS)
    higher = any(token in key for token in _HIGHER_IS_BETTER_TOKENS)
    if lower and higher:
        raise ValueError(f"baseline metric {name!r} matches both a higher-is-better token and a lower-is-better token")
    if lower:
        return True
    if higher:
        return False
    raise ValueError(
        f"baseline metric {name!r} has no regression direction; "
        "use a latency/memory/rtf-style name or a throughput/qps-style name"
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _worse_ratio(measured: float, baseline: float, *, lower_is_better: bool) -> float:
    """Fraction by which ``measured`` is worse than ``baseline``.

    Zero baseline has no percentage scale: any move in the bad direction is
    treated as an unbounded regression, and a value that is not worse passes.
    """
    if baseline == 0:
        bad = measured > 0 if lower_is_better else measured < 0
        return math.inf if bad else 0.0
    if lower_is_better:
        return (measured - baseline) / baseline
    return (baseline - measured) / baseline


def _regressed(worse_ratio: float, tolerance: float) -> bool:
    # ``超过 20%`` is strict. Float noise around an exact 20% boundary still passes.
    if worse_ratio <= tolerance:
        return False
    return not math.isclose(worse_ratio, tolerance, rel_tol=1e-9, abs_tol=1e-12)


def assert_baseline_within_tolerance(
    measured: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    hardware: str | None,
    *,
    tolerance: float = BASELINE_REGRESSION_TOLERANCE,
) -> None:
    """Fail when any metric in the current hardware bucket is more than ``tolerance`` worse.

    ``baseline`` is the sweep-resolved hardware-nested map
    (``{"H100": {"request_throughput": 1.2}}``). Missing hardware, an empty
    baseline, or a machine with no bucket returns without comparing.
    """
    if not isinstance(baseline, dict) or not baseline:
        return
    if not isinstance(hardware, str) or not hardware or hardware not in baseline:
        return
    bucket = baseline[hardware]
    if not isinstance(bucket, dict) or not bucket:
        raise AssertionError(f"baseline[{hardware!r}] must be a non-empty metric map, got {bucket!r}")

    failures: list[str] = []
    for name, expected in bucket.items():
        if isinstance(expected, (list, tuple)):
            failures.append(f"{name}: baseline is still a sweep list; resolve the sweep step before comparing")
            continue
        if not _is_finite_number(expected):
            failures.append(f"{name}: baseline {expected!r} is not a finite number")
            continue
        if expected < 0:
            failures.append(f"{name}: negative baseline {expected} is not supported")
            continue
        actual = measured.get(name)
        if not _is_finite_number(actual):
            failures.append(f"{name}: measured value {actual!r} is not a finite number")
            continue
        try:
            lower_is_better = metric_lower_is_better(str(name))
        except ValueError as exc:
            failures.append(str(exc))
            continue
        ratio = _worse_ratio(float(actual), float(expected), lower_is_better=lower_is_better)
        if not _regressed(ratio, tolerance):
            continue
        if lower_is_better:
            limit = float(expected) * (1 + tolerance)
            direction = "above"
        else:
            limit = float(expected) * (1 - tolerance)
            direction = "below"
        failures.append(
            f"{name}: measured {actual} is more than {tolerance:.0%} {direction} "
            f"baseline {expected} on {hardware} (limit {limit})"
        )
    if failures:
        raise AssertionError("performance baseline regression:\n" + "\n".join(failures))
