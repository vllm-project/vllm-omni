# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead, parseable timings for startup critical paths."""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol


class _Logger(Protocol):
    def info(self, message: str, *args: object) -> None: ...


def log_startup_duration(
    logger: _Logger,
    phase: str,
    duration_s: float,
    status: str = "ok",
    **labels: object,
) -> None:
    sorted_labels = sorted(labels.items())
    suffix = "".join(f" {key}=%s" for key, _ in sorted_labels)
    logger.info(
        f"[StartupTiming] phase=%s duration_s=%.6f status=%s{suffix}",
        phase,
        duration_s,
        status,
        *(value for _, value in sorted_labels),
    )


@contextmanager
def startup_span(logger: _Logger, phase: str, **labels: object) -> Iterator[None]:
    """Log one startup phase without changing its exception semantics."""
    started = time.perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        status = "ok" if completed else "error"
        log_startup_duration(logger, phase, time.perf_counter() - started, status, **labels)


def process_age_seconds() -> float | None:
    """Return this process's age from Linux monotonic clocks when available."""
    try:
        stat = open("/proc/self/stat").read()  # noqa: SIM115 - one tiny procfs read
        fields_after_comm = stat[stat.rfind(")") + 2 :].split()
        start_ticks = int(fields_after_comm[19])
        ticks_per_second = os.sysconf("SC_CLK_TCK")
        return max(0.0, time.clock_gettime(time.CLOCK_BOOTTIME) - start_ticks / ticks_per_second)
    except (AttributeError, IndexError, OSError, TypeError, ValueError):
        return None


def log_process_checkpoint(logger: _Logger, phase: str, **labels: object) -> None:
    """Log elapsed process lifetime at a startup checkpoint on Linux."""
    duration_s = process_age_seconds()
    if duration_s is not None:
        log_startup_duration(logger, phase, duration_s, checkpoint=True, **labels)
