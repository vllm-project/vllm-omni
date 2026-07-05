# this module will be activated if MOSS_TTS_TIMING=1 to record per-phase timing in Stage 0
from __future__ import annotations

import atexit
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

_TIMING_ENABLED: bool = os.environ.get("MOSS_TTS_TIMING", "0") == "1"

class _Stage0Timing:
    def __init__(self) -> None:
        self._gpu_events: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self._cpu_ms: dict[str, list[float]] = defaultdict(list)
        self._dumped: bool = False

    @contextmanager
    def gpu(self, name: str):
        if not _TIMING_ENABLED:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._gpu_events[name].append((start, end))

    @contextmanager
    def cpu(self, name: str):
        if not _TIMING_ENABLED:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._cpu_ms[name].append((time.perf_counter_ns() - t0) / 1e6)

    def reset(self) -> None:
        if not _TIMING_ENABLED:
            return
        if self._gpu_events:
            torch.cuda.synchronize()
        self._gpu_events.clear()
        self._cpu_ms.clear()
        self._dumped = False

    def dump(self) -> None:
        if not _TIMING_ENABLED or self._dumped:
            return
        self._dumped = True

        gpu_ms: dict[str, list[float]] = {}
        if self._gpu_events:
            torch.cuda.synchronize()
            for name, pairs in self._gpu_events.items():
                gpu_ms[name] = [s.elapsed_time(e) for s, e in pairs]

        if not gpu_ms and not self._cpu_ms:
            return

        rows: list[tuple[str, str, list[float]]] = []
        for name, samples in gpu_ms.items():
            rows.append((name, "GPU", samples))
        for name, samples in self._cpu_ms.items():
            if name in gpu_ms:
                rows.append((f"{name} (cpu)", "CPU", samples))
            else:
                rows.append((name, "CPU", samples))

        rows.sort(key=lambda r: -(sum(r[2]) / max(len(r[2]), 1)))

        line = "═" * 86
        thin = "─" * 86
        print()
        print(line)
        print("  MOSS-TTS Stage 0 — per-phase timing")
        print(f"  pid={os.getpid()}   MOSS_TTS_TIMING={os.environ.get('MOSS_TTS_TIMING')}")
        print(line)
        print(
            f"  {'phase':<38} {'kind':<5} {'mean':>9} {'p50':>9} {'p99':>9} {'count':>8}"
        )
        print(thin)
        for name, kind, samples in rows:
            samples_sorted = sorted(samples)
            n = len(samples)
            mean = sum(samples) / n
            p50 = samples_sorted[n // 2]
            p99_idx = min(n - 1, max(0, int(n * 0.99) - 1))
            p99 = samples_sorted[p99_idx]
            print(
                f"  {name:<38} {kind:<5} {mean:>9.3f} {p50:>9.3f} {p99:>9.3f} {n:>8}"
            )
        print(thin)
        print("  All durations in milliseconds.  Sorted by mean (biggest first).")
        print(line)

_TIMER = _Stage0Timing()

def get_timer() -> _Stage0Timing:
    return _TIMER

atexit.register(_TIMER.dump)
