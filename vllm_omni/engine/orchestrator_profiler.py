"""Optional orchestrator phase profiler (diagnostic only).

Enable with:
  export VLLM_OMNI_ORCH_PROFILE=1
  export VLLM_OMNI_ORCH_PROFILE_PATH=/path/to/orch_profile.jsonl
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

_ENABLED = os.environ.get("VLLM_OMNI_ORCH_PROFILE", "").lower() in ("1", "true", "yes")
_OUTPUT_PATH = os.environ.get(
    "VLLM_OMNI_ORCH_PROFILE_PATH",
    "/tmp/vllm_omni_orch_profile.jsonl",
)
_FLUSH_INTERVAL_S = float(os.environ.get("VLLM_OMNI_ORCH_PROFILE_FLUSH_S", "1.0"))


class OrchestratorProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals_s: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._value_sums: dict[str, float] = defaultdict(float)
        self._value_counts: dict[str, int] = defaultdict(int)
        self._value_max: dict[str, float] = defaultdict(float)
        self._value_last: dict[str, float] = {}
        self._window_start = time.monotonic()
        self._last_flush = self._window_start
        self._loop_iterations = 0
        self._loop_idle_iterations = 0

    @property
    def enabled(self) -> bool:
        return _ENABLED

    def record(self, phase: str, elapsed_s: float, **tags: Any) -> None:
        if not _ENABLED or elapsed_s < 0:
            return
        key = phase
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{phase}|{tag_str}"
        with self._lock:
            self._totals_s[key] += elapsed_s
            self._counts[key] += 1
            now = time.monotonic()
            if now - self._last_flush >= _FLUSH_INTERVAL_S:
                self._flush_locked(now)

    def record_value(self, name: str, value: float, **tags: Any) -> None:
        if not _ENABLED:
            return
        key = name
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{name}|{tag_str}"
        with self._lock:
            self._value_sums[key] += float(value)
            self._value_counts[key] += 1
            self._value_max[key] = max(self._value_max[key], float(value))
            self._value_last[key] = float(value)
            now = time.monotonic()
            if now - self._last_flush >= _FLUSH_INTERVAL_S:
                self._flush_locked(now)

    def note_loop(self, *, idle: bool) -> None:
        if not _ENABLED:
            return
        with self._lock:
            self._loop_iterations += 1
            if idle:
                self._loop_idle_iterations += 1

    def _flush_locked(self, now: float) -> None:
        window_s = max(now - self._window_start, 1e-9)
        profiled_s = sum(self._totals_s.values()) or 1e-9
        payload = {
            "ts": time.time(),
            "window_s": window_s,
            "loop_iterations": self._loop_iterations,
            "loop_idle_iterations": self._loop_idle_iterations,
            "phases": {
                key: {
                    "total_ms": totals * 1000.0,
                    "count": self._counts[key],
                    "avg_ms": (totals / self._counts[key]) * 1000.0 if self._counts[key] else 0.0,
                    "pct": (totals / profiled_s) * 100.0,
                }
                for key, totals in self._totals_s.items()
            },
            "values": {
                key: {
                    "count": self._value_counts[key],
                    "avg": self._value_sums[key] / self._value_counts[key] if self._value_counts[key] else 0.0,
                    "max": self._value_max[key],
                    "last": self._value_last.get(key, 0.0),
                }
                for key in self._value_counts
            },
        }
        try:
            with open(_OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            pass
        self._totals_s.clear()
        self._counts.clear()
        self._value_sums.clear()
        self._value_counts.clear()
        self._value_max.clear()
        self._value_last.clear()
        self._loop_iterations = 0
        self._loop_idle_iterations = 0
        self._window_start = now
        self._last_flush = now

    def flush(self) -> None:
        if not _ENABLED:
            return
        with self._lock:
            self._flush_locked(time.monotonic())


_profiler: OrchestratorProfiler | None = None
_profiler_lock = threading.Lock()


def get_orchestrator_profiler() -> OrchestratorProfiler:
    global _profiler
    if _profiler is None:
        with _profiler_lock:
            if _profiler is None:
                _profiler = OrchestratorProfiler()
    return _profiler


@contextmanager
def profile_phase(phase: str, **tags: Any) -> Iterator[None]:
    prof = get_orchestrator_profiler()
    if not prof.enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        prof.record(phase, time.perf_counter() - t0, **tags)
