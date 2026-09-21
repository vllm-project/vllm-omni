# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-step wall-clock profiler for MiniCPM-o NPU runners.

Zero-ish overhead when disabled. Enable with ``OMNI_STEP_PROF=1`` and set the
aggregation cadence with ``OMNI_STEP_PROF_FLUSH`` (totals between dumps,
default 200). Thread-safe; spans are keyed per thread so nested/overlapping
regions in the engine loop stay correct.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

_ENABLED = os.environ.get("OMNI_STEP_PROF", "0") == "1"
_FLUSH_EVERY = int(os.environ.get("OMNI_STEP_PROF_FLUSH", "200"))


class _StepProf:
    def __init__(self) -> None:
        self.acc: dict[str, float] = defaultdict(float)
        self.cnt: dict[str, int] = defaultdict(int)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._flushed = 0

    def tic(self, name: str) -> None:
        marks = getattr(self._local, "marks", None)
        if marks is None:
            marks = {}
            self._local.marks = marks
        marks[name] = time.perf_counter()

    def toc(self, name: str) -> None:
        marks = getattr(self._local, "marks", None)
        if not marks:
            return
        t0 = marks.pop(name, None)
        if t0 is None:
            return
        dt = time.perf_counter() - t0
        with self._lock:
            self.acc[name] += dt
            self.cnt[name] += 1
            total = self.cnt.get("em_total", 0)
            if total and total - self._flushed >= _FLUSH_EVERY:
                self._flushed = total
                self._dump_locked()

    def _dump_locked(self) -> None:
        parts = []
        for name in sorted(self.acc):
            n = self.cnt[name]
            if n:
                parts.append(f"{name}={self.acc[name] / n * 1e3:.3f}ms")
        sys.stderr.write("[STEPPROF] " + " ".join(parts) + "\n")
        sys.stderr.flush()


_PROF = _StepProf()


def tic(name: str) -> None:
    if _ENABLED:
        _PROF.tic(name)


def toc(name: str) -> None:
    if _ENABLED:
        _PROF.toc(name)


@contextmanager
def span(name: str):
    if not _ENABLED:
        yield
        return
    _PROF.tic(name)
    try:
        yield
    finally:
        _PROF.toc(name)
