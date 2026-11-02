# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk-shape counters for the MiniCPM-o Code2Wav (stage-2) solve path.

How much stage 2 costs per request depends on how many codec frames a single
CFM / HiFT solve covers: a multi-frame Talker step hands stage 2 eight frames
at once, a single-frame step hands one. A solve that only ever covers one frame
pays the fixed per-solve cost once per frame, so the frame histogram is the
first thing to look at when stage 2, not the Talker, decides throughput.

These counters are diagnostic only. They are off unless
``VLLM_OMNI_CODE2WAV_STATS`` is set, and when off every call site is a single
``None`` check on the caller's side, so the serving path is unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter

CODE2WAV_STATS_ENV = "VLLM_OMNI_CODE2WAV_STATS"

_FLUSH_EVERY = 200
_TOP_BUCKETS = 6

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class Code2WavSolveStats:
    """Window counters for CFM and HiFT solves in one Code2Wav worker.

    Every ``record`` call is one solve. The window is flushed every
    ``flush_every`` solves: it reports the frame / batch histograms and the
    fraction of wall time spent between consecutive solves (``duty``). A duty
    close to 100% means stage-2 work is back to back, i.e. it is the stage that
    bounds throughput; a low duty means it is waiting on its upstream.
    """

    def __init__(self, label: str = "code2wav", flush_every: int = _FLUSH_EVERY) -> None:
        self._label = label
        self._flush_every = max(int(flush_every), 1)
        self._enabled = _env_flag(CODE2WAV_STATS_ENV)
        self._solves = 0
        self._frames: Counter[tuple[str, int]] = Counter()
        self._batches: Counter[tuple[str, int]] = Counter()
        self._ragged: Counter[str] = Counter()
        self._pending = 0
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._window_busy = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, kind: str, *, batch: int, frames: int, ragged: bool = False) -> None:
        """Count one solve of ``kind`` covering ``frames`` frames per row."""
        if not self._enabled:
            return
        now = time.perf_counter()
        if self._first_ts is None:
            self._first_ts = now
        elif self._last_ts is not None:
            # Gap between consecutive solves: the window "busy" time. It
            # overstates stage-2 work by the solve latency itself, which is
            # what makes duty a usable saturation indicator, not a profile.
            self._window_busy += now - self._last_ts
        self._last_ts = now
        self._solves += 1
        self._frames[(kind, int(frames))] += 1
        self._batches[(kind, int(batch))] += 1
        if ragged:
            self._ragged[kind] += 1
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Emit the current window and start a new one."""
        if not self._enabled or not self._pending:
            return
        wall = (self._last_ts - self._first_ts) if self._first_ts and self._last_ts else 0.0
        parts = []
        for kind in sorted({key[0] for key in self._frames}):
            frames = sorted((f, n) for (k, f), n in self._frames.items() if k == kind)
            batches = sorted((b, n) for (k, b), n in self._batches.items() if k == kind)
            parts.append(
                f"{kind}(n={sum(n for _, n in frames)} "
                f"frames={{{self._hist(frames)}}} "
                f"batch={{{self._hist(batches)}}} "
                f"ragged={self._ragged.get(kind, 0)})"
            )
        duty = (self._window_busy / wall * 100.0) if wall > 0 else 0.0
        logger.info(
            "[%s-stats] solves=%d window=%.1fs busy=%.1fs duty=%.0f%% %s",
            self._label,
            self._solves,
            wall,
            self._window_busy,
            duty,
            " ".join(parts),
        )
        self._pending = 0
        self._frames.clear()
        self._batches.clear()
        self._ragged.clear()
        self._window_busy = 0.0
        self._last_ts = None

    def summary(self) -> dict[str, object]:
        """Return the window in a form tests can assert on."""
        return {
            "solves": self._solves,
            "pending": self._pending,
            "frames": {(k, f): n for (k, f), n in self._frames.items()},
            "batches": {(k, b): n for (k, b), n in self._batches.items()},
        }

    @staticmethod
    def _hist(pairs: list[tuple[int, int]]) -> str:
        head = ",".join(f"{value}:{count}" for value, count in pairs[:_TOP_BUCKETS])
        rest = len(pairs) - _TOP_BUCKETS
        return f"{head},+{rest}" if rest > 0 else head
