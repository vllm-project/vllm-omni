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
from collections.abc import Callable

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
    fraction of wall time actually spent inside the solves (``duty``). A duty
    close to 100% means stage-2 work is back to back, i.e. it is the stage that
    bounds throughput; a low duty means it is waiting on its upstream.
    """

    def __init__(
        self,
        label: str = "code2wav",
        flush_every: int = _FLUSH_EVERY,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._label = label
        self._flush_every = max(int(flush_every), 1)
        self._clock = clock
        self._enabled = _env_flag(CODE2WAV_STATS_ENV)
        self._solves = 0
        self._frames: Counter[tuple[str, int]] = Counter()
        self._batches: Counter[tuple[str, int]] = Counter()
        self._ragged: Counter[str] = Counter()
        self._pending = 0
        # Window-local start: a global first timestamp would measure the load
        # and graph-capture phase, not the serving window, in `duty`.
        self._window_start: float | None = None
        self._last_ts: float | None = None
        self._window_busy = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self, kind: str, *, batch: int, frames: int, ragged: bool = False, duration: float | None = None
    ) -> None:
        """Count one solve of ``kind`` covering ``frames`` frames per row.

        ``duration`` is the solve's measured wall time (start/stop around the
        CFM/HiFT call). It is what makes ``duty`` a real occupancy figure:
        accumulating the inter-record gaps instead telescopes every solve
        into the next one, so ``busy`` always equals ``wall`` and duty reads
        a flat 100% however idle stage 2 is. Without a duration the gap is
        the only signal available and the old, overstated behaviour applies.
        """
        if not self._enabled:
            return
        now = self._clock()
        if self._last_ts is not None:
            if duration is not None:
                self._window_busy += duration
            else:
                self._window_busy += now - self._last_ts
        else:
            # Anchor the window at the *start* of the first solve: its
            # duration is already counted in busy, so anchoring at its end
            # would report busy > wall (duty above 100%) for back-to-back
            # solves.
            self._window_start = now - (duration if duration is not None else 0.0)
            if duration is not None:
                self._window_busy += duration
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
        wall = (
            (self._last_ts - self._window_start)
            if self._window_start is not None and self._last_ts is not None
            else 0.0
        )
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
        self._window_start = None
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
