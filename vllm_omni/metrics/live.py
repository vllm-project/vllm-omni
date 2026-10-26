# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prometheus metrics for OpenAI Live sessions (``/v1/live/sessions``)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge

from vllm_omni.metrics import definitions as defs

_LABELS = list(defs.PIPELINE_LABELS)

_active_sessions = Gauge(defs.LIVE_ACTIVE_SESSIONS, "Number of active Live sessions.", labelnames=_LABELS)
_rejected_sessions = Counter(
    defs.LIVE_REJECTED_SESSIONS, "session.start refused at the max_sessions limit.", labelnames=_LABELS
)
_audio_seconds = Counter(
    defs.LIVE_AUDIO_SECONDS,
    "Live session audio, in seconds, by direction (input or output).",
    labelnames=[*_LABELS, "direction"],
)
_context_trims = Counter(
    defs.LIVE_CONTEXT_TRIMS, "History trims and resubmissions after context_length_exceeded.", labelnames=_LABELS
)
_errors = Counter(
    defs.LIVE_ERRORS, "Live sessions ended by a server-side error, by reason.", labelnames=[*_LABELS, "reason"]
)


class LiveSessionMetrics:
    """Per-session handle; every method is a no-op when ``log_stats`` is off."""

    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._log = log_stats
        self._labels = {"model_name": model_name}
        self._active = _active_sessions.labels(**self._labels)
        self._input = _audio_seconds.labels(**self._labels, direction="input")
        self._output = _audio_seconds.labels(**self._labels, direction="output")
        self._started = False

    def session_started(self) -> None:
        if self._log and not self._started:
            self._started = True
            self._active.inc()

    def session_finished(self) -> None:
        if self._log and self._started:
            self._started = False
            self._active.dec()

    def session_rejected(self) -> None:
        if self._log:
            _rejected_sessions.labels(**self._labels).inc()

    def input_audio(self, seconds: float) -> None:
        if self._log:
            self._input.inc(seconds)

    def output_audio(self, seconds: float) -> None:
        if self._log:
            self._output.inc(seconds)

    def context_trim(self) -> None:
        if self._log:
            _context_trims.labels(**self._labels).inc()

    def error(self, reason: str) -> None:
        if self._log:
            _errors.labels(**self._labels, reason=reason).inc()
