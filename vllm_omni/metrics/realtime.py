# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs

_active_sessions = Gauge(
    defs.REALTIME_VAD_ACTIVE_SESSIONS,
    "Number of active Realtime sessions using server-side VAD.",
    labelnames=list(defs.PIPELINE_LABELS),
)
_inference_latency = Histogram(
    defs.REALTIME_VAD_INFERENCE_LATENCY_S,
    "Server VAD inference time for one appended audio batch, in seconds.",
    labelnames=list(defs.PIPELINE_LABELS),
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_endpoint_delay = Histogram(
    defs.REALTIME_VAD_ENDPOINT_DELAY_S,
    "Detected silence between the last speech frame and the committed endpoint, in seconds.",
    labelnames=list(defs.PIPELINE_LABELS),
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_errors = Counter(
    defs.REALTIME_VAD_ERRORS,
    "Server VAD errors and input-overflow rejections by reason.",
    labelnames=list(defs.FAILED_LABELS),
)


class RealtimeVADMetrics:
    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._model_name = model_name
        self._log_stats = log_stats
        self._active_sessions = _active_sessions.labels(model_name=model_name)
        self._inference_latency = _inference_latency.labels(model_name=model_name)
        self._endpoint_delay = _endpoint_delay.labels(model_name=model_name)

    def session_started(self) -> None:
        if self._log_stats:
            self._active_sessions.inc()

    def session_finished(self) -> None:
        if self._log_stats:
            self._active_sessions.dec()

    def observe_inference(self, latency_ms: float) -> None:
        if self._log_stats:
            self._inference_latency.observe(latency_ms / 1000)

    def observe_endpoint_delay(self, delay_ms: int) -> None:
        if self._log_stats:
            self._endpoint_delay.observe(delay_ms / 1000)

    def error(self, reason: str) -> None:
        if self._log_stats:
            _errors.labels(model_name=self._model_name, reason=reason).inc()
