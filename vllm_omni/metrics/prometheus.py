from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs

_labelnames = list(defs.PIPELINE_LABELS)
_diffusion_labelnames = ["model_name", "engine"]

# Mapping from stage-emitted metric key (engine internal name) to the
# (prometheus family name, help text) we expose. Keys must match what the
# diffusion engine puts into its per-request metrics dict.
_DIFFUSION_METRIC_DEFS: dict[str, tuple[str, str]] = {
    "preprocess_time_ms": (
        defs.DIFFUSION_PREPROCESS_TIME_MS,
        "Diffusion preprocess time per request in milliseconds.",
    ),
    "diffusion_engine_exec_time_ms": (
        defs.DIFFUSION_EXEC_TIME_MS,
        "Diffusion model execution time per request in milliseconds.",
    ),
    "postprocess_time_ms": (
        defs.DIFFUSION_POSTPROCESS_TIME_MS,
        "Diffusion postprocess time per request in milliseconds.",
    ),
    "diffusion_engine_total_time_ms": (
        defs.DIFFUSION_STEP_TIME_MS,
        "Total diffusion step time per request in milliseconds.",
    ),
}

_running_family = Gauge(
    defs.NUM_REQUESTS_RUNNING,
    "Number of requests currently running across all pipeline stages.",
    labelnames=_labelnames,
)
_waiting_family = Gauge(
    defs.NUM_REQUESTS_WAITING,
    "Number of requests waiting to be scheduled.",
    labelnames=_labelnames,
)
_success_family = Counter(
    defs.NUM_REQUESTS_SUCCESS,
    "Number of requests that completed without error.",
    labelnames=_labelnames,
)
_fail_family = Counter(
    defs.NUM_REQUESTS_FAIL,
    "Number of requests that returned an error.",
    labelnames=_labelnames,
)
_e2e_latency_family = Histogram(
    defs.E2E_REQUEST_LATENCY_SECONDS,
    "Histogram of end-to-end request latency in seconds.",
    labelnames=_labelnames,
)
_queue_time_family = Histogram(
    defs.REQUEST_QUEUE_TIME_SECONDS,
    "Histogram of request queue wait time in seconds.",
    labelnames=_labelnames,
)
_diffusion_families: dict[str, Histogram] = {
    key: Histogram(metric_name, desc, labelnames=_diffusion_labelnames)
    for key, (metric_name, desc) in _DIFFUSION_METRIC_DEFS.items()
}


class OmniPrometheusMetrics:
    """Label-bound wrapper around the raw Prometheus metrics.

    Metric collectors use the ``vllm:omni_`` prefix to avoid being
    removed by upstream vLLM's ``unregister_vllm_metrics()``, which
    strips every collector whose ``_name`` contains ``"vllm"``.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._running = _running_family.labels(model_name=model_name)
        self._waiting = _waiting_family.labels(model_name=model_name)
        self._success = _success_family.labels(model_name=model_name)
        self._fail = _fail_family.labels(model_name=model_name)
        self._e2e_latency = _e2e_latency_family.labels(model_name=model_name)
        self._queue_time = _queue_time_family.labels(model_name=model_name)
        self._diffusion_by_stage: dict[tuple[str, int], Histogram] = {}

    def set_running(self, n: int) -> None:
        self._running.set(n)

    def set_waiting(self, n: int) -> None:
        self._waiting.set(n)

    def request_succeeded(self, e2e_seconds: float, queue_seconds: float | None = None) -> None:
        self._success.inc()
        self._e2e_latency.observe(e2e_seconds)
        if queue_seconds is not None:
            self._queue_time.observe(queue_seconds)

    def request_failed(self) -> None:
        self._fail.inc()

    def observe_diffusion_metrics(self, stage_id: int, metrics: dict[str, float]) -> None:
        for key, parent in _diffusion_families.items():
            value = metrics.get(key)
            if value is None:
                continue
            bound = self._diffusion_by_stage.get((key, stage_id))
            if bound is None:
                bound = parent.labels(
                    model_name=self._model_name,
                    engine=str(stage_id),
                )
                self._diffusion_by_stage[(key, stage_id)] = bound
            bound.observe(value)


class OmniRequestCounter:
    """Running-request counter written by the orchestrator thread, read by the client thread."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value = max(0, self.value - 1)
