from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs

_labelnames = list(defs.PIPELINE_LABELS)
_diffusion_labelnames = list(defs.STAGE_LABELS)

# Map from stage-emitted metric key (engine internal name, in milliseconds) to
# (prometheus family name, help text). The Prometheus side exposes seconds;
# observe_diffusion_metrics divides by 1000 at emit time.
_DIFFUSION_METRIC_DEFS: dict[str, tuple[str, str]] = {
    "preprocess_time_ms": (
        defs.DIFFUSION_PREPROCESS_S,
        "Diffusion preprocess time per request in seconds "
        "(tokenizer + text_encoder + vae.encode).",
    ),
    "diffusion_engine_exec_time_ms": (
        defs.DIFFUSION_EXEC_S,
        "Diffusion executor work time per request in seconds.",
    ),
    "postprocess_time_ms": (
        defs.DIFFUSION_POSTPROCESS_S,
        "Diffusion postprocess time per request in seconds (vae.decode).",
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
_completion_family = Counter(
    defs.REQUESTS_SUCCESS,
    "Total requests by completion reason "
    "(stop / length / abort / ...). Aborts cover client-disconnect / "
    "cancellation paths in addition to upstream FinishReason.ABORT.",
    labelnames=list(defs.SUCCESS_LABELS),
)
_e2e_latency_family = Histogram(
    defs.E2E_REQUEST_LATENCY_S,
    "Pipeline-global end-to-end request latency in seconds "
    "(user arrival to complete response).",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)
_diffusion_families: dict[str, Histogram] = {
    key: Histogram(
        metric_name, desc, labelnames=_diffusion_labelnames, buckets=defs.SECONDS_FAST_BUCKETS
    )
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
        self._e2e_latency = _e2e_latency_family.labels(model_name=model_name)
        # Cache per (metric_key, stage, replica) bound child so .labels() is
        # only paid once per replica even though observe_diffusion_metrics()
        # is invoked per finished diffusion request.
        self._diffusion_by_replica: dict[tuple[str, int, int], Histogram] = {}

    def set_running(self, n: int) -> None:
        self._running.set(n)

    def set_waiting(self, n: int) -> None:
        self._waiting.set(n)

    def request_succeeded(
        self,
        e2e_seconds: float,
        finished_reason: str = "stop",
    ) -> None:
        _completion_family.labels(
            model_name=self._model_name,
            finished_reason=finished_reason,
        ).inc()
        self._e2e_latency.observe(e2e_seconds)

    def request_failed(self) -> None:
        # Pipeline-level "fail" maps to the upstream FinishReason.ABORT bucket;
        # a single counter family now covers both normal stops and aborts.
        _completion_family.labels(
            model_name=self._model_name,
            finished_reason="abort",
        ).inc()

    def observe_diffusion_metrics(
        self,
        stage_id: int,
        replica_id: int,
        metrics: dict[str, float],
    ) -> None:
        for key, parent in _diffusion_families.items():
            value = metrics.get(key)
            if value is None:
                continue
            cache_key = (key, stage_id, replica_id)
            bound = self._diffusion_by_replica.get(cache_key)
            if bound is None:
                bound = parent.labels(
                    model_name=self._model_name,
                    stage=str(stage_id),
                    replica=str(replica_id),
                )
                self._diffusion_by_replica[cache_key] = bound
            # Source values are in milliseconds (legacy engine_outputs dict
            # keys); the exposed families use seconds.
            bound.observe(float(value) / 1000.0)


class OmniRequestCounter:
    """Running-request counter written by the orchestrator thread, read by the client thread."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value = max(0, self.value - 1)
