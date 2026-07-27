from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs

_labelnames = list(defs.PIPELINE_LABELS)

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
    "Pipeline-global end-to-end request latency in seconds (user arrival to complete response).",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)
_prompt_tokens_family = Counter(
    defs.PROMPT_TOKENS,
    "Total prompt (input) tokens processed across all pipeline stages.",
    labelnames=_labelnames,
)
_generation_tokens_family = Counter(
    defs.GENERATION_TOKENS,
    "Total generation (output) tokens produced across all pipeline stages.",
    labelnames=_labelnames,
)


class OmniPrometheusMetrics:
    """Label-bound wrapper around the raw Prometheus metrics.

    Metric collectors use the ``vllm_omni:`` prefix, distinct from the
    upstream ``vllm:*`` families.
    """

    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._model_name = model_name
        self._log_stats = log_stats
        self._running = _running_family.labels(model_name=model_name)
        self._waiting = _waiting_family.labels(model_name=model_name)
        self._e2e_latency = _e2e_latency_family.labels(model_name=model_name)
        self._prompt_tokens = _prompt_tokens_family.labels(model_name=model_name)
        self._generation_tokens = _generation_tokens_family.labels(model_name=model_name)

    def set_running(self, n: int) -> None:
        if not self._log_stats:
            return
        self._running.set(n)

    def set_waiting(self, n: int) -> None:
        if not self._log_stats:
            return
        self._waiting.set(n)

    def observe_tokens(self, prompt_tokens: int, generation_tokens: int) -> None:
        if not self._log_stats:
            return
        if prompt_tokens > 0:
            self._prompt_tokens.inc(prompt_tokens)
        if generation_tokens > 0:
            self._generation_tokens.inc(generation_tokens)

    def request_succeeded(
        self,
        e2e_seconds: float,
        finished_reason: str = "stop",
    ) -> None:
        if not self._log_stats:
            return
        _completion_family.labels(
            model_name=self._model_name,
            finished_reason=finished_reason,
        ).inc()
        self._e2e_latency.observe(e2e_seconds)

    def request_failed(self) -> None:
        if not self._log_stats:
            return
        # Pipeline-level "fail" maps to the upstream FinishReason.ABORT bucket;
        # a single counter family now covers both normal stops and aborts.
        _completion_family.labels(
            model_name=self._model_name,
            finished_reason="abort",
        ).inc()


class OmniRequestCounter:
    """Running-request counter written by the orchestrator thread, read by the client thread."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value = max(0, self.value - 1)
