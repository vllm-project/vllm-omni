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

_stage_gen_time_family = Histogram(
    defs.STAGE_GEN_TIME_S,
    "Per-stage generation time in seconds for diffusion / image stages.",
    labelnames=list(defs.STAGE_GEN_TIME_LABELS),
    buckets=defs.SECONDS_BUCKETS,
)
_request_queue_wait_family = Histogram(
    defs.REQUEST_QUEUE_WAIT_S,
    "Per-request queue wait between submit arrival and stage submit, in seconds.",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)
_stage_waiting_requests_family = Gauge(
    defs.STAGE_WAITING_REQUESTS,
    "Sum of the latest scheduler waiting snapshots across a stage's replicas; "
    "diffusion snapshots update when outputs arrive.",
    labelnames=list(defs.DIFFUSION_LABELS),
)
_num_inference_steps_family = Histogram(
    defs.NUM_INFERENCE_STEPS,
    "Diffusion step count distribution per request.",
    labelnames=_labelnames,
)
_image_count_family = Counter(
    defs.IMAGE_COUNT_METRIC,
    "Cumulative image count; per-model rate. Throughput recovered via rate().",
    labelnames=_labelnames,
)
_image_pixels_family = Histogram(
    defs.IMAGE_PIXELS_METRIC,
    "Image pixel count distribution per request.",
    labelnames=_labelnames,
)
_peak_memory_family = Gauge(
    defs.PEAK_MEMORY_MB,
    "Peak GPU memory in MB observed during a stage's generation.",
    labelnames=list(defs.DIFFUSION_LABELS),
)
_stage_in_queue_family = Histogram(
    defs.STAGE_IN_QUEUE_S,
    "Time in seconds from diffusion scheduler enqueue to initial execution admission. "
    "Zero is the healthy low-load baseline and is counted.",
    labelnames=list(defs.DIFFUSION_LABELS),
    buckets=defs.SECONDS_FAST_BUCKETS,
)

_requests_failed_family = Counter(
    defs.REQUESTS_FAILED,
    "Total requests by failure reason. Pairs with requests_success_total "
    "(abort bucket) to give the full failure surface. Reasons are bounded to "
    "client_abort, client_disconnect, stage_error, and unknown.",
    labelnames=list(defs.FAILED_LABELS),
)
_kv_wait_s_family = Histogram(
    defs.KV_WAIT_S,
    "KV cache transfer wait time per request, sliced by connector backend.",
    labelnames=list(defs.KV_WAIT_LABELS),
    buckets=defs.SECONDS_BUCKETS,
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
        self._request_queue_wait = _request_queue_wait_family.labels(model_name=model_name)
        self._num_inference_steps = _num_inference_steps_family.labels(model_name=model_name)
        self._image_count = _image_count_family.labels(model_name=model_name)
        self._image_pixels = _image_pixels_family.labels(model_name=model_name)

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
        _completion_family.labels(
            model_name=self._model_name,
            finished_reason="abort",
        ).inc()

    def observe_stage_gen_time(self, stage: int, stage_type: str, gen_time_s: float) -> None:
        if not self._log_stats:
            return
        _stage_gen_time_family.labels(
            model_name=self._model_name,
            stage=str(stage),
            stage_type=stage_type,
        ).observe(max(gen_time_s, 0.0))

    def observe_stage_in_queue(self, stage: int, in_queue_s: float) -> None:
        if not self._log_stats:
            return
        _stage_in_queue_family.labels(
            model_name=self._model_name,
            stage=str(stage),
        ).observe(max(in_queue_s, 0.0))

    def observe_queue_wait(self, queue_wait_s: float) -> None:
        if not self._log_stats:
            return
        self._request_queue_wait.observe(max(queue_wait_s, 0.0))

    def set_stage_waiting_requests(self, stage: int, n_waiting: int) -> None:
        if not self._log_stats:
            return
        _stage_waiting_requests_family.labels(
            model_name=self._model_name,
            stage=str(stage),
        ).set(max(n_waiting, 0))

    def observe_num_inference_steps(self, n_steps: int) -> None:
        if not self._log_stats or n_steps <= 0:
            return
        self._num_inference_steps.observe(n_steps)

    def inc_image_count(self, n_images: int = 1) -> None:
        if not self._log_stats or n_images <= 0:
            return
        self._image_count.inc(n_images)

    def observe_image_pixels(self, n_pixels: int) -> None:
        if not self._log_stats or n_pixels <= 0:
            return
        self._image_pixels.observe(n_pixels)

    def set_peak_memory(self, stage: int, peak_memory_mb: float) -> None:
        if not self._log_stats:
            return
        _peak_memory_family.labels(
            model_name=self._model_name,
            stage=str(stage),
        ).set(max(peak_memory_mb, 0.0))

    def inc_requests_failed(self, reason: str) -> None:
        if not self._log_stats:
            return
        _requests_failed_family.labels(
            model_name=self._model_name,
            reason=reason or "unknown",
        ).inc()

    def observe_kv_wait(self, connector_type: str, kv_wait_s: float) -> None:
        if not self._log_stats or kv_wait_s < 0:
            return
        _kv_wait_s_family.labels(
            model_name=self._model_name,
            connector_type=connector_type or "unknown",
        ).observe(kv_wait_s)


class OmniRequestCounter:
    """Running-request counter written by the orchestrator thread, read by the client thread."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value = max(0, self.value - 1)
