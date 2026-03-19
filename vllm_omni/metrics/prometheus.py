from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import CollectorRegistry, Counter, Histogram
from vllm.v1.metrics.prometheus import get_prometheus_registry

DEFAULT_OUTPUT_TYPE = "unknown"
E2E_LATENCY_BUCKETS = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
    45.0,
    60.0,
    90.0,
    120.0,
    180.0,
    300.0,
)


def normalize_output_type(output_type: str | None) -> str:
    if not output_type:
        return DEFAULT_OUTPUT_TYPE
    return str(output_type)


def infer_request_output_type(output_modalities: Sequence[str] | None) -> str:
    if not output_modalities:
        return DEFAULT_OUTPUT_TYPE
    if isinstance(output_modalities, str):
        return normalize_output_type(output_modalities)

    unique_modalities = {normalize_output_type(modality) for modality in output_modalities if modality}
    if len(unique_modalities) != 1:
        return DEFAULT_OUTPUT_TYPE
    return unique_modalities.pop()


class OmniPrometheusMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or get_prometheus_registry()
        self.stage_generation_seconds = Histogram(
            name="vllm:omni_stage_generation_seconds",
            documentation="Generation time for completed vLLM-Omni stage events.",
            labelnames=("model_name", "stage_id", "final_output_type"),
            buckets=E2E_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.stage_postprocess_seconds = Histogram(
            name="vllm:omni_stage_postprocess_seconds",
            documentation="Postprocess time for completed vLLM-Omni stage events.",
            labelnames=("model_name", "stage_id", "final_output_type"),
            buckets=E2E_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.transfer_bytes_total = Counter(
            name="vllm:omni_transfer_bytes_total",
            documentation="Total bytes transferred between vLLM-Omni stages.",
            labelnames=("model_name", "from_stage", "to_stage", "used_shm"),
            registry=self.registry,
        )
        self.requests_total = Counter(
            name="vllm:omni_requests_total",
            documentation="Total number of vLLM-Omni orchestrator requests.",
            labelnames=("model_name", "final_output_type"),
            registry=self.registry,
        )
        self.requests_aborted_total = Counter(
            name="vllm:omni_requests_aborted_total",
            documentation="Total number of aborted vLLM-Omni orchestrator requests.",
            labelnames=("model_name", "final_output_type"),
            registry=self.registry,
        )
        self.e2e_request_latency_seconds = Histogram(
            name="vllm:omni_e2e_request_latency_seconds",
            documentation="End-to-end latency of successful vLLM-Omni requests.",
            labelnames=("model_name", "final_output_type"),
            buckets=E2E_LATENCY_BUCKETS,
            registry=self.registry,
        )

    def on_request_started(self, model_name: str, final_output_type: str | None = None) -> None:
        self.requests_total.labels(
            model_name=model_name,
            final_output_type=normalize_output_type(final_output_type),
        ).inc()

    def on_stage_completed(
        self,
        model_name: str,
        stage_id: int,
        final_output_type: str | None,
        generation_seconds: float,
    ) -> None:
        self.stage_generation_seconds.labels(
            model_name=model_name,
            stage_id=str(stage_id),
            final_output_type=normalize_output_type(final_output_type),
        ).observe(generation_seconds)

    def on_stage_postprocessed(
        self,
        model_name: str,
        stage_id: int,
        final_output_type: str | None,
        postprocess_seconds: float,
    ) -> None:
        self.stage_postprocess_seconds.labels(
            model_name=model_name,
            stage_id=str(stage_id),
            final_output_type=normalize_output_type(final_output_type),
        ).observe(postprocess_seconds)

    def on_transfer_recorded(
        self,
        model_name: str,
        from_stage: int,
        to_stage: int,
        used_shm: bool,
        size_bytes: int,
    ) -> None:
        self.transfer_bytes_total.labels(
            model_name=model_name,
            from_stage=str(from_stage),
            to_stage=str(to_stage),
            used_shm=str(bool(used_shm)),
        ).inc(size_bytes)

    def on_request_succeeded(self, model_name: str, final_output_type: str | None, latency_seconds: float) -> None:
        self.e2e_request_latency_seconds.labels(
            model_name=model_name,
            final_output_type=normalize_output_type(final_output_type),
        ).observe(latency_seconds)

    def on_request_aborted(self, model_name: str, final_output_type: str | None = None) -> None:
        self.requests_aborted_total.labels(
            model_name=model_name,
            final_output_type=normalize_output_type(final_output_type),
        ).inc()


omni_prometheus_metrics = OmniPrometheusMetrics()
