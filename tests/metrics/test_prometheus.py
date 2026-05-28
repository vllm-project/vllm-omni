from __future__ import annotations

import re

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

from vllm_omni.metrics import OmniPrometheusMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "test-model"

_PIPELINE_METRICS = [
    "vllm:omni_num_requests_running",
    "vllm:omni_num_requests_waiting",
    "vllm:omni_requests_success",
    "vllm:omni_e2e_request_latency_s",
]


@pytest.fixture(scope="module")
def registry() -> CollectorRegistry:
    return REGISTRY


@pytest.fixture(scope="module")
def prom() -> OmniPrometheusMetrics:
    return OmniPrometheusMetrics(model_name=_MODEL)


@pytest.fixture(scope="module")
def scrape_output(prom: OmniPrometheusMetrics, registry: CollectorRegistry) -> str:
    # Two natural completions (stop) + one length-cap + one failure (abort)
    # exercise three distinct finished_reason buckets in the merged Counter.
    prom.request_succeeded(e2e_seconds=1.5, finished_reason="stop")
    prom.request_succeeded(e2e_seconds=2.0, finished_reason="stop")
    prom.request_succeeded(e2e_seconds=3.0, finished_reason="length")
    prom.request_failed()  # → finished_reason="abort"
    prom.set_running(5)
    prom.set_waiting(2)
    return generate_latest(registry).decode()


def _sample_value(output: str, metric_line: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(metric_line):
            return float(line.split()[-1])
    return None


class TestMetricObservation:
    def test_all_metric_families_present(self, scrape_output: str) -> None:
        for name in _PIPELINE_METRICS:
            assert f"# HELP {name}" in scrape_output, f"missing metric family: {name}"

    def test_counter_values(self, scrape_output: str) -> None:
        # Per-reason buckets sourced from the merged completion Counter.
        stop = _sample_value(
            scrape_output,
            f'vllm:omni_requests_success_total{{finished_reason="stop",model_name="{_MODEL}"}}',
        )
        assert stop == 2.0

        length = _sample_value(
            scrape_output,
            f'vllm:omni_requests_success_total{{finished_reason="length",model_name="{_MODEL}"}}',
        )
        assert length == 1.0

        abort = _sample_value(
            scrape_output,
            f'vllm:omni_requests_success_total{{finished_reason="abort",model_name="{_MODEL}"}}',
        )
        assert abort == 1.0

    def test_gauge_values(self, scrape_output: str) -> None:
        running = _sample_value(
            scrape_output,
            f'vllm:omni_num_requests_running{{model_name="{_MODEL}"}}',
        )
        assert running == 5.0

        waiting = _sample_value(
            scrape_output,
            f'vllm:omni_num_requests_waiting{{model_name="{_MODEL}"}}',
        )
        assert waiting == 2.0

    def test_histogram_counts(self, scrape_output: str) -> None:
        # 3 successful completions (stop x2 + length x1) all observe e2e;
        # the 1 failed completion only increments the Counter without
        # observing the latency histogram, so the count stays at 3.
        e2e_count = _sample_value(
            scrape_output,
            f'vllm:omni_e2e_request_latency_s_count{{model_name="{_MODEL}"}}',
        )
        assert e2e_count == 3.0


class TestLabelCorrectness:
    def test_pipeline_metrics_carry_model_name(self, scrape_output: str) -> None:
        for name in _PIPELINE_METRICS:
            pattern = rf'^{re.escape(name)}.*model_name="{re.escape(_MODEL)}"'
            assert re.search(pattern, scrape_output, re.MULTILINE), f"{name} missing model_name label"

    def test_no_legacy_engine_label(self, scrape_output: str) -> None:
        assert 'engine="' not in scrape_output

    def test_no_legacy_seconds_or_ms_families(self, scrape_output: str) -> None:
        # Renamed: *_time_ms → *_s; *_time_seconds dropped.
        for legacy in (
            "vllm:omni_request_queue_time_seconds",
            "vllm:omni_e2e_request_latency_seconds",
        ):
            assert legacy not in scrape_output, f"legacy family {legacy} still registered"


class TestScrapeOutput:
    def test_omni_metrics_in_default_registry(self, scrape_output: str) -> None:
        for name in _PIPELINE_METRICS:
            assert name in scrape_output

    def test_process_metrics_in_default_registry(self, scrape_output: str) -> None:
        # vllm:* metrics require a full PrometheusStatLogger with VllmConfig
        # and are registered by the Orchestrator at server startup. Verifying
        # their presence is covered by integration tests. Here we confirm the
        # default registry is being scraped by checking for process_* metrics
        # from the Python prometheus_client runtime.
        assert "process_" in scrape_output
