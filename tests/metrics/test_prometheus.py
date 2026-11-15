# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import re
from functools import partial

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, generate_latest, multiprocess, values

from vllm_omni.metrics import OmniPrometheusMetrics
from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics import prometheus as prom_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "test-model"

_PIPELINE_METRICS = [
    "vllm_omni:num_requests_running",
    "vllm_omni:num_requests_waiting",
    "vllm_omni:requests_success",
    "vllm_omni:e2e_request_latency_s",
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
            f'vllm_omni:requests_success_total{{finished_reason="stop",model_name="{_MODEL}"}}',
        )
        assert stop == 2.0

        length = _sample_value(
            scrape_output,
            f'vllm_omni:requests_success_total{{finished_reason="length",model_name="{_MODEL}"}}',
        )
        assert length == 1.0

        abort = _sample_value(
            scrape_output,
            f'vllm_omni:requests_success_total{{finished_reason="abort",model_name="{_MODEL}"}}',
        )
        assert abort == 1.0

    def test_gauge_values(self, scrape_output: str) -> None:
        running = _sample_value(
            scrape_output,
            f'vllm_omni:num_requests_running{{model_name="{_MODEL}"}}',
        )
        assert running == 5.0

        waiting = _sample_value(
            scrape_output,
            f'vllm_omni:num_requests_waiting{{model_name="{_MODEL}"}}',
        )
        assert waiting == 2.0

    def test_histogram_counts(self, scrape_output: str) -> None:
        # 3 successful completions (stop x2 + length x1) all observe e2e;
        # the 1 failed completion only increments the Counter without
        # observing the latency histogram, so the count stays at 3.
        e2e_count = _sample_value(
            scrape_output,
            f'vllm_omni:e2e_request_latency_s_count{{model_name="{_MODEL}"}}',
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
            "vllm_omni:request_queue_time_seconds",
            "vllm_omni:e2e_request_latency_seconds",
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


@pytest.fixture
def duplex_registry(monkeypatch) -> CollectorRegistry:
    isolated = CollectorRegistry()
    monkeypatch.setattr(prom_module, "_duplex_families", None)
    monkeypatch.setattr(prom_module, "Gauge", partial(Gauge, registry=isolated))
    monkeypatch.setattr(prom_module, "Counter", partial(Counter, registry=isolated))
    return isolated


class TestDuplexPlacementMetrics:
    def test_legacy_metrics_do_not_register_duplex_collectors(self, duplex_registry) -> None:
        metrics = OmniPrometheusMetrics(model_name="duplex-legacy-metrics")
        metrics.set_running(1)
        metrics.set_waiting(0)

        assert prom_module._duplex_families is None
        assert generate_latest(duplex_registry) == b""

    def test_disabled_metrics_do_not_register_duplex_collectors(self, duplex_registry) -> None:
        metrics = OmniPrometheusMetrics(model_name="duplex-disabled-metrics", log_stats=False)
        metrics.set_duplex_reserved_sessions(stage=0, replica=1, reserved=2)
        metrics.inc_duplex_admission_rejected("global_capacity_exhausted")

        assert prom_module._duplex_families is None
        assert generate_latest(duplex_registry) == b""

    def test_reserved_sessions_are_per_replica_and_return_to_zero(self, duplex_registry) -> None:
        metrics = OmniPrometheusMetrics(model_name="duplex-reserved-metrics")
        metrics.set_duplex_reserved_sessions(stage=0, replica=0, reserved=2)
        metrics.set_duplex_reserved_sessions(stage=0, replica=1, reserved=1)
        metrics.set_duplex_reserved_sessions(stage=1, replica=0, reserved=3)
        metrics.set_duplex_reserved_sessions(stage=0, replica=0, reserved=0)

        for stage, replica, expected in [(0, 0, 0), (0, 1, 1), (1, 0, 3)]:
            assert (
                duplex_registry.get_sample_value(
                    defs.DUPLEX_RESERVED_SESSIONS,
                    {"model_name": "duplex-reserved-metrics", "stage": str(stage), "replica": str(replica)},
                )
                == expected
            )
        family = next(family for family in duplex_registry.collect() if family.name == defs.DUPLEX_RESERVED_SESSIONS)
        assert all(set(sample.labels) == set(defs.STAGE_LABELS) for sample in family.samples)

    def test_reject_reasons_are_bounded_and_collectors_are_shared(self, duplex_registry) -> None:
        metrics = OmniPrometheusMetrics(model_name="duplex-reject-metrics")
        other = OmniPrometheusMetrics(model_name="duplex-other-metrics")
        metrics.inc_duplex_admission_rejected("global_capacity_exhausted")
        metrics.inc_duplex_admission_rejected("replica_capacity_exhausted")
        metrics.inc_duplex_admission_rejected("arbitrary-session-id")
        metrics.inc_duplex_admission_rejected("another-session-id")
        other.inc_duplex_admission_rejected("replica_capacity_exhausted")

        for reason, expected in [("global_capacity_exhausted", 1), ("replica_capacity_exhausted", 1), ("unknown", 2)]:
            assert (
                duplex_registry.get_sample_value(
                    defs.DUPLEX_ADMISSION_REJECTED + "_total",
                    {"model_name": "duplex-reject-metrics", "reason": reason},
                )
                == expected
            )
        assert (
            duplex_registry.get_sample_value(
                defs.DUPLEX_ADMISSION_REJECTED + "_total",
                {"model_name": "duplex-other-metrics", "reason": "replica_capacity_exhausted"},
            )
            == 1
        )
        output = generate_latest(duplex_registry).decode()
        assert "session-id" not in output
        assert "session_id=" not in output

    def test_reserved_sessions_aggregate_live_frontend_owners(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        # Independent value classes model the mmap files of two processes,
        # without changing the backend used by existing global collectors.
        for pid, replica, reserved in [(101, 0, 3), (102, 1, 2)]:
            monkeypatch.setattr(values, "ValueClass", values.MultiProcessValue(lambda pid=pid: pid))
            monkeypatch.setattr(prom_module, "_duplex_families", None)
            registry = CollectorRegistry()
            monkeypatch.setattr(prom_module, "Gauge", partial(Gauge, registry=registry))
            monkeypatch.setattr(prom_module, "Counter", partial(Counter, registry=registry))
            reserved_family, rejected_family = prom_module._duplex_metric_families()
            # A frontend writes only its owned replica, including a zero on
            # release; it does not instantiate labels for another owner.
            reserved_family.labels(model_name="multi-frontend", stage="0", replica=str(replica)).set(reserved)
            reserved_family.labels(model_name="multi-frontend", stage="1", replica=str(replica)).set(0)
            rejected_family.labels(model_name="multi-frontend", reason="replica_capacity_exhausted").inc()

        aggregate = CollectorRegistry()
        multiprocess.MultiProcessCollector(aggregate, path=str(tmp_path))
        family = next(family for family in aggregate.collect() if family.name == defs.DUPLEX_RESERVED_SESSIONS)
        assert {(sample.labels["stage"], sample.labels["replica"]): sample.value for sample in family.samples} == {
            ("0", "0"): 3,
            ("1", "0"): 0,
            ("0", "1"): 2,
            ("1", "1"): 0,
        }
        assert all(set(sample.labels) == set(defs.STAGE_LABELS) for sample in family.samples)

        multiprocess.mark_process_dead(101, path=str(tmp_path))
        assert (
            aggregate.get_sample_value(
                defs.DUPLEX_RESERVED_SESSIONS, {"model_name": "multi-frontend", "stage": "0", "replica": "0"}
            )
            is None
        )
        assert (
            aggregate.get_sample_value(
                defs.DUPLEX_RESERVED_SESSIONS, {"model_name": "multi-frontend", "stage": "0", "replica": "1"}
            )
            == 2
        )
        assert (
            aggregate.get_sample_value(
                defs.DUPLEX_ADMISSION_REJECTED + "_total",
                {"model_name": "multi-frontend", "reason": "replica_capacity_exhausted"},
            )
            == 2
        )


class TestRequestLifecycleGauges:
    """Regression tests for the running/waiting gauge race at request finalize.

    Before the fix, the per-stage publish in OmniBase._process_single_result
    ran while the completed request was still in self.request_states even
    though the orchestrator had already decremented _running_counter — so a
    single completed request briefly exposed running=0, total=1 → waiting=1,
    which stuck on /metrics until another request triggered a refresh.

    Two-layer fix: _process_single_result excludes the finalizing request
    from `total`, and _log_summary_and_cleanup republishes gauges after the
    pop as a fallback. These tests pin the post-cleanup state.
    """

    def test_running_and_waiting_zero_after_request_completes(self, registry: CollectorRegistry) -> None:
        from types import SimpleNamespace

        from vllm_omni.entrypoints.omni_base import OmniBase
        from vllm_omni.metrics.prometheus import OmniPrometheusMetrics, OmniRequestCounter

        obj = object.__new__(OmniBase)
        obj.engine = SimpleNamespace(_running_counter=OmniRequestCounter())
        obj.prom_metrics = OmniPrometheusMetrics(model_name="lifecycle-test")
        obj.request_states = {}
        obj.log_stats = False

        # Simulate request lifecycle: start (counter 0→1, dict {} → {req}),
        # then finalize (counter 1→0, dict still holds the request because
        # _log_summary_and_cleanup hasn't run yet — this is the exact race
        # window the fix addresses).
        obj.engine._running_counter.increment()
        obj.request_states["req-1"] = SimpleNamespace(metrics=SimpleNamespace(e2e_done={"req-1"}))
        obj.engine._running_counter.decrement()

        obj._log_summary_and_cleanup("req-1")

        assert obj.engine._running_counter.value == 0
        assert len(obj.request_states) == 0
        out = generate_latest(registry).decode()
        assert _sample_value(out, 'vllm_omni:num_requests_running{model_name="lifecycle-test"}') == 0.0
        assert _sample_value(out, 'vllm_omni:num_requests_waiting{model_name="lifecycle-test"}') == 0.0

    def test_gauges_reflect_remaining_requests_after_one_completes(self, registry: CollectorRegistry) -> None:
        from types import SimpleNamespace

        from vllm_omni.entrypoints.omni_base import OmniBase
        from vllm_omni.metrics.prometheus import OmniPrometheusMetrics, OmniRequestCounter

        obj = object.__new__(OmniBase)
        obj.engine = SimpleNamespace(_running_counter=OmniRequestCounter())
        obj.prom_metrics = OmniPrometheusMetrics(model_name="lifecycle-test-2")
        obj.request_states = {}
        obj.log_stats = False

        # Two in flight, one finalizes — running should report 1, waiting 0.
        obj.engine._running_counter.increment()
        obj.engine._running_counter.increment()
        obj.request_states["req-1"] = SimpleNamespace(metrics=SimpleNamespace(e2e_done={"req-1"}))
        obj.request_states["req-2"] = SimpleNamespace(metrics=SimpleNamespace(e2e_done=set()))
        obj.engine._running_counter.decrement()

        obj._log_summary_and_cleanup("req-1")

        out = generate_latest(registry).decode()
        assert _sample_value(out, 'vllm_omni:num_requests_running{model_name="lifecycle-test-2"}') == 1.0
        assert _sample_value(out, 'vllm_omni:num_requests_waiting{model_name="lifecycle-test-2"}') == 0.0

    def test_engine_queued_requests_count_as_waiting(self, registry: CollectorRegistry) -> None:
        from types import SimpleNamespace

        from vllm_omni.entrypoints.omni_base import OmniBase
        from vllm_omni.metrics.prometheus import OmniPrometheusMetrics, OmniRequestCounter

        obj = object.__new__(OmniBase)
        obj.engine = SimpleNamespace(
            _running_counter=OmniRequestCounter(),
            _engines_waiting_counter=OmniRequestCounter(),
        )
        obj.prom_metrics = OmniPrometheusMetrics(model_name="lifecycle-test-3")
        obj.request_states = {}
        obj.log_stats = False

        # Three requests dispatched to a stage engine whose scheduler admits
        # one and queues two (e.g. a diffusion engine with
        # max_num_running_reqs=1). The orchestrator reports the queued count
        # via _engines_waiting_counter; they must show as waiting, not running.
        for _ in range(3):
            obj.engine._running_counter.increment()
        obj.engine._engines_waiting_counter.value = 2

        obj._publish_request_gauges(3)

        out = generate_latest(registry).decode()
        assert _sample_value(out, 'vllm_omni:num_requests_running{model_name="lifecycle-test-3"}') == 1.0
        assert _sample_value(out, 'vllm_omni:num_requests_waiting{model_name="lifecycle-test-3"}') == 2.0
