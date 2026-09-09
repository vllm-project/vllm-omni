# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify emit-call wiring in production code paths.

Uses behavior and Prometheus exposition to verify production call semantics.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from vllm_omni.metrics import OmniPrometheusMetrics
from vllm_omni.metrics import definitions as defs

if TYPE_CHECKING:
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "emit-calls-test"


# ---------------------------------------------------------------------------
# Behavioral pins — verify call semantics with mock OmniPrometheusMetrics.
# ---------------------------------------------------------------------------


def _make_omni_base_with_mock_prom(mocker):
    """Build a minimal OmniBase shell wired with a mock prom_metrics.

    Mirrors the pattern in ``test_prometheus.py::TestRequestLifecycleGauges``
    — ``object.__new__`` skips ``__init__`` so we don't need a real engine.
    """
    from vllm_omni.entrypoints.omni_base import OmniBase

    obj = object.__new__(OmniBase)
    obj.prom_metrics = mocker.Mock(spec=OmniPrometheusMetrics)
    obj.request_states = {}
    obj._consumed_metric_messages = {}
    obj.log_stats = True
    return obj, obj.prom_metrics


class TestFailureCounterWiring:
    """All failure paths share one request-idempotent counter helper."""

    def test_failure_reason_taxonomy_is_bounded(self) -> None:
        from vllm_omni.metrics.utils import normalize_failure_reason

        assert normalize_failure_reason("client_abort") == "client_abort"
        assert normalize_failure_reason("client_disconnect") == "client_disconnect"
        assert normalize_failure_reason("stage_error") == "stage_error"
        assert normalize_failure_reason("unknown") == "unknown"
        assert normalize_failure_reason("timeout for request req-1") == "unknown"
        assert normalize_failure_reason("") == "unknown"
        assert normalize_failure_reason(None) == "unknown"

    def test_fire_failure_counter_passes_reason_through(self, mocker) -> None:
        obj, prom = _make_omni_base_with_mock_prom(mocker)
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(e2e_done=set()),
            failure_recorded=False,
        )

        obj._record_request_failure_once("req-1", reason="client_disconnect")

        prom.request_failed.assert_called_once()
        prom.inc_requests_failed.assert_called_once_with("client_disconnect")

    def test_fire_failure_counter_normalizes_unrecognized_reason(self, mocker) -> None:
        obj, prom = _make_omni_base_with_mock_prom(mocker)
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(e2e_done=set()),
            failure_recorded=False,
        )

        obj._record_request_failure_once("req-1", reason="CUDA error for req-1")

        prom.inc_requests_failed.assert_called_once_with("unknown")

    def test_fire_failure_counter_skips_when_request_already_succeeded(self, mocker) -> None:
        # When the request is in e2e_done (i.e. finalize already fired
        # request_succeeded), the failure path must NOT double-count.
        obj, prom = _make_omni_base_with_mock_prom(mocker)
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(e2e_done={"req-1"}),
            failure_recorded=False,
        )

        obj._record_request_failure_once("req-1", reason="client_disconnect")

        prom.request_failed.assert_not_called()
        prom.inc_requests_failed.assert_not_called()

    def test_fire_failure_counter_skips_when_request_state_missing(self, mocker) -> None:
        # No request_states entry — already popped by abort path. Fail-safe
        # must NOT raise.
        obj, prom = _make_omni_base_with_mock_prom(mocker)

        obj._record_request_failure_once("missing-req", reason="oom")

        prom.request_failed.assert_not_called()
        prom.inc_requests_failed.assert_not_called()

    def test_failure_counter_is_idempotent_across_abort_and_cleanup(self, mocker) -> None:
        obj, prom = _make_omni_base_with_mock_prom(mocker)
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(
                e2e_done=set(),
                build_and_log_summary=lambda: None,
            ),
            failure_recorded=False,
        )

        obj._record_request_failure_once("req-1", reason="client_disconnect")
        obj._log_summary_and_cleanup("req-1", reason="stage_error")

        prom.request_failed.assert_called_once()
        prom.inc_requests_failed.assert_called_once_with("client_disconnect")

    def test_log_summary_and_cleanup_default_reason(self, mocker) -> None:
        obj, prom = _make_omni_base_with_mock_prom(mocker)
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(
                e2e_done=set(),
                build_and_log_summary=lambda: None,
            ),
            failure_recorded=False,
        )

        obj._log_summary_and_cleanup("req-1")

        prom.inc_requests_failed.assert_called_once_with("stage_error")


class TestEarlyReturnOnLogStatsOff:
    """When --log-stats is off (default), observe methods must be silent no-ops.

    This is the gating contract — emit sites can call unconditionally and the
    helper short-circuits.
    """

    def test_observe_methods_silent_when_log_stats_false(self) -> None:
        prom = OmniPrometheusMetrics(model_name=_MODEL, log_stats=False)
        # None of these should raise or write to the registry.
        prom.observe_stage_gen_time(stage=0, stage_type="llm", gen_time_s=1.5)
        prom.observe_stage_in_queue(stage=0, in_queue_s=0.2)
        prom.observe_queue_wait(queue_wait_s=0.5)
        prom.set_stage_waiting_requests(stage=0, n_waiting=3)
        prom.observe_num_inference_steps(n_steps=20)
        prom.inc_image_count(n_images=1)
        prom.observe_image_pixels(n_pixels=512 * 512)
        prom.set_peak_memory(stage=0, peak_memory_mb=1024.0)
        prom.inc_requests_failed(reason="oom")
        prom.observe_kv_wait(connector_type="shm", kv_wait_s=0.01)


class TestStageGenerationLabels:
    def test_stage_gen_time_exposes_bounded_stage_type(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        model = _MODEL + "-stage-type"
        prom = OmniPrometheusMetrics(model_name=model, log_stats=True)
        prom.observe_stage_gen_time(stage=0, stage_type="llm", gen_time_s=0.25)
        prom.observe_stage_gen_time(stage=1, stage_type="diffusion", gen_time_s=1.5)

        out = generate_latest(REGISTRY).decode()
        assert f'{defs.STAGE_GEN_TIME_S}_count{{model_name="{model}",stage="0",stage_type="llm"}} 1.0' in out
        assert f'{defs.STAGE_GEN_TIME_S}_count{{model_name="{model}",stage="1",stage_type="diffusion"}} 1.0' in out


class TestCounterExpositionNames:
    def test_counter_families_have_exactly_one_total_suffix(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        model = _MODEL + "-counter-suffix"
        prom = OmniPrometheusMetrics(model_name=model, log_stats=True)
        prom.inc_image_count(2)
        prom.inc_requests_failed("stage_error")

        out = generate_latest(REGISTRY).decode()
        assert f'{defs.IMAGE_COUNT_METRIC}_total{{model_name="{model}"}} 2.0' in out
        assert f'{defs.REQUESTS_FAILED}_total{{model_name="{model}",reason="stage_error"}} 1.0' in out
        assert f"{defs.IMAGE_COUNT_METRIC}_total_total" not in out
        assert f"{defs.REQUESTS_FAILED}_total_total" not in out


class TestQueueWaitExtraction:
    def test_present_zero_is_a_valid_observation(self) -> None:
        from vllm_omni.metrics.utils import extract_queue_wait_s

        assert extract_queue_wait_s({"queue_wait_ms": 0.0}) == 0.0

    def test_missing_queue_wait_is_not_synthetic_zero(self) -> None:
        from vllm_omni.metrics.utils import extract_queue_wait_s

        assert extract_queue_wait_s({"preprocess_ms": 2.0}) is None
        assert extract_queue_wait_s(None) is None


class TestStageInQueueObservation:
    def test_prometheus_records_valid_zero_wait(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        model = _MODEL + "-zero-stage-wait"
        prom = OmniPrometheusMetrics(model_name=model, log_stats=True)

        prom.observe_stage_in_queue(stage=1, in_queue_s=0.0)

        out = generate_latest(REGISTRY).decode()
        sample = f'{defs.STAGE_IN_QUEUE_S}_count{{model_name="{model}",stage="1"}} 1.0'
        assert sample in out


class TestStageWorkloadMetricScope:
    @staticmethod
    def _observe(mocker, *, stage_type: str, output_unit_type: str):
        from vllm_omni.metrics.utils import observe_stage_workload_metrics

        prom = mocker.Mock(spec=OmniPrometheusMetrics)
        stage_metrics = SimpleNamespace(
            num_inference_steps=50,
            output_unit_type=output_unit_type,
            image_pixels=1024 * 1024,
            output_unit_count=2,
        )
        observe_stage_workload_metrics(
            prom,
            stage_type=stage_type,
            stage_metrics=stage_metrics,
        )
        return prom

    def test_diffusion_image_observes_steps_and_image_workload(self, mocker) -> None:
        prom = self._observe(mocker, stage_type="diffusion", output_unit_type="image")

        prom.observe_num_inference_steps.assert_called_once_with(50)
        prom.observe_image_pixels.assert_called_once_with(1024 * 1024)
        prom.inc_image_count.assert_called_once_with(2)

    def test_diffusion_video_observes_steps_only(self, mocker) -> None:
        prom = self._observe(mocker, stage_type="diffusion", output_unit_type="video")

        prom.observe_num_inference_steps.assert_called_once_with(50)
        prom.observe_image_pixels.assert_not_called()
        prom.inc_image_count.assert_not_called()

    def test_llm_text_observes_no_workload_metrics(self, mocker) -> None:
        prom = self._observe(mocker, stage_type="llm", output_unit_type="text")

        prom.observe_num_inference_steps.assert_not_called()
        prom.observe_image_pixels.assert_not_called()
        prom.inc_image_count.assert_not_called()

    def test_non_diffusion_image_observes_image_workload_only(self, mocker) -> None:
        prom = self._observe(mocker, stage_type="llm", output_unit_type="image")

        prom.observe_num_inference_steps.assert_not_called()
        prom.observe_image_pixels.assert_called_once_with(1024 * 1024)
        prom.inc_image_count.assert_called_once_with(2)

    @pytest.mark.parametrize("peak_memory_mb", [2048.0, 0.0])
    def test_same_finished_image_message_is_observed_exactly_once(self, mocker, peak_memory_mb: float) -> None:
        from vllm_omni.entrypoints.omni_base import OmniBase

        obj = object.__new__(OmniBase)
        obj._enable_ar_profiler = False
        obj._consumed_metric_messages = {}
        obj.prom_metrics = mocker.Mock(spec=OmniPrometheusMetrics)
        obj.mod_metrics = mocker.Mock()
        obj.engine = SimpleNamespace(
            get_stage_metadata=lambda _stage_id: SimpleNamespace(
                stage_type="diffusion",
                final_output=False,
                final_output_type="image",
            )
        )
        stage_metrics = SimpleNamespace(
            stage_gen_time_ms=1000.0,
            diffusion_metrics={"diffusion_engine_exec_time_s": 0.8, "scheduler_queue_wait_s": 0.1},
            num_inference_steps=20,
            output_unit_type="image",
            image_pixels=1024 * 1024,
            output_unit_count=1,
            serving_time_to_first_output_ms=250.0,
            image_time_to_first_output_ms=250.0,
            pipeline_timings={},
        )
        result = SimpleNamespace(
            request_id="req-replay",
            stage_id=1,
            replica_id=0,
            stage_submit_ts=10.0,
            metrics=stage_metrics,
            engine_outputs=SimpleNamespace(
                stage_durations={},
                peak_memory_mb=peak_memory_mb,
                finished=True,
                final_output_type="image",
            ),
        )
        aggregator = mocker.Mock()
        aggregator.stage_events = {}
        aggregator.stage_first_ts = [None, None]
        aggregator.stage_last_ts = [None, None]

        for _ in range(2):
            assert obj._process_single_result(result, 1, aggregator, {}, 0.0, 1) is None

        aggregator.on_stage_metrics.assert_called_once_with(1, "req-replay", stage_metrics, "image")
        obj.prom_metrics.inc_image_count.assert_called_once_with(1)
        obj.prom_metrics.observe_image_pixels.assert_called_once_with(1024 * 1024)
        obj.prom_metrics.observe_num_inference_steps.assert_called_once_with(20)
        obj.prom_metrics.observe_stage_in_queue.assert_called_once_with(1, 0.1)
        if peak_memory_mb > 0:
            obj.prom_metrics.set_peak_memory.assert_called_once_with(1, peak_memory_mb)
        else:
            obj.prom_metrics.set_peak_memory.assert_not_called()
        obj.mod_metrics.observe_image_ttfp.assert_called_once_with("1", "0", 0.25)


class TestStageWaitingAggregation:
    @staticmethod
    def _make_orchestrator(mocker):
        from vllm_omni.engine.orchestrator import Orchestrator

        orchestrator = object.__new__(Orchestrator)
        orchestrator._prom_metrics = mocker.Mock()
        orchestrator._stage_replica_waiting = {}
        return orchestrator

    def test_multiple_replicas_are_summed(self, mocker) -> None:
        orchestrator = self._make_orchestrator(mocker)

        orchestrator._update_stage_replica_waiting(1, 0, 3)
        orchestrator._update_stage_replica_waiting(1, 1, 5)

        orchestrator._prom_metrics.set_stage_waiting_requests.assert_called_with(1, 8)

    def test_zero_update_and_stage_isolation(self, mocker) -> None:
        orchestrator = self._make_orchestrator(mocker)

        orchestrator._update_stage_replica_waiting(1, 0, 3)
        orchestrator._update_stage_replica_waiting(2, 0, 7)
        orchestrator._update_stage_replica_waiting(1, 0, 0)

        orchestrator._prom_metrics.set_stage_waiting_requests.assert_called_with(1, 0)
        assert orchestrator._stage_replica_waiting[(2, 0)] == 7

    def test_dead_replica_snapshot_is_removed(self, mocker) -> None:
        orchestrator = self._make_orchestrator(mocker)
        orchestrator._update_stage_replica_waiting(1, 0, 3)
        orchestrator._update_stage_replica_waiting(1, 1, 5)

        orchestrator._remove_stage_replica_waiting(1, 1)

        orchestrator._prom_metrics.set_stage_waiting_requests.assert_called_with(1, 3)


# ---------------------------------------------------------------------------
# Behavioral pin: call sites now scope workload observations by stage/output
# type. Keep the Prometheus wrapper's <= 0 guard as a second line of defense.
# ---------------------------------------------------------------------------


class TestZeroGuardContract:
    """Zero-valued observations must not bump histogram ``_count``."""

    def test_zero_image_pixels_not_observed(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        prom = OmniPrometheusMetrics(model_name=_MODEL + "-zero-guard")
        prom.observe_image_pixels(n_pixels=0)
        out = generate_latest(REGISTRY).decode()
        # ``.labels()`` in OmniPrometheusMetrics.__init__ already creates the
        # child sample, so the ``_count`` line exists with value 0.0 even
        # without observations. The guard prevents ``_count`` from bumping
        # to 1.0 — assert the value stays at 0.
        needle = f'vllm_omni:image_pixels_count{{model_name="{_MODEL}-zero-guard"}}'
        count_lines = [ln for ln in out.splitlines() if ln.startswith(needle)]
        assert count_lines, "expected image_pixels_count line to exist after OmniPrometheusMetrics construction"
        assert float(count_lines[0].split()[-1]) == 0.0, (
            "zero-pixel observation leaked to registry — guard should early-return on <= 0"
        )

    def test_zero_num_inference_steps_not_observed(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        prom = OmniPrometheusMetrics(model_name=_MODEL + "-zero-steps")
        prom.observe_num_inference_steps(n_steps=0)
        out = generate_latest(REGISTRY).decode()
        needle = f'vllm_omni:num_inference_steps_count{{model_name="{_MODEL}-zero-steps"}}'
        count_lines = [ln for ln in out.splitlines() if ln.startswith(needle)]
        assert count_lines, "expected num_inference_steps_count line to exist after construction"
        assert float(count_lines[0].split()[-1]) == 0.0, (
            "zero-step observation leaked to registry — guard should early-return on <= 0"
        )

    def test_positive_image_pixels_observed(self) -> None:
        # Sanity check: positive values DO get observed.
        from prometheus_client import REGISTRY, generate_latest

        prom = OmniPrometheusMetrics(model_name=_MODEL + "-pos-guard")
        prom.observe_image_pixels(n_pixels=512)
        out = generate_latest(REGISTRY).decode()
        needle = f'vllm_omni:image_pixels_count{{model_name="{_MODEL}-pos-guard"}}'
        count_lines = [ln for ln in out.splitlines() if ln.startswith(needle)]
        assert count_lines, "expected image_pixels_count line after positive observation"
        assert float(count_lines[0].split()[-1]) == 1.0, "positive-pixel observation did not increment count to 1"


# ---------------------------------------------------------------------------
# KV-wait emit wiring — scheduler ENTER/EXIT lifecycle + orchestrator dispatch.
# ---------------------------------------------------------------------------


def _make_scheduler_shell() -> OmniARScheduler:
    """Minimal OmniARScheduler shell — skips upstream __init__ via object.__new__."""
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    obj = object.__new__(OmniARScheduler)
    obj.requests = {}
    obj.running = []
    obj.waiting = []
    obj._kv_wait_start_ts = {}
    obj._omni_kv_config = None
    return obj


class TestKvWaitSchedulerEmit:
    """Scheduler-side half of kv_wait_s: pops start ts, carries the wait across."""

    def test_skips_when_no_start_ts_recorded(self) -> None:
        sched = _make_scheduler_shell()
        outputs: dict[int, list] = {}

        sched._emit_kv_wait_output(outputs, "req-no-wait", req=SimpleNamespace(client_index=0))

        assert outputs == {}, "emit must not append output when no start ts recorded"
        assert "req-no-wait" not in sched._kv_wait_start_ts

    def test_emits_wait_duration_and_connector_type(self) -> None:
        sched = _make_scheduler_shell()

        sched._kv_wait_start_ts["req-1"] = time.monotonic() - 0.25
        outputs: dict[int, list] = {}
        live_req = SimpleNamespace(client_index=3)

        sched._emit_kv_wait_output(outputs, "req-1", req=live_req)

        assert "req-1" not in sched._kv_wait_start_ts, "start ts must be popped after emit"
        assert 3 in outputs and len(outputs[3]) == 1
        params = outputs[3][0].kv_transfer_params
        assert "kv_wait_s" in params and "connector_type" in params
        assert 0.0 < params["kv_wait_s"] < 1.0
        assert params["connector_type"] == "unknown"

    def test_connector_type_resolved_from_omni_kv_config(self) -> None:
        sched = _make_scheduler_shell()
        sched._omni_kv_config = {"connector_config": {"type": "SharedMemoryConnector"}}

        sched._kv_wait_start_ts["req-3"] = time.monotonic() - 0.01
        outputs: dict[int, list] = {}

        sched._emit_kv_wait_output(outputs, "req-3", req=SimpleNamespace(client_index=0))

        eco = outputs[0][0]
        assert eco.kv_transfer_params["connector_type"] == "SharedMemoryConnector"


class TestKvWaitTerminalCleanup:
    def test_finish_requests_clears_requested_wait_only(self) -> None:
        from vllm.v1.request import RequestStatus

        scheduler = _make_scheduler_shell()
        scheduler._kv_wait_start_ts = {"req-1": 10.0, "req-2": 20.0}

        finished = scheduler.finish_requests(
            (request_id for request_id in ["req-1"]),
            RequestStatus.FINISHED_ABORTED,
        )

        assert finished == []
        assert "req-1" not in scheduler._kv_wait_start_ts
        assert scheduler._kv_wait_start_ts["req-2"] == 20.0

    def test_cleanup_is_idempotent_when_timestamp_is_already_gone(self) -> None:
        from vllm.v1.request import RequestStatus

        scheduler = _make_scheduler_shell()

        scheduler.finish_requests("req-already-emitted", RequestStatus.FINISHED_ABORTED)

        assert scheduler._kv_wait_start_ts == {}
