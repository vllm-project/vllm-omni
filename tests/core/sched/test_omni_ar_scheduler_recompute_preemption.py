# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduler contract tests for ``recompute_preemption: allow | fail``."""

from __future__ import annotations

from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import (
    RECOMPUTE_PREEMPTION_FAIL_MESSAGE,
    OmniARScheduler,
)
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MIXIN_UPDATE_HELPERS = (
    "_remove_stopped_requests_from_queues",
    "_handle_failed_kv_load_outputs",
    "_aggregate_kv_connector_stats",
    "_publish_kv_cache_events",
    "_attach_finished_request_sets",
    "_attach_scheduler_stats",
    "_capture_omni_connector_output",
    "_maybe_decode_pooling_output",
)


class _Request:
    def __init__(self, request_id: str, *, num_prompt_tokens: int = 4) -> None:
        self.request_id = request_id
        self.client_index = 0
        self.status = RequestStatus.RUNNING
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_prompt_tokens + 3
        self.num_output_placeholders = 0
        self.num_in_flight_tokens = 0
        self.num_stale_output_tokens = 0
        self.output_token_ids: list[int] = [1, 2]
        self._output_token_ids = self.output_token_ids
        self.spec_token_ids: list[int] = []
        self.drop_stale_output = False
        self.resumable = True
        self.stop_reason = None
        self.sampling_params = SimpleNamespace(num_logprobs=None)
        self.pooling_params = None
        self.has_encoder_inputs = False
        self.trace_headers = None
        self.num_nans_in_logits = 0

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self):
        return RequestStatus.get_finished_reason(self.status)

    def take_events(self):
        return None

    def take_prefill_stats(self):
        return None

    def record_event(self, *_args, **_kwargs) -> None:
        return None


class _RequestQueue(list):
    def prepend_request(self, request) -> None:
        self.insert(0, request)

    def remove_requests(self, requests) -> None:
        for request in requests:
            if request in self:
                self.remove(request)


def _make_scheduler(*, policy: str) -> OmniARScheduler:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(recompute_preemption=policy))
    scheduler.log_stats = False
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler._pending_recompute_preemption_error_requests = []
    scheduler._apply_recompute_preemption_fail = True
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue()
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_event_publisher = MagicMock()
    scheduler.connector = None
    scheduler.perf_metrics = None
    scheduler._new_prompt_len_snapshot = {}
    scheduler.structured_output_manager = MagicMock()
    scheduler.transfer_triggered_requests = set()
    scheduler.active_kv_transfers = set()
    scheduler.pending_stop_after_extraction = set()
    scheduler.waiting_for_transfer_free = set()
    for name in _MIXIN_UPDATE_HELPERS:
        setattr(scheduler, name, MethodType(getattr(OmniSchedulerMixin, name), scheduler))
    scheduler._cleanup_kv_tracking = MethodType(OmniARScheduler._cleanup_kv_tracking, scheduler)
    scheduler.make_spec_decoding_stats = lambda *args, **kwargs: None
    scheduler.make_stats = lambda *args, **kwargs: None
    scheduler._process_kv_transfer_trigger = lambda _request, _tokens: False
    scheduler._update_request_with_output = lambda _request, token_ids: (token_ids, False)
    scheduler._handle_stopped_request = lambda _request: True
    scheduler._free_request = lambda request: (None, None)
    return scheduler


def test_allow_delegates_to_upstream_preempt_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _Request("req-allow")
    scheduler = _make_scheduler(policy="allow")
    upstream_calls: list[str] = []

    def fake_preempt(self, req, timestamp, drop_stale_output=False):
        upstream_calls.append(req.request_id)

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "_preempt_request", fake_preempt)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)

    assert upstream_calls == ["req-allow"]
    assert scheduler._pending_recompute_preemption_error_requests == []


def test_fail_terminates_with_single_terminal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _Request("req-fail")
    scheduler = _make_scheduler(policy="fail")
    scheduler.requests = {request.request_id: request}
    finish_calls: list[tuple[list[str], RequestStatus]] = []

    def fake_finish(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        finish_calls.append((ids, finished_status))
        request.status = finished_status
        return [request]

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "finish_requests", fake_finish)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)

    assert finish_calls == [(["req-fail"], RequestStatus.FINISHED_ERROR)]
    assert request.stop_reason == RECOMPUTE_PREEMPTION_FAIL_MESSAGE
    assert len(scheduler._pending_recompute_preemption_error_requests) == 1


def test_fail_emits_exactly_one_error_output_and_never_requeues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _Request("req-fail")
    scheduler = _make_scheduler(policy="fail")
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]

    def fake_finish(self, request_ids, finished_status):
        request.status = finished_status
        scheduler.requests.pop(request.request_id, None)
        return [request]

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "finish_requests", fake_finish)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)
    assert request.request_id not in scheduler.waiting

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={},
        routed_experts=None,
    )

    first = OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)
    second = OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)

    first_outputs = first[0].outputs
    assert len(first_outputs) == 1
    assert first_outputs[0].request_id == "req-fail"
    assert first_outputs[0].finish_reason is FinishReason.ERROR
    assert first_outputs[0].stop_reason == RECOMPUTE_PREEMPTION_FAIL_MESSAGE
    second_outputs = second.get(request.client_index)
    assert second_outputs is None or second_outputs.outputs == []


def test_fail_prefill_only_still_allows_upstream_preempt(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _Request("req-prefill")
    request.output_token_ids = []
    request._output_token_ids = []
    request.num_computed_tokens = request.num_prompt_tokens
    scheduler = _make_scheduler(policy="fail")
    upstream_calls: list[str] = []

    def fake_preempt(self, req, timestamp, drop_stale_output=False):
        upstream_calls.append(req.request_id)

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "_preempt_request", fake_preempt)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)

    assert upstream_calls == ["req-prefill"]


def test_fail_empty_output_one_placeholder_is_decode_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async first-decode window: last prefill dispatched, one placeholder reserved."""
    request = _Request("req-async-first-decode")
    request.output_token_ids = []
    request._output_token_ids = []
    request.num_computed_tokens = request.num_prompt_tokens
    request.num_output_placeholders = 1
    scheduler = _make_scheduler(policy="fail")
    scheduler.requests = {request.request_id: request}
    finish_calls: list[tuple[list[str], RequestStatus]] = []

    def fake_finish(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        finish_calls.append((ids, finished_status))
        request.status = finished_status
        return [request]

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "finish_requests", fake_finish)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)

    assert finish_calls == [(["req-async-first-decode"], RequestStatus.FINISHED_ERROR)]
    assert request.stop_reason == RECOMPUTE_PREEMPTION_FAIL_MESSAGE


def test_fail_confirmed_past_prompt_without_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _Request("req-confirmed")
    request.output_token_ids = []
    request._output_token_ids = []
    request.num_computed_tokens = request.num_prompt_tokens + 1
    request.num_output_placeholders = 0
    scheduler = _make_scheduler(policy="fail")
    scheduler.requests = {request.request_id: request}
    finish_calls: list[tuple[list[str], RequestStatus]] = []

    def fake_finish(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        finish_calls.append((ids, finished_status))
        request.status = finished_status
        return [request]

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "finish_requests", fake_finish)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)

    assert finish_calls == [(["req-confirmed"], RequestStatus.FINISHED_ERROR)]
    assert request.stop_reason == RECOMPUTE_PREEMPTION_FAIL_MESSAGE


def test_fail_discards_in_flight_async_output(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _Request("req-async")
    request.num_in_flight_tokens = 2
    request.num_output_placeholders = 1
    scheduler = _make_scheduler(policy="fail")
    scheduler.requests = {request.request_id: request}

    def fake_finish(self, request_ids, finished_status):
        request.status = finished_status
        return [request]

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "finish_requests", fake_finish)

    OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)

    assert request.num_stale_output_tokens == 2
    assert request.num_output_placeholders == 0

    update_calls: list[str] = []

    def track_update(req, token_ids):
        update_calls.append(req.request_id)
        return token_ids, False

    scheduler._update_request_with_output = track_update

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-async": 1},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[9]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={"req-async": 0},
        routed_experts=None,
    )

    OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)

    assert update_calls == []


def test_reset_prefix_cache_does_not_use_fail_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _Request("req-reset")
    scheduler = _make_scheduler(policy="fail")
    upstream_calls: list[str] = []

    def fake_preempt(self, req, timestamp, drop_stale_output=False):
        upstream_calls.append(req.request_id)

    def fake_reset(self, *args, **kwargs):
        OmniARScheduler._preempt_request(scheduler, request, timestamp=0.0)
        return True

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "_preempt_request", fake_preempt)
    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "reset_prefix_cache", fake_reset)

    assert OmniARScheduler.reset_prefix_cache(scheduler, reset_running_requests=True) is True
    assert upstream_calls == ["req-reset"]
    assert scheduler._pending_recompute_preemption_error_requests == []
    assert request.stop_reason is None
    assert scheduler._apply_recompute_preemption_fail is True
