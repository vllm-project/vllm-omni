# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as scheduler_mod

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MockQueue:
    def __init__(self, items: list | None = None) -> None:
        self._items = list(items or [])

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add_request(self, request) -> None:
        self._items.append(request)

    def prepend_requests(self, requests) -> None:
        self._items = list(requests) + self._items


class _MockRequest:
    def __init__(
        self,
        request_id: str,
        *,
        status: RequestStatus = RequestStatus.RUNNING,
        num_prompt_tokens: int = 4,
        num_computed_tokens: int = 4,
        num_output_placeholders: int = 0,
    ) -> None:
        self.request_id = request_id
        self.status = status
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_output_placeholders = num_output_placeholders

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


def _make_scheduler():
    sched = scheduler_mod.OmniARAsyncScheduler.__new__(scheduler_mod.OmniARAsyncScheduler)
    sched._should_defer_waiting_admission = lambda: True
    sched.kv_holding_waiting = _MockQueue()
    sched.deferred_waiting = set()
    return sched


def test_unified_decode_graph_deferral_restores_waiting_queue(monkeypatch) -> None:
    scheduler = _make_scheduler()
    scheduler.max_num_active_reqs = 8
    scheduler.max_num_running_reqs = 8
    scheduler.running = [_MockRequest("decode")]
    original_waiting_req = _MockRequest("waiting", status=RequestStatus.WAITING)
    deferred_by_upstream = _MockRequest("deferred-by-upstream", status=RequestStatus.WAITING)
    original_waiting = _MockQueue([original_waiting_req])
    scheduler.waiting = original_waiting
    kv_holder = _MockRequest("kv-holder", status=RequestStatus.WAITING)
    original_kv_holding = _MockQueue([kv_holder])
    scheduler.kv_holding_waiting = original_kv_holding
    preempted_by_upstream = _MockRequest("preempted-by-upstream", status=RequestStatus.PREEMPTED)
    scheduler.policy = "fcfs"
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler._consume_pending_connector_output = lambda model_mode: None

    monkeypatch.setattr(scheduler_mod, "create_request_queue", lambda _policy: _MockQueue())

    def fake_upstream_schedule(self, throttle_prefills: bool = False):
        assert self.waiting is not original_waiting
        assert not self.waiting
        assert self.kv_holding_waiting is not original_kv_holding
        assert not self.kv_holding_waiting
        self.waiting.add_request(deferred_by_upstream)
        self.kv_holding_waiting.add_request(preempted_by_upstream)
        raise RuntimeError("stop before output wrapping")

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "schedule", fake_upstream_schedule)

    with pytest.raises(RuntimeError, match="stop before output wrapping"):
        scheduler.schedule()

    assert scheduler.waiting._items == [deferred_by_upstream, original_waiting_req]
    assert scheduler.waiting is original_waiting
    assert scheduler.kv_holding_waiting is original_kv_holding
    assert scheduler.kv_holding_waiting._items == [preempted_by_upstream, kv_holder]
