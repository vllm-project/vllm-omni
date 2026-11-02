# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""V1 coalescing preserves scheduler state until a batch can execute."""

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import RequestStatus

from tests.core.sched.test_generation_scheduler_atomic_chunk import (
    _make_request,
    _make_scheduler,
    _stage_running_chunk,
    _stage_waiting_chunk,
)
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler, VLLMScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def scheduler(monkeypatch, queue_name="waiting"):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_generation_scheduler.time.monotonic", lambda: clock[0])
    s = _make_scheduler(budget=32)
    s._generation_min_batch_size = 4
    s._generation_max_wait_s = 0.012
    s._generation_pending_inputs = set()
    for i in range(4):
        s.requests[str(i)] = _make_request(str(i), 8)
    stage = _stage_running_chunk if queue_name == "running" else _stage_waiting_chunk
    stage(s, s.requests["0"])
    return s, clock


@pytest.mark.parametrize("queue_name", ["running", "waiting", "kv_holding_waiting"])
def test_v1_wait_does_not_allocate_or_consume_input(monkeypatch, mocker, queue_name):
    s, clock = scheduler(monkeypatch, queue_name)
    if queue_name == "kv_holding_waiting":
        s.waiting.remove_requests([s.requests["0"]])
        s.kv_holding_waiting.add_request(s.requests["0"])
    allocate = mocker.spy(s.kv_cache_manager, "allocate_slots")
    request = s.requests["0"]
    payload = request.additional_information = {"codes": {"audio": [1, 2]}}
    before_status = request.status
    output = s.schedule()
    assert output.num_scheduled_tokens == {}
    allocate.assert_not_called()
    assert s.chunk_transfer_adapter.requests_with_ready_chunks == {"0"}
    assert request.additional_information is payload
    assert request.num_computed_tokens == request.num_in_flight_tokens == 0
    assert request.status == before_status
    assert list(getattr(s, queue_name)) == [request]
    clock[0] += 0.006
    assert s.schedule().num_scheduled_tokens == {}
    assert s._generation_batch_deadline == pytest.approx(10.012)
    clock[0] += 0.006
    assert s.schedule().num_scheduled_tokens == {"0": 8}
    assert s._generation_batch_deadline is None


def test_v1_releases_when_more_chunks_arrive(monkeypatch):
    s, clock = scheduler(monkeypatch)
    assert not s.schedule().num_scheduled_tokens
    clock[0] += 0.003
    for i in range(1, 4):
        _stage_waiting_chunk(s, s.requests[str(i)])
    assert s.schedule().num_scheduled_tokens == {str(i): 8 for i in range(4)}
    assert s._generation_batch_deadline is None


def test_v1_cancelled_request_is_not_restored(monkeypatch):
    s, clock = scheduler(monkeypatch)
    assert not s.schedule().num_scheduled_tokens
    s.finish_requests({"0"}, RequestStatus.FINISHED_ABORTED)
    clock[0] += 0.012
    assert not s.schedule().num_scheduled_tokens
    assert "0" not in s.requests
    assert not s.running and not s.waiting
    assert s._generation_batch_deadline is None


@pytest.mark.parametrize("reason", ["disabled", "low_concurrency", "pending", "terminal"])
def test_v1_immediate_progress(monkeypatch, reason):
    s, _ = scheduler(monkeypatch)
    if reason == "disabled":
        s._generation_max_wait_s = 0
    elif reason == "low_concurrency":
        s.requests.pop("3")
    elif reason == "pending":
        s._generation_pending_inputs.add("0")
    else:
        # A terminal input participates in the same bounded window.
        s.chunk_transfer_adapter.is_done_receiving_chunks = lambda request_id: request_id == "0"
        assert not s.schedule().num_scheduled_tokens
        s._generation_min_batch_size = 1
    assert s.schedule().num_scheduled_tokens == {"0": 8}


def test_v1_retains_in_flight_readiness_until_output_retires(monkeypatch):
    s, clock = scheduler(monkeypatch)
    s._supports_pending_generation = True
    s.requests["0"].num_in_flight_tokens = 8
    assert not s.schedule().num_scheduled_tokens
    assert s.chunk_transfer_adapter.requests_with_ready_chunks == {"0"}
    s.requests["0"].num_in_flight_tokens = 0
    assert not s.schedule().num_scheduled_tokens
    clock[0] += 0.012
    assert s.schedule().num_scheduled_tokens == {"0": 8}


def test_v1_pause_does_not_keep_a_stale_deadline(monkeypatch):
    s, clock = scheduler(monkeypatch)
    assert not s.schedule().num_scheduled_tokens
    s._pause_state = PauseState.PAUSED_ALL
    assert not s.schedule().num_scheduled_tokens
    assert s._generation_batch_deadline is None
    clock[0] += 1
    s._pause_state = PauseState.UNPAUSED
    assert not s.schedule().num_scheduled_tokens
    assert s._generation_batch_deadline == pytest.approx(11.012)


def test_free_request_clears_pending_input_ownership(monkeypatch):
    s, _ = scheduler(monkeypatch)
    s._generation_pending_inputs.add("0")
    monkeypatch.setattr(VLLMScheduler, "_free_request", lambda *args: (None, None))
    OmniGenerationScheduler._free_request(s, s.requests["0"])
    assert not s._generation_pending_inputs
