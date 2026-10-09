# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded readiness coalescing must not allocate or consume request state."""

import queue
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_idle_wait_wakes_on_input_without_extending_coalescing_deadline(monkeypatch):
    events = [notification("0"), notification("1", "2", "3")]
    s, clock = scheduler(monkeypatch, delayed=[(0.004, events[0]), (0.002, events[1])])
    s._generation_coalescing_policy = "idle_wait"
    for request in s.requests.values():
        request.status = RequestStatus.WAITING_FOR_CHUNK
    assert s._drain_omni_connector_outputs() == events
    assert clock[0] == pytest.approx(10.006)
    assert s._omni_connector_output_inbox.waits == pytest.approx([0.012, 0.012])


@pytest.mark.parametrize("pending", ["in_flight", "admission", "registration", "terminal", "empty"])
def test_idle_wait_preserves_completion_and_control_progress(monkeypatch, pending):
    s, clock = scheduler(monkeypatch)
    s._generation_coalescing_policy = "idle_wait"
    for request in s.requests.values():
        request.status = RequestStatus.WAITING_FOR_CHUNK
    if pending == "in_flight":
        s.requests["0"].num_in_flight_tokens = 1
    elif pending == "admission":
        s.requests["0"].status = RequestStatus.WAITING
    elif pending == "registration":
        s.input_coordinator.pending_chunk_registrations = [object()]
    elif pending == "terminal":
        s._pending_data_plane_terminal_req_ids = {"0"}
    else:
        s.requests.clear()
    assert s._drain_omni_connector_outputs() == []
    assert clock[0] == 10
    assert not s._omni_connector_output_inbox.waits


def test_idle_wait_without_arrivals_is_bounded(monkeypatch):
    s, clock = scheduler(monkeypatch)
    s._generation_coalescing_policy = "idle_wait"
    for request in s.requests.values():
        request.status = RequestStatus.WAITING_FOR_CHUNK
    assert s._drain_omni_connector_outputs() == []
    assert clock[0] == pytest.approx(10.012)


def notification(*ids, terminal=()):
    return OmniConnectorOutput(chunk_ready_req_ids=set(ids), chunk_finished_req_ids=set(terminal))


class Inbox:
    def __init__(self, clock, initial=(), delayed=()):
        self.clock = clock
        self.initial = list(initial)
        self.delayed = list(delayed)
        self.waits = []

    def get_nowait(self):
        if self.initial:
            return self.initial.pop(0)
        raise queue.Empty

    def get(self, timeout):
        self.waits.append(timeout)
        if not self.delayed or self.delayed[0][0] > timeout:
            self.clock[0] += timeout
            raise queue.Empty
        delay, value = self.delayed.pop(0)
        self.clock[0] += delay
        return value


def scheduler(monkeypatch, initial=(), delayed=()):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_generation_scheduler.time.monotonic", lambda: clock[0])
    s = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    s._generation_max_wait_s = 0.012
    s._generation_min_batch_size = 4
    s._pause_state = PauseState.UNPAUSED
    s.requests = {
        str(i): SimpleNamespace(request_id=str(i), num_in_flight_tokens=0, is_finished=lambda: False) for i in range(8)
    }
    s.running = list(s.requests.values())
    s.input_coordinator = SimpleNamespace(
        requests_with_ready_chunks=set(), finished_requests=set(), input_terminal_req_ids=set()
    )
    s._latest_omni_connector_output = None
    s._omni_connector_output_inbox = Inbox(clock, initial, delayed)
    return s, clock


def test_arrivals_form_target_batch_in_order(monkeypatch):
    events = [notification("0"), notification("1", "2"), notification("3")]
    s, clock = scheduler(monkeypatch, events[:1], [(0.002, events[1]), (0.003, events[2])])
    assert s._drain_omni_connector_outputs() == events
    assert clock[0] == pytest.approx(10.005)
    assert not s.input_coordinator.requests_with_ready_chunks
    assert all(r.num_in_flight_tokens == 0 for r in s.requests.values())


def test_duplicate_stale_and_finished_notifications_do_not_fill_batch(monkeypatch):
    events = [notification("0", "gone", "1", "2"), notification("0")]
    s, clock = scheduler(monkeypatch, events[:1], [(0.003, events[1])])
    s.requests["1"].is_finished = lambda: True
    s.requests["2"].is_finished = lambda: True
    assert s._drain_omni_connector_outputs() == events
    assert clock[0] == pytest.approx(10.012)
    assert s._omni_connector_output_inbox.waits == pytest.approx([0.012, 0.009])


@pytest.mark.parametrize("reason", ["enough", "empty", "low", "paused", "off"])
def test_immediate_dispatch(monkeypatch, reason):
    event = notification("0")
    s, clock = scheduler(monkeypatch, [event])
    if reason == "enough":
        event.chunk_ready_req_ids.update(["1", "2", "3"])
    elif reason == "empty":
        event.chunk_ready_req_ids.clear()
    elif reason == "low":
        s.requests = {"0": s.requests["0"]}
    elif reason == "paused":
        s._pause_state = PauseState.PAUSED_ALL
    elif reason == "off":
        s._generation_max_wait_s = 0
    assert s._drain_omni_connector_outputs() == [event]
    assert not s._omni_connector_output_inbox.waits
    assert clock[0] == 10


def test_terminal_arrival_keeps_original_deadline(monkeypatch):
    events = [notification("0"), notification("1", terminal=["1"])]
    s, clock = scheduler(monkeypatch, events[:1], [(0.002, events[1])])
    assert s._drain_omni_connector_outputs() == events
    assert clock[0] == pytest.approx(10.012)


def test_pending_ready_and_legacy_notification_preserved(monkeypatch):
    s, _ = scheduler(monkeypatch)
    s.input_coordinator.requests_with_ready_chunks.update(["0", "1", "2"])
    last = notification("3")
    s._latest_omni_connector_output = last
    assert s._drain_omni_connector_outputs() == [last]
    assert s._latest_omni_connector_output is None
    assert not s._omni_connector_output_inbox.waits


def test_deadline_is_new_for_each_batch_not_extended_by_arrivals(monkeypatch):
    s, clock = scheduler(monkeypatch, [notification("0")], [(0.008, notification("1")), (0.008, notification("2"))])
    assert len(s._drain_omni_connector_outputs()) == 2
    assert clock[0] == pytest.approx(10.012)
    s._omni_connector_output_inbox.initial.append(notification("3"))
    assert len(s._drain_omni_connector_outputs()) == 2
    assert clock[0] == pytest.approx(10.024)


def test_cancelled_notification_is_filtered_before_coordinator(monkeypatch, mocker):
    event = notification("0", "cancelled")
    event.request_metadata = {"0": {"code_predictor_codes": [7]}, "cancelled": {"code_predictor_codes": [9]}}
    s, _ = scheduler(monkeypatch, [event])
    s._generation_max_wait_s = 0
    s.waiting = []
    s.kv_holding_waiting = []
    s.input_coordinator._async_chunk = True
    s.input_coordinator.update_request_metadata = mocker.Mock()
    s.input_coordinator.process_pending_chunks = mocker.Mock()
    s._consume_pending_connector_output("generation")
    s.input_coordinator.update_request_metadata.assert_called_once_with(
        s.requests, {"0": {"code_predictor_codes": [7]}}, model_mode="generation"
    )
    s.input_coordinator.process_pending_chunks.assert_called_once()
    waiting, running, ready, finished = s.input_coordinator.process_pending_chunks.call_args.args
    assert list(waiting) == []
    assert running is s.running
    assert ready == {"0"} and finished == set()


@pytest.mark.parametrize("target,wait", [(0, 1), (257, 1), (4, -1), (4, float("nan")), (4, float("inf"))])
def test_invalid_config_rejected(monkeypatch, target, wait):
    from vllm_omni.core.sched.omni_generation_scheduler import VLLMScheduler

    def initialize(s, *args, **kwargs):
        s.max_num_running_reqs = 256
        s.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                retains_state_across_chunks=True,
                stage_connector_config={"extra": {"generation_min_batch_size": target, "generation_max_wait_ms": wait}},
            )
        )

    monkeypatch.setattr(VLLMScheduler, "__init__", initialize)
    monkeypatch.setattr(OmniGenerationScheduler, "_init_omni_io_scheduling_state", lambda s: None)
    with pytest.raises(ValueError):
        OmniGenerationScheduler()


@pytest.mark.parametrize("terminal", [False, True])
def test_first_and_terminal_chunks_have_bounded_wait(monkeypatch, terminal):
    event = notification("0", terminal=["0"] if terminal else [])
    s, clock = scheduler(monkeypatch, [event])
    s.running = s.running[1:]
    assert s._drain_omni_connector_outputs() == [event]
    assert clock[0] == pytest.approx(10.012)


def test_retire_in_flight_output_before_waiting_without_resetting_deadline(monkeypatch):
    events = [notification("0"), notification("1")]
    s, clock = scheduler(monkeypatch, events[:1])
    s.requests["7"].num_in_flight_tokens = 1
    assert s._drain_omni_connector_outputs() == events[:1]
    assert s._generation_defer_batch
    assert not s._omni_connector_output_inbox.waits
    s.input_coordinator.requests_with_ready_chunks.add("0")
    # Simulate EngineCore retiring the previous batch after five milliseconds.
    clock[0] += 0.005
    s.requests["7"].num_in_flight_tokens = 0
    s._omni_connector_output_inbox.initial.append(events[1])
    assert s._drain_omni_connector_outputs() == events[1:]
    assert not s._generation_defer_batch
    assert s._omni_connector_output_inbox.waits == pytest.approx([0.007])
    assert clock[0] == pytest.approx(10.012)
    assert s._generation_batch_deadline is None


def test_expired_deadline_dispatches_even_with_other_requests_in_flight(monkeypatch):
    s, clock = scheduler(monkeypatch, [notification("0")])
    s.requests["7"].num_in_flight_tokens = 1
    s._generation_batch_deadline = clock[0] - 0.001
    assert len(s._drain_omni_connector_outputs()) == 1
    assert not s._generation_defer_batch
    assert not s._omni_connector_output_inbox.waits


def test_deferred_schedule_keeps_pending_request_and_allocates_nothing(monkeypatch, mocker):
    from tests.core.sched.test_generation_scheduler_restore import _chunk_request, _make_generation_scheduler

    request = _chunk_request("pending")
    s = _make_generation_scheduler(request)
    s._process_pending_omni_inputs = lambda **kwargs: setattr(s, "_generation_defer_batch", True)
    allocate = mocker.spy(s.kv_cache_manager, "allocate_slots")
    output = s.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert not output.num_scheduled_tokens
    allocate.assert_not_called()
    assert list(s.waiting) == [request]
    assert request.num_computed_tokens == request.num_in_flight_tokens == 0
    assert s.chunk_transfer_adapter.restore_called
