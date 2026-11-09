# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Abort of a MiniCPM-o length stop that waits for its terminal forward.

The cancel point is fixed by the test: the terminal forward is booked (the
worker's reply is held back) and the abort arrives before or after it. The
real finish_requests, _free_request and update_from_output run; only block
return and unrelated post-loop collaborators are recorded or stubbed.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from tests.core.sched.test_minicpmo_mrv2_terminal_row import _payload, _Request, _scheduler, _step
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Hashable(_Request):
    # finish_requests keeps running requests in a set.
    __hash__ = object.__hash__


def _cancellable(monkeypatch):
    scheduler, _ = _scheduler(monkeypatch)
    request = _Hashable(max_tokens=2)
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.deferred_waiting = set()
    scheduler._outputs_awaiting_stage_payload = {}
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.current_step = 3
    freed = []

    def free_blocks(req):
        freed.append(req.request_id)
        del scheduler.requests[req.request_id]

    scheduler._free_blocks = free_blocks
    scheduler._omits_kv_transfer_cache = {}
    scheduler._connector_finished = lambda req: (False, None)
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda req: None)
    scheduler._pending_encoder_input_ids = {}
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._should_transfer_kv_for_request = lambda rid: False
    scheduler._free_input_coordinator_request = lambda rid: None
    return scheduler, request, freed


def _book_terminal_forward(scheduler, request):
    """schedule() booked the draft-free terminal input; its output is held back."""
    held = scheduler._hold_minicpmo45_terminal_steps()
    assert held == [] and request.spec_token_ids == []
    request.num_computed_tokens += 1
    request.num_in_flight_tokens += 1
    request.num_output_placeholders += 1


def _late_output(monkeypatch, scheduler, request):
    """Deliver the held terminal output after the abort, through the real update path."""
    for name in (
        "_remove_stopped_requests_from_queues", "_emit_streaming_context_overflow_outputs", "_cleanup_kv_tracking",
        "_publish_kv_cache_events", "_attach_finished_request_sets", "_attach_scheduler_stats",
        "_capture_omni_connector_output",
    ):  # fmt: skip
        setattr(scheduler, name, lambda *a, **k: None)
    scheduler._handle_failed_kv_load_outputs = lambda *a: []
    scheduler._aggregate_kv_connector_stats = lambda *a: None
    monkeypatch.setattr(OmniSchedulerMixin, "_finish_error_requests", lambda *a: None)
    monkeypatch.setattr(OmniSchedulerMixin, "_attach_duplex_stage_work", lambda *a: None)
    position = request.num_tokens - 1
    payload = _payload(request.output_token_ids[-1], position)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: 1},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
        finished_req_ids=set(),
    )
    runner_output = SimpleNamespace(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=[payload],
        inter_stage_outputs=[payload],
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
    )
    # Publication and the grammar hook still raise: reaching them would be a handoff.
    return scheduler.update_from_output(scheduler_output, runner_output)


def _aborted_once(scheduler, request, freed):
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.request_id not in scheduler._minicpmo45_terminal_rows
    assert freed == [request.request_id]
    assert request.request_id not in scheduler.requests and request not in scheduler.running


def test_abort_while_the_terminal_forward_is_in_flight(monkeypatch):
    scheduler, request, freed = _cancellable(monkeypatch)
    _step(scheduler, request, [], [12])
    assert not request.is_finished()
    _book_terminal_forward(scheduler, request)
    assert scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED) == [request]
    _aborted_once(scheduler, request, freed)
    outputs = _late_output(monkeypatch, scheduler, request)
    # The late row is dropped: no handoff, no re-registration, no second free.
    assert all(o.request_id != request.request_id for out in outputs.values() for o in out.outputs)
    assert request.output_token_ids == [11, 12]
    _aborted_once(scheduler, request, freed)
    assert scheduler._hold_minicpmo45_terminal_steps() == []


def test_abort_before_the_terminal_forward_is_booked(monkeypatch):
    scheduler, request, freed = _cancellable(monkeypatch)
    _step(scheduler, request, [], [12])
    assert scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED) == [request]
    _aborted_once(scheduler, request, freed)
    # Nothing is left for schedule() to book.
    assert scheduler._hold_minicpmo45_terminal_steps() == []
    assert request.spec_token_ids == []


def test_abort_after_the_terminal_forward_finished_is_a_no_op(monkeypatch):
    scheduler, request, freed = _cancellable(monkeypatch)
    _step(scheduler, request, [], [12])
    _book_terminal_forward(scheduler, request)
    request.num_in_flight_tokens -= 1
    request.num_computed_tokens -= 1
    request.num_output_placeholders -= 1
    _step(scheduler, request, [], [999], _payload(12, 3))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED) == []
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED and freed == []
