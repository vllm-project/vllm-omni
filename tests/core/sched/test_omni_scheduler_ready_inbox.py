# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drain ready events once, merging live requests and filtering cancellations."""

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.outputs import OmniConnectorOutput, SchedulingMetadataUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ready_inbox_coalesces_live_events_and_drops_cancelled(mocker):
    scheduler = OmniSchedulerMixin()
    scheduler.requests, scheduler.waiting, scheduler.running = {"r": object()}, [], []
    scheduler.kv_holding_waiting = []
    scheduler._is_blocked_waiting_status = OmniARScheduler._is_blocked_waiting_status
    coordinator = SimpleNamespace(
        _async_chunk=True, update_request_metadata=mocker.Mock(), process_pending_chunks=mocker.Mock()
    )
    scheduler.input_coordinator = coordinator
    scheduler._init_omni_connector_output_inbox()
    scheduler.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"r"}))
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            chunk_ready_req_ids={"r", "aborted"},
            chunk_finished_req_ids={"r", "aborted"},
            request_metadata={
                "r": SchedulingMetadataUpdate(input_terminal=True),
                "aborted": SchedulingMetadataUpdate(resize_prompt_to=99),
            },
        )
    )
    scheduler._consume_pending_connector_output(model_mode="ar")
    coordinator.update_request_metadata.assert_called_once_with(
        scheduler.requests, {"r": SchedulingMetadataUpdate(input_terminal=True)}
    )
    coordinator.process_pending_chunks.assert_called_once_with(mocker.ANY, [], {"r"}, {"r"})
    assert list(coordinator.process_pending_chunks.call_args.args[0]) == []
    scheduler._consume_pending_connector_output(model_mode="ar")
    assert coordinator.update_request_metadata.call_count == 1
    coordinator.process_pending_chunks.assert_called_with(mocker.ANY, [], set(), set())
    assert list(coordinator.process_pending_chunks.call_args.args[0]) == []


@pytest.mark.parametrize("policy", [SchedulingPolicy.FCFS, SchedulingPolicy.PRIORITY])
@pytest.mark.parametrize("async_chunk", [False, True])
def test_native_input_gate_parks_and_restores_kv_holders(policy, async_chunk):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.waiting = create_request_queue(policy)
    scheduler.kv_holding_waiting = create_request_queue(policy)
    scheduler.deferred_waiting = set()
    scheduler.running = []
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = OmniSchedulingCoordinator(
        stage_id=1, scheduler_max_num_seqs=2, async_chunk=async_chunk
    )
    scheduler._init_omni_connector_output_inbox()
    holder = Request("holder", [1, 2, 3], SamplingParams(max_tokens=1), pooling_params=None)
    holder.num_computed_tokens = 2
    fresh = Request("fresh", [1], SamplingParams(max_tokens=1), pooling_params=None)
    scheduler.requests = {request.request_id: request for request in (holder, fresh)}
    scheduler._enqueue_waiting_request(fresh)
    scheduler._enqueue_waiting_request(holder)

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert not scheduler.waiting and not scheduler.kv_holding_waiting
    expected_status = RequestStatus.WAITING_FOR_CHUNK if async_chunk else RequestStatus.WAITING_FOR_INPUT
    assert holder.status == fresh.status == expected_status
    coordinator = scheduler.input_coordinator
    registrations = coordinator.pending_chunk_registrations if async_chunk else coordinator.pending_input_registrations
    assert [entry.request_id for entry in registrations] == ["holder", "fresh"]

    scheduler._restore_omni_wait_queues()
    assert list(scheduler.kv_holding_waiting) == [holder]
    assert list(scheduler.waiting) == [fresh]
    ready_output = (
        OmniConnectorOutput(chunk_ready_req_ids={"holder"})
        if async_chunk
        else OmniConnectorOutput(stage_recv_req_ids={"holder"})
    )
    scheduler.enqueue_omni_connector_output(ready_output)
    scheduler._consume_pending_connector_output(model_mode="ar")

    assert holder.status == RequestStatus.WAITING
    assert fresh.status == expected_status
    assert list(scheduler.kv_holding_waiting) == [holder]
    assert not scheduler.waiting
    scheduler._restore_omni_wait_queues()
    assert list(scheduler.kv_holding_waiting) == [holder]
    assert list(scheduler.waiting) == [fresh]


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.WAITING_FOR_STREAMING_REQ,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
    ],
)
def test_native_input_gate_preserves_upstream_blocked_waits(async_chunk, status):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.deferred_waiting = set()
    scheduler.running = []
    request = Request("blocked", [1, 2, 3], SamplingParams(max_tokens=1), pooling_params=None)
    request.status = status
    request.num_computed_tokens = 2
    scheduler.requests = {request.request_id: request}
    scheduler._enqueue_waiting_request(request)
    scheduler.input_coordinator = OmniSchedulingCoordinator(
        stage_id=1, scheduler_max_num_seqs=1, async_chunk=async_chunk
    )
    scheduler._init_omni_connector_output_inbox()

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert request.status == status
    assert list(scheduler.kv_holding_waiting) == [request]
    assert scheduler.deferred_waiting == {request}
    assert not scheduler.input_coordinator.pending_chunk_registrations
    assert not scheduler.input_coordinator.pending_input_registrations
