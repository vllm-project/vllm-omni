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


@pytest.mark.parametrize("native_resumable", [False, True])
def test_ready_inbox_coalesces_live_events_and_drops_cancelled(mocker, native_resumable):
    scheduler = OmniSchedulerMixin()
    scheduler.requests, scheduler.waiting, scheduler.running = {"r": SimpleNamespace(resumable=False)}, [], []
    scheduler._native_resumable = native_resumable
    scheduler.kv_holding_waiting = []
    scheduler._is_blocked_waiting_status = OmniARScheduler._is_blocked_waiting_status
    coordinator = SimpleNamespace(
        _async_chunk=True,
        receives_chunks=True,
        update_request_metadata=mocker.Mock(),
        process_pending_chunks=mocker.Mock(),
    )
    scheduler.input_coordinator = coordinator
    scheduler._outputs_awaiting_stage_payload = {}
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
    # Upstream _enqueue_waiting_request reads self.connector (None unless a KV
    # connector is configured); __new__ bypasses the __init__ that sets it.
    scheduler.connector = None
    scheduler.deferred_waiting = set()
    scheduler._outputs_awaiting_stage_payload = {}
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
    scheduler._outputs_awaiting_stage_payload = {}
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


@pytest.mark.parametrize("input_terminal", [False, True])
def test_native_resume_prompt_replacement_clears_token_buffer(mocker, input_terminal):
    scheduler = OmniSchedulerMixin()
    request = Request("r", [1, 2, 3], SamplingParams(max_tokens=16), pooling_params=None)
    request.resumable = True
    request.append_output_token_ids([4, 5])
    request.num_computed_tokens = 4
    request.block_hashes.append(b"old")
    token_buffer = request._all_token_ids
    scheduler.requests = {"r": request}
    scheduler.waiting, scheduler.running, scheduler.kv_holding_waiting = [], [], []
    scheduler._native_resumable = True
    scheduler._resume_chunk_receivers = set()
    scheduler._pending_data_plane_terminal_req_ids = set()
    scheduler.finish_requests = mocker.Mock()
    scheduler._outputs_awaiting_stage_payload = {}
    scheduler._release_replaced_streaming_prompt_cache = mocker.Mock()
    scheduler._reset_streaming_session_replacement_state = mocker.Mock()
    scheduler.input_coordinator = SimpleNamespace(
        receives_chunks=True,
        _async_chunk=True,
        update_request_metadata=mocker.Mock(),
        process_pending_chunks=mocker.Mock(),
    )
    scheduler._init_omni_connector_output_inbox()
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            request_metadata={"r": SchedulingMetadataUpdate(replace_prompt=True, resize_prompt_to=2)},
            chunk_ready_req_ids={"r"},
        )
    )

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert request._all_token_ids is token_buffer
    assert len(request.all_token_ids) == 0
    assert not request.output_token_ids
    assert not request.block_hashes
    assert request.num_computed_tokens == 0
    assert request.prompt_token_ids == []
    scheduler._release_replaced_streaming_prompt_cache.assert_called_once_with(request)

    # A complete segment can arrive while this request waits behind another
    # request. Zero computed tokens do not mean the delivered prompt is absent.
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            request_metadata={
                "r": SchedulingMetadataUpdate(segment_finished=not input_terminal, input_terminal=input_terminal)
            },
        )
    )
    scheduler._consume_pending_connector_output(model_mode="ar")

    assert scheduler.input_coordinator.update_request_metadata.call_count == 2
    scheduler.finish_requests.assert_not_called()
    assert not scheduler._resume_chunk_receivers


@pytest.mark.parametrize("prompt_length", [8192, 8193])
def test_ready_connector_prompt_respects_stage_context_limit(monkeypatch, mocker, prompt_length):
    from collections import defaultdict

    from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
    from vllm.v1.engine import EngineCoreOutputs, FinishReason

    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    request = Request("long-prompt", [1] * prompt_length, SamplingParams(max_tokens=1), pooling_params=None)
    scheduler.requests = {request.request_id: request}
    scheduler.max_model_len = 8192
    scheduler.waiting, scheduler.running = [], []
    scheduler.requests_needing_kv_transfer = {}
    scheduler.active_kv_transfers = set()
    scheduler.waiting_for_transfer_free = set()
    scheduler._outputs_awaiting_stage_payload = {}
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        receives_chunks=True,
        update_request_metadata=mocker.Mock(),
        process_pending_chunks=mocker.Mock(),
    )

    def finish(ids, status):
        for req_id in ids:
            scheduler.requests.pop(req_id).status = status

    monkeypatch.setattr(VLLMScheduler, "has_requests", lambda _: False)
    scheduler.finish_requests = mocker.Mock(side_effect=finish)
    scheduler._init_omni_connector_output_inbox()

    # A prewarm placeholder can be resized when its real payload arrives.
    scheduler._consume_pending_connector_output(model_mode="ar")
    scheduler.finish_requests.assert_not_called()

    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            chunk_ready_req_ids={request.request_id},
        )
    )
    scheduler._consume_pending_connector_output(model_mode="ar")

    if prompt_length == scheduler.max_model_len:
        assert request.request_id in scheduler.requests
        scheduler.finish_requests.assert_not_called()
        return
    scheduler.finish_requests.assert_called_once_with([request.request_id], RequestStatus.FINISHED_ERROR)
    assert scheduler.has_requests()
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=False)
    [error] = outputs[request.client_index].outputs
    assert error.finish_reason == FinishReason.ERROR
    assert "8193" in error.stop_reason and "8192" in error.stop_reason
    assert not scheduler.has_requests()


@pytest.mark.parametrize("replace_ids", [False, True])
def test_connector_metadata_rejects_oversized_prompt_without_blocking_peer(mocker, replace_ids):
    scheduler = OmniSchedulerMixin()
    request = Request("large", [1], SamplingParams(max_tokens=1), pooling_params=None)
    peer = Request("peer", [1], SamplingParams(max_tokens=1), pooling_params=None)
    scheduler.requests = {"large": request, "peer": peer}
    scheduler.max_model_len = 8192
    scheduler.waiting, scheduler.running = [], []
    scheduler._outputs_awaiting_stage_payload = {}
    scheduler.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        receives_chunks=True,
        update_request_metadata=mocker.Mock(),
        process_pending_chunks=mocker.Mock(),
    )

    def finish(ids, status):
        for req_id in ids:
            scheduler.requests.pop(req_id).status = status

    scheduler.finish_requests = mocker.Mock(side_effect=finish)
    invalid = (
        SchedulingMetadataUpdate(prompt_token_ids=(1,) * 8193)
        if replace_ids
        else SchedulingMetadataUpdate(resize_prompt_to=8193)
    )
    valid = SchedulingMetadataUpdate(resize_prompt_to=2)
    scheduler._init_omni_connector_output_inbox()
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            request_metadata={"large": invalid, "peer": valid},
            chunk_ready_req_ids={"large", "peer"},
            chunk_finished_req_ids={"large"},
        )
    )

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert scheduler.requests == {"peer": peer}
    assert request.num_prompt_tokens == 1
    scheduler.finish_requests.assert_called_once_with(["large"], RequestStatus.FINISHED_ERROR)
    scheduler.input_coordinator.update_request_metadata.assert_called_once_with(scheduler.requests, {"peer": valid})
    args = scheduler.input_coordinator.process_pending_chunks.call_args.args
    assert args[2:] == ({"peer"}, set())
    assert "8193" in scheduler._pending_input_timeout_outputs["large"][1].stop_reason
