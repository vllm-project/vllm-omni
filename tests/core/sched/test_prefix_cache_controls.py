# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduler-owned content controls, including the real zero-token dispatch path."""

from collections import defaultdict, deque
from contextlib import nullcontext
from queue import SimpleQueue
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from tests.core.sched.test_omni_ar_scheduler_stale_drain import _make_drain_sched
from tests.core.sched.test_omni_ar_scheduler_streaming import _make_request, _make_scheduler, _make_update
from tests.core.sched.test_omni_scheduling_coordinator import MockQueue
from tests.helpers.fixtures import ipc
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheRequestOwner,
    PrefixCacheSchedulerAdapter,
)
from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
executor_roundtrip = ipc.executor_roundtrip


def test_prefix_cache_state_is_initialized_before_first_request(monkeypatch):
    monkeypatch.setattr(Scheduler, "__init__", lambda self: None)

    class MinimalScheduler(OmniSchedulerMixin, Scheduler):
        pass

    first, second = MinimalScheduler(), MinimalScheduler()
    assert first._prefix_cache_next_admission == first._prefix_cache_step_sequence == 0
    assert first._prefix_cache_pending_replacements == []
    assert first._prefix_cache_pending_terminal_owners == {}
    assert first._prefix_cache_pending_replacements is not second._prefix_cache_pending_replacements
    assert first._prefix_cache_pending_terminal_owners is not second._prefix_cache_pending_terminal_owners


def test_lookup_entry_rejects_placeholder_before_native_lookup(monkeypatch, mocker):
    scheduler, request = scheduler_request()
    request._omni_input_finalized = False
    native_lookup = mocker.Mock(return_value=(None, 0, 0, False))
    monkeypatch.setattr(Scheduler, "_get_local_prefix_cache_hit", native_lookup)
    with pytest.raises(AssertionError, match="before input finalization"):
        scheduler._get_local_prefix_cache_hit(request)
    native_lookup.assert_not_called()

    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata(scheduler.requests, {request.request_id: {"next_stage_prompt_ids": [1, 2]}})
    scheduler._get_local_prefix_cache_hit(request)
    native_lookup.assert_called_once_with(request)


def test_length_only_notice_finalizes_even_when_placeholder_length_matches():
    scheduler, request = scheduler_request()
    request._omni_input_finalized = False
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata(
        scheduler.requests, {request.request_id: {"next_stage_prompt_len": request.num_prompt_tokens}}
    )
    assert request._omni_input_finalized


@pytest.mark.parametrize("parked", [False, True])
@pytest.mark.parametrize("metadata", [None, {}, {"input_terminal": True}])
@pytest.mark.parametrize("embedding_only", [False, True])
def test_full_payload_readiness_finalizes_unchanged_prompt_before_lookup(
    parked, metadata, embedding_only, monkeypatch, mocker
):
    from tests.core.sched.test_input_finalization import _request

    scheduler, request = scheduler_request()
    if embedding_only:
        request = _request(torch.ones(3, 4), None, None)
        scheduler.requests = {request.request_id: request}
    coordinator = scheduler.input_coordinator = OmniSchedulingCoordinator(stage_id=1)
    scheduler.waiting = MockQueue([request])
    scheduler.running = []
    request._omni_input_finalized = False
    tokens = list(request.prompt_token_ids) if request.prompt_token_ids is not None else None
    prompt_length = request.num_prompt_tokens
    hashes = list(request.block_hashes)
    if parked:
        scheduler._consume_pending_connector_output("ar")
        assert request.status == RequestStatus.WAITING_FOR_INPUT

    notice = OmniConnectorOutput(
        request_metadata={} if metadata is None else {request.request_id: metadata},
        stage_recv_req_ids={request.request_id},
        input_owners={request.request_id: scheduler._prefix_cache_owner(request)},
    )
    scheduler._latest_omni_connector_output = notice
    finalize = mocker.spy(coordinator, "_finalize_prompt")
    native_lookup = mocker.Mock(return_value=(None, 0, 0, False))
    monkeypatch.setattr(Scheduler, "_get_local_prefix_cache_hit", native_lookup)
    scheduler._consume_pending_connector_output("ar")
    scheduler._get_local_prefix_cache_hit(request)
    assert request._omni_input_finalized
    assert request.prompt_token_ids == tokens
    assert request.num_prompt_tokens == prompt_length
    assert request.block_hashes == hashes
    assert request.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [request]
    request.num_computed_tokens = 1
    scheduler._latest_omni_connector_output = notice
    scheduler._consume_pending_connector_output("ar")
    assert request.num_computed_tokens == 1
    finalize.assert_called_once_with(request, tokens)
    native_lookup.assert_called_once_with(request)


def test_executor_roundtrip_cleans_writer_when_reader_setup_fails(monkeypatch, mocker):
    writer = SimpleNamespace(
        export_handle=mocker.Mock(return_value=object()),
        shutdown=mocker.Mock(),
        local_socket=SimpleNamespace(context=mocker.Mock()),
        buffer=object(),
    )
    queue_factory = mocker.Mock(return_value=writer)
    queue_factory.create_from_handle.side_effect = RuntimeError("reader setup failed")
    monkeypatch.setattr(ipc, "MessageQueue", queue_factory)

    fixture = ipc.executor_roundtrip.__wrapped__()
    with pytest.raises(RuntimeError, match="reader setup failed"):
        next(fixture)

    writer.shutdown.assert_called_once_with()
    writer.local_socket.context.destroy.assert_called_once_with(linger=0)
    assert not hasattr(writer, "buffer")


def scheduler_request():
    scheduler = _make_scheduler(stage_id=1)
    request = _make_request()
    scheduler.requests = {request.request_id: request}
    scheduler.requests_needing_kv_transfer = {}
    scheduler.active_kv_transfers = set()
    scheduler.waiting_for_transfer_free = set()
    scheduler.input_coordinator = None
    return scheduler, request


@pytest.fixture(params=[(OmniARScheduler, "ar"), (OmniGenerationScheduler, "generation")], ids=["ar", "generation"])
def input_failure_scheduler(request, mocker):
    scheduler_cls, model_mode = request.param
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.requests = {}
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.deferred_waiting = set()
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.running = []
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = OmniSchedulingCoordinator(stage_id=1)
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._prefix_cache_next_admission = 0
    scheduler._prefix_cache_step_sequence = 0
    scheduler._prefix_cache_pending_replacements = []
    scheduler._prefix_cache_pending_terminal_owners = {}
    scheduler._omits_kv_transfer_cache = {}
    scheduler._omni_kv_config = {}
    scheduler._new_prompt_len_snapshot = {}
    scheduler._inflight_prefills = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.ec_connector = None
    scheduler.aux_output_connector = None
    # Keep native finish/free logic; replace only connector and cache I/O.
    scheduler._connector_finished = mocker.Mock(return_value=(False, None))
    scheduler._free_request_blocks = mocker.Mock()
    scheduler.encoder_cache_manager = mocker.Mock()
    return scheduler, model_mode


@pytest.mark.parametrize("bad_first", [False, True])
@pytest.mark.parametrize("parked", [False, True])
@pytest.mark.parametrize("invalid_length", [True, 0, -1])
def test_invalid_full_payload_finishes_only_its_request(input_failure_scheduler, bad_first, parked, invalid_length):
    scheduler, model_mode = input_failure_scheduler
    bad, healthy = _make_request(), _make_request()
    bad.request_id, healthy.request_id = "invalid-input", "healthy-input"
    ordered = [bad, healthy] if bad_first else [healthy, bad]
    for req in ordered:
        scheduler.requests[req.request_id] = req
        scheduler._prefix_cache_owner(req)
        scheduler.waiting.add_request(req)
    coordinator = scheduler.input_coordinator
    if parked:
        scheduler._consume_pending_connector_output(model_mode)
        assert list(scheduler.waiting) == []
        assert list(coordinator._waiting_for_input) == ordered

    metadata = {
        req.request_id: {"next_stage_prompt_len": invalid_length if req is bad else 5, "input_terminal": True}
        for req in ordered
    }
    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        request_metadata=metadata,
        stage_recv_req_ids=set(metadata),
        chunk_ready_req_ids=set(metadata),
        chunk_finished_req_ids=set(metadata),
        input_owners={req.request_id: scheduler._prefix_cache_owner(req) for req in ordered},
    )
    scheduler._process_pending_omni_inputs(model_mode)

    assert bad.status == RequestStatus.FINISHED_ERROR
    assert bad.request_id not in scheduler.requests
    assert bad.prompt_token_ids == [1, 2, 3]
    assert list(scheduler.waiting) == [healthy]
    assert healthy.status == RequestStatus.WAITING
    assert healthy._omni_input_finalized and healthy.num_prompt_tokens == 5
    assert coordinator._full_payload_input_received == {healthy.request_id}
    assert coordinator.finished_requests == {healthy.request_id}
    assert coordinator.input_terminal_req_ids == {healthy.request_id}
    assert not coordinator._waiting_for_input
    assert not coordinator._waiting_since
    assert not coordinator.pending_input_registrations
    scheduler._free_request_blocks.assert_called_once_with(bad)
    scheduler.encoder_cache_manager.free.assert_called_once_with(bad)
    scheduler._connector_finished.assert_called_once_with(bad)
    assert scheduler._prefix_cache_pending_terminal_owners == {bad.request_id: scheduler._prefix_cache_owner(bad)}

    # Both schedulers must deliver ERROR even when the worker has no tokens.
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=model_mode == "ar")
    error = outputs[bad.client_index].outputs
    assert len(error) == 1
    assert error[0].request_id == bad.request_id
    assert error[0].finish_reason == FinishReason.ERROR
    assert "positive integer" in error[0].stop_reason
    assert outputs[bad.client_index].finished_requests == {bad.request_id}
    subsequent_outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(subsequent_outputs, synthesize_abort_outputs=model_mode == "ar")
    assert subsequent_outputs == {}

    # The next cycle must still admit the healthy request, not re-park it.
    scheduler._process_pending_omni_inputs(model_mode)
    assert list(scheduler.waiting) == [healthy]
    assert healthy.status == RequestStatus.WAITING


@pytest.mark.parametrize("bad_first", [False, True])
@pytest.mark.parametrize("invalid_ids", [[1.5], [True], ["1"], [float("inf")], [{}], [[[1]]]])
def test_invalid_input_ids_do_not_block_healthy_request(input_failure_scheduler, bad_first, invalid_ids):
    scheduler, model_mode = input_failure_scheduler
    bad, healthy = _make_request(), _make_request()
    bad.request_id, healthy.request_id = "invalid-ids", "healthy-ids"
    ordered = [bad, healthy] if bad_first else [healthy, bad]
    ids_key = "next_stage_prompt_ids" if model_mode == "ar" else "code_predictor_codes"
    for req in ordered:
        scheduler.requests[req.request_id] = req
        scheduler._prefix_cache_owner(req)
        scheduler.waiting.add_request(req)
    metadata = {
        req.request_id: {ids_key: invalid_ids if req is bad else [4, 5], "input_terminal": True} for req in ordered
    }
    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        request_metadata=metadata,
        stage_recv_req_ids=set(metadata),
        input_owners={req.request_id: scheduler._prefix_cache_owner(req) for req in ordered},
    )

    scheduler._process_pending_omni_inputs(model_mode)

    assert bad.status == RequestStatus.FINISHED_ERROR
    assert bad.request_id not in scheduler.requests
    assert bad.prompt_token_ids == [1, 2, 3]
    assert list(scheduler.waiting) == [healthy]
    assert healthy.prompt_token_ids == [4, 5]
    assert scheduler.input_coordinator.input_terminal_req_ids == {healthy.request_id}
    scheduler._free_request_blocks.assert_called_once_with(bad)
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=model_mode == "ar")
    assert len(outputs[bad.client_index].outputs) == 1
    assert outputs[bad.client_index].outputs[0].request_id == bad.request_id
    assert outputs[bad.client_index].outputs[0].finish_reason == FinishReason.ERROR


@pytest.mark.parametrize("deferred_free", [False, True])
def test_invalid_chunk_receive_emits_each_error_once(input_failure_scheduler, deferred_free, mocker):
    scheduler, model_mode = input_failure_scheduler
    bad_length, bad_ids, healthy = (_make_request() for _ in range(3))
    bad_length.request_id, bad_ids.request_id, healthy.request_id = "bad-length", "bad-ids", "healthy"
    for req in (bad_length, healthy, bad_ids):
        scheduler.requests[req.request_id] = req
        scheduler._prefix_cache_owner(req)
        scheduler.waiting.add_request(req)
    failures = {
        bad_length.request_id: "invalid prompt length",
        bad_ids.request_id: "invalid codec IDs",
        "freed": "late",
    }
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        collect_failed_receive_request_ids=mocker.Mock(side_effect=[failures, failures, {}]),
        finish_requests=mocker.Mock(),
        cleanup_receiver=mocker.Mock(),
    )
    scheduler._connector_finished.return_value = (deferred_free, None)

    scheduler._process_chunk_receive_failures()

    assert bad_length.status == bad_ids.status == RequestStatus.FINISHED_ERROR
    assert list(scheduler.waiting) == [healthy]
    assert healthy.status == RequestStatus.WAITING
    assert healthy.prompt_token_ids == bad_length.prompt_token_ids == bad_ids.prompt_token_ids == [1, 2, 3]
    assert scheduler._prefix_cache_pending_terminal_owners == {
        req.request_id: scheduler._prefix_cache_owner(req) for req in (bad_length, bad_ids)
    }
    for req in (bad_length, bad_ids):
        assert (req.request_id in scheduler.requests) == deferred_free
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=model_mode == "ar")
    errors = {output.request_id: output for output in outputs[bad_length.client_index].outputs}
    assert len(outputs[bad_length.client_index].outputs) == 2
    assert set(errors) == {bad_length.request_id, bad_ids.request_id}
    for req_id, output in errors.items():
        assert output.finish_reason == FinishReason.ERROR
        assert output.stop_reason == f"Invalid connector input: {failures[req_id]}"
    assert outputs[bad_length.client_index].finished_requests == set(errors)

    # A finished request can remain live while its connector delays KV freeing.
    # Repeated failure notices must not clean it up or emit its ERROR again.
    scheduler._process_chunk_receive_failures()
    scheduler._process_chunk_receive_failures()
    outputs = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=model_mode == "ar")
    assert outputs == {}
    assert scheduler._connector_finished.call_count == 2
    assert scheduler._free_request_blocks.call_count == (0 if deferred_free else 2)
    assert scheduler.encoder_cache_manager.free.call_count == 2
    assert list(scheduler.waiting) == [healthy]


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, TypeError])
def test_full_payload_finalizer_isolates_only_validation_errors(input_failure_scheduler, exception_type):
    scheduler, model_mode = input_failure_scheduler
    req = _make_request()
    scheduler.requests[req.request_id] = req
    scheduler.waiting.add_request(req)

    def reject_conditioning(request, metadata):
        raise exception_type("invalid conditioning")

    scheduler.input_coordinator._conditioning_finalizer = reject_conditioning
    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        request_metadata={req.request_id: {"next_stage_prompt_len": 3}},
        stage_recv_req_ids={req.request_id},
        input_owners={req.request_id: scheduler._prefix_cache_owner(req)},
    )
    if exception_type is ValueError:
        scheduler._process_pending_omni_inputs(model_mode)
        assert req.status == RequestStatus.FINISHED_ERROR
        assert not scheduler.requests
        assert not scheduler.input_coordinator._full_payload_input_received
    else:
        with pytest.raises(exception_type, match="invalid conditioning"):
            scheduler._process_pending_omni_inputs(model_mode)
        assert req.status == RequestStatus.WAITING
        assert scheduler.requests == {req.request_id: req}
        scheduler._free_request_blocks.assert_not_called()


@pytest.mark.parametrize("deferred_free", [False, True])
def test_late_invalid_notice_cannot_refinish_or_poison_reused_request(input_failure_scheduler, deferred_free):
    scheduler, model_mode = input_failure_scheduler
    retired = _make_request()
    scheduler.requests[retired.request_id] = retired
    scheduler.waiting.add_request(retired)
    owner = scheduler._prefix_cache_owner(retired)
    scheduler._connector_finished.return_value = (deferred_free, None)
    notice = OmniConnectorOutput(
        request_metadata={retired.request_id: {"next_stage_prompt_len": -1}},
        stage_recv_req_ids={retired.request_id},
        input_owners={retired.request_id: owner},
    )
    scheduler._latest_omni_connector_output = notice
    scheduler._process_pending_omni_inputs(model_mode)
    assert retired.status == RequestStatus.FINISHED_ERROR
    assert bool(scheduler.requests) == deferred_free
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=model_mode == "ar")
    assert outputs[retired.client_index].outputs[0].finish_reason == FinishReason.ERROR

    if not deferred_free:
        replacement = _make_request()
        scheduler.requests[replacement.request_id] = replacement
        scheduler._prefix_cache_owner(replacement)
        scheduler.waiting.add_request(replacement)
    scheduler._latest_omni_connector_output = notice
    scheduler._process_pending_omni_inputs(model_mode)
    scheduler._connector_finished.assert_called_once_with(retired)
    assert not scheduler.input_coordinator._full_payload_input_received
    assert not scheduler.input_coordinator.finished_requests
    outputs = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=model_mode == "ar")
    assert not outputs
    if not deferred_free:
        assert replacement.status == RequestStatus.WAITING_FOR_INPUT
        assert scheduler.requests == {replacement.request_id: replacement}


def test_each_accepted_replacement_advances_content_once_before_admission():
    scheduler, request = scheduler_request()
    first = scheduler._prefix_cache_owner(request)
    scheduler._replace_streaming_session(request, _make_update([4, 5]))
    scheduler._replace_streaming_session(request, _make_update([6, 7]))
    current = scheduler._prefix_cache_owner(request)
    assert current.admission_id == first.admission_id
    assert current.generation == first.generation + 2
    output = scheduler._wrap_omni_scheduler_output(SchedulerOutput.make_empty())
    assert [event.owner.generation for event in output.prefix_cache_replacements] == [1, 2]
    assert all(not event.lookup_complete for event in output.prefix_cache_replacements)
    assert output.total_num_scheduled_tokens == 0
    assert scheduler._wrap_omni_scheduler_output(SchedulerOutput.make_empty()).prefix_cache_replacements == ()


def test_invalid_replacement_does_not_assign_or_advance_owner():
    scheduler, request = scheduler_request()
    with pytest.raises(ValueError, match="non-negative"):
        scheduler._replace_streaming_session(request, _make_update([-1]))
    assert not hasattr(request, "_omni_prefix_cache_owner")
    assert not getattr(scheduler, "_prefix_cache_pending_replacements", ())


def test_full_payload_replacement_waits_for_new_identity_and_does_not_recompose_old_salt():
    from vllm_omni.core.sched.input_finalization import compose_conditioning_cache_salt

    scheduler, request = scheduler_request()
    coordinator = scheduler.input_coordinator = OmniSchedulingCoordinator(stage_id=1)
    scheduler.waiting = MockQueue([request])
    scheduler.running = []
    request.cache_salt = "caller"
    previous = scheduler._prefix_cache_owner(request)
    coordinator.update_request_metadata(
        {request.request_id: request},
        {
            request.request_id: {
                "next_stage_prompt_ids": [11, 12, 13],
                "next_stage_prompt_len": 3,
                "next_stage_conditioning_digest": "ab" * 32,
            }
        },
    )
    coordinator._full_payload_input_received.add(request.request_id)
    coordinator.finished_requests.add(request.request_id)
    coordinator.input_terminal_req_ids.add(request.request_id)

    scheduler._replace_streaming_session(request, _make_update([0] * 4))
    current = scheduler._prefix_cache_owner(request)
    assert current.generation == previous.generation + 1
    assert request.cache_salt == "caller"
    assert not request._omni_input_finalized
    scheduler._consume_pending_connector_output("ar")
    assert request.status == RequestStatus.WAITING_FOR_INPUT
    assert list(scheduler.waiting) == []
    assert not coordinator.input_terminal_req_ids
    assert coordinator.pending_input_registrations[0].input_owner == current

    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        stage_recv_req_ids={request.request_id},
        request_metadata={
            request.request_id: {
                "next_stage_prompt_ids": [21, 22, 23, 24],
                "next_stage_prompt_len": 4,
                "next_stage_conditioning_digest": "cd" * 32,
            }
        },
        input_owners={request.request_id: current},
    )
    scheduler._consume_pending_connector_output("ar")
    assert request.prompt_token_ids == [21, 22, 23, 24]
    assert request.cache_salt == compose_conditioning_cache_salt("caller", "cd" * 32)
    assert request._omni_original_cache_salt == "caller"
    assert request._omni_input_finalized
    assert request.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [request]
    assert scheduler._prefix_cache_owner(request) == current


def test_repeated_full_payload_replacement_preserves_coordinator_owned_waiter():
    scheduler, request = scheduler_request()
    coordinator = scheduler.input_coordinator = OmniSchedulingCoordinator(stage_id=1)
    scheduler.waiting = MockQueue([request])
    scheduler.running = []
    for length in (4, 6, 2):
        scheduler._replace_streaming_session(request, _make_update([0] * length))
        current = scheduler._prefix_cache_owner(request)
        scheduler._consume_pending_connector_output("ar")
        assert list(scheduler.waiting) == []
        assert list(coordinator._waiting_for_input) == [request]
        assert request.status == RequestStatus.WAITING_FOR_INPUT
        assert not request._omni_input_finalized
        assert len(coordinator.pending_input_registrations) == 1
        assert coordinator.pending_input_registrations[0].input_owner == current


def test_append_keeps_content_owner_independent_from_transport_segment():
    scheduler, request = scheduler_request()
    scheduler.vllm_config.model_config.stage_id = 0
    owner = scheduler._prefix_cache_owner(request)
    scheduler._update_request_as_session(request, _make_update([4, 5]))
    assert scheduler._prefix_cache_owner(request) == owner
    assert request._omni_segment_generation == 1
    assert not getattr(scheduler, "_prefix_cache_pending_replacements", ())


def test_async_ready_replacement_is_accepted_once():
    scheduler, request = scheduler_request()
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        replaced_streaming_prompt_ids={request.request_id},
        requests_with_ready_chunks={request.request_id},
        requests_num_chunks_sent={},
    )
    initial = scheduler._prefix_cache_owner(request)
    scheduler._reset_ready_async_chunk_replacements()
    scheduler._reset_ready_async_chunk_replacements()
    assert scheduler._prefix_cache_owner(request).generation == initial.generation + 1
    assert len(scheduler._prefix_cache_pending_replacements) == 1


@pytest.mark.parametrize("retained_for_kv_transfer", [False, True])
def test_finished_replacement_does_not_leave_a_reactivating_control(retained_for_kv_transfer):
    scheduler, request = scheduler_request()
    scheduler._replace_streaming_session(request, _make_update([4, 5]))
    owner = scheduler._prefix_cache_owner(request)
    scheduler._record_prefix_cache_finished(request)
    request.status = RequestStatus.FINISHED_STOPPED
    if not retained_for_kv_transfer:
        scheduler.requests.pop(request.request_id)
    base = SchedulerOutput.make_empty()
    base.finished_req_ids = {request.request_id}
    output = scheduler._wrap_omni_scheduler_output(base)
    assert output.prefix_cache_replacements == ()
    assert output.prefix_cache_terminal_owners == {request.request_id: owner}
    assert not scheduler._prefix_cache_pending_terminal_owners


def test_lifecycle_metadata_round_trips_through_executor_serialization(executor_roundtrip):
    scheduler, request = scheduler_request()
    scheduler._replace_streaming_session(request, _make_update([4, 5]))
    output = scheduler._wrap_omni_scheduler_output(SchedulerOutput.make_empty())
    restored = executor_roundtrip(output)
    assert restored.prefix_cache_replacements == output.prefix_cache_replacements
    assert restored.prefix_cache_step_sequence == output.prefix_cache_step_sequence
    assert PrefixCacheSchedulerAdapter().translate_step(restored).events[0].kind is PrefixCacheEventKind.REPLACED
    # The new immutable carrier is also independently typed-msgpack compatible.
    controls = MsgpackDecoder(tuple[PrefixCacheRequestEvent, ...]).decode(
        MsgpackEncoder().encode(output.prefix_cache_replacements)
    )
    assert controls == output.prefix_cache_replacements


@pytest.mark.parametrize("scheduler_type", [OmniARScheduler, OmniGenerationScheduler])
@pytest.mark.parametrize("reused_id", [False, True])
@pytest.mark.parametrize("stale_window", [False, True])
def test_retired_output_does_not_mutate_reused_request_or_drop_surviving_batch_member(
    scheduler_type, reused_id, stale_window, mocker
):
    retired = _make_request()
    survivor = _make_request()
    survivor.request_id = "survivor"
    retired.status = survivor.status = RequestStatus.RUNNING
    retired._omni_prefix_cache_owner = PrefixCacheRequestOwner(2, 0) if reused_id else PrefixCacheRequestOwner(1, 1)
    survivor._omni_prefix_cache_owner = PrefixCacheRequestOwner(3)
    retired.num_in_flight_tokens = 7 if reused_id else 3
    retired.num_stale_output_tokens = 2 if stale_window and not reused_id else 0
    survivor.num_in_flight_tokens = 1
    scheduler = _make_drain_sched(retired)
    scheduler.requests[survivor.request_id] = survivor
    scheduler.running.append(survivor)
    scheduler.chunk_transfer_adapter = None
    scheduler._pending_finish_reqs = []
    scheduler.recompute_kv_load_failures = False
    scheduler._first_chunk_express = False
    scheduler._express_min_slack_s = 0
    scheduler._async_chunk_transport_enabled.return_value = False
    base = SchedulerOutput.make_empty()
    base.num_scheduled_tokens = {retired.request_id: 2, survivor.request_id: 1}
    base.total_num_scheduled_tokens = 3
    base.prefix_cache_owners = {
        retired.request_id: PrefixCacheRequestOwner(1),
        survivor.request_id: PrefixCacheRequestOwner(3),
    }
    output = mocker.MagicMock(spec=ModelRunnerOutput)
    output.sampled_token_ids = [[99], [42]]
    output.req_id_to_index = {retired.request_id: 0, survivor.request_id: 1}
    output.logprobs = None
    output.prompt_logprobs_dict = {}
    output.pooler_output = None
    output.multimodal_outputs = None
    output.inter_stage_outputs = None
    output.num_nans_in_logits = None
    output.kv_connector_output = None
    output.cudagraph_stats = None
    output.routed_experts = None
    result = scheduler_type.update_from_output(scheduler, base, output)
    assert retired.num_in_flight_tokens == (7 if reused_id else 1)
    assert retired.num_stale_output_tokens == 0
    assert survivor.num_in_flight_tokens == 0
    assert list(retired.output_token_ids) == []
    emitted = [item for client_output in result.values() for item in client_output.outputs]
    assert [(item.request_id, item.new_token_ids) for item in emitted] == [(survivor.request_id, [42])]
    assert all(call.args[0] is survivor for call in scheduler._update_request_with_output.call_args_list)


@pytest.mark.parametrize("current_notice", [False, True])
@pytest.mark.parametrize("reused_id", [False, True])
def test_connector_notices_are_owner_filtered_before_merging(current_notice, reused_id):
    scheduler, request = scheduler_request()
    owner = PrefixCacheRequestOwner(2) if reused_id else PrefixCacheRequestOwner(1, 1)
    request._omni_prefix_cache_owner = owner
    request.status = RequestStatus.WAITING
    scheduler.waiting = MockQueue()
    scheduler.waiting.add_request(request)
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    scheduler.input_coordinator = coordinator
    scheduler._init_omni_connector_output_inbox()
    coordinator.process_pending_full_payload_inputs(scheduler.waiting, set())
    assert request.status == RequestStatus.WAITING_FOR_INPUT
    handle = coordinator.pending_input_registrations[0]
    assert handle.input_owner == owner
    assert MsgpackDecoder(type(handle)).decode(MsgpackEncoder().encode(handle)).input_owner == owner
    if current_notice:
        current = OmniConnectorOutput(
            request_metadata={request.request_id: {"next_stage_prompt_len": 5, "input_terminal": True}},
            stage_recv_req_ids={request.request_id},
        )
        current.input_owners = {request.request_id: owner}
        scheduler.enqueue_omni_connector_output(current)
    old = OmniConnectorOutput(
        request_metadata={request.request_id: {"next_stage_prompt_len": -99, "input_terminal": True}},
        stage_recv_req_ids={request.request_id},
        chunk_ready_req_ids={request.request_id},
        chunk_finished_req_ids={request.request_id},
    )
    old.input_owners = {request.request_id: PrefixCacheRequestOwner(1)}
    scheduler.enqueue_omni_connector_output(old)
    scheduler._consume_pending_connector_output("ar")
    if current_notice:
        assert request.status == RequestStatus.WAITING
        assert request.num_prompt_tokens == 5
        assert request.request_id in coordinator.input_terminal_req_ids
    else:
        assert request.status == RequestStatus.WAITING_FOR_INPUT
        assert request.request_id not in coordinator.input_terminal_req_ids
        assert request.request_id not in coordinator.finished_requests


@pytest.mark.parametrize("queued", [False, True])
def test_control_only_step_reaches_real_executor_and_runner_before_zero_return(monkeypatch, queued, mocker):
    from vllm_omni.worker import gpu_ar_model_runner as runner_module

    scheduler, request = scheduler_request()
    # Isolate the work indication: the base scheduler reports no runnable or
    # finished work, so only the accepted control can keep EngineCore active.
    monkeypatch.setattr(Scheduler, "has_requests", lambda self: False)
    scheduler._replace_streaming_session(request, _make_update([4, 5]))
    scheduler.schedule = lambda *args: scheduler._wrap_omni_scheduler_output(SchedulerOutput.make_empty())
    scheduler.update_from_output = mocker.Mock(return_value={})
    scheduler.get_grammar_bitmask = mocker.Mock(return_value=None)
    assert scheduler.has_requests()

    cache = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=4, block_size=4), eager=True)
    monkeypatch.setattr(cache, "save_outputs", mocker.Mock(side_effect=AssertionError("control step saved tensors")))
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner._warmup_state_cleared = True
    runner.model = SimpleNamespace()
    runner.omni_prefix_cache = cache
    runner.kv_transfer_manager = SimpleNamespace(handle_finished_requests_kv_transfer=lambda **kwargs: None)
    runner.kv_caches = []
    runner.cache_config = SimpleNamespace(block_size=4, cache_dtype="auto")
    runner._resolve_global_request_id = lambda req_id: req_id
    runner.speculative_config = None
    runner.parallel_config = SimpleNamespace(distributed_executor_backend="uni", data_parallel_size=1)
    runner.synchronize_input_prep = nullcontext
    runner._update_states = mocker.Mock(return_value=None)
    runner.attach_omni_connector_output = lambda output: output
    monkeypatch.setattr(runner_module, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(runner_module, "has_ec_transfer", lambda: False)
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = runner
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = executor
    engine.aborts_queue = SimpleQueue()
    engine.capture_iteration_details = lambda output: nullcontext(None)
    engine.log_error_detail = lambda output: nullcontext()
    engine._attach_iteration_details = lambda outputs, details: None
    engine.batch_queue = deque() if queued else None
    engine.batch_queue_size = 2
    engine.engines_running = False
    engine.is_ec_consumer = False
    engine.is_mm_encoder_only = False
    engine.is_pooling_model = False
    try:
        assert EngineCoreProc.has_work(engine)
        result, executed = engine.step_with_batch_queue() if queued else engine.step()
        assert result == {} and not executed
        assert cache._request_progress[request.request_id].owner == scheduler._prefix_cache_owner(request)
        assert cache._request_progress[request.request_id].lookup is None
        assert not cache._step_ctxs
        cache.save_outputs.assert_not_called()
        scheduler.update_from_output.assert_called_once()
        runner._update_states.assert_called_once()
        assert not scheduler.has_requests()
        assert not EngineCoreProc.has_work(engine)
    finally:
        cache.shutdown()
