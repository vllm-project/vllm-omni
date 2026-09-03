# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(*, stage_id: int = 0, session_mode: str = "turn") -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_id=stage_id, session_mode=session_mode),
    )
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    sched.skipped_waiting = set()
    sched._free_request_blocks = MagicMock()
    sched.encoder_cache_manager = MagicMock()
    sched._inflight_prefills = set()
    return sched


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def _run_resumable_segment_stop(
    session: Request,
    *,
    session_finished: bool = False,
    handle_stopped=None,
    chunk_transfer_adapter=None,
    inter_stage_output=None,
):
    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False

    def stop_request(request: Request, _token_ids: list[int]):
        request.status = RequestStatus.FINISHED_STOPPED
        return [42], True

    sched._update_request_with_output.side_effect = stop_request
    # vLLM 0.26 returns (kv_xfer_params, ec_xfer_params); an unconfigured
    # MagicMock iterates empty and fails to unpack at the call site.
    sched._free_request.return_value = (None, None)
    if handle_stopped is None:
        sched._handle_stopped_request.return_value = session_finished
    else:
        sched._handle_stopped_request.side_effect = handle_stopped
    sched.chunk_transfer_adapter = chunk_transfer_adapter
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.inter_stage_outputs = [inter_stage_output] if inter_stage_output is not None else None

    return OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)


@pytest.mark.parametrize("outstanding_async_tokens", [0, 1, 2])
def test_resumable_segment_stop_reconciles_async_placeholders(
    outstanding_async_tokens: int,
) -> None:
    """A segment stop discards and rolls back only in-flight async tokens."""
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.append_output_token_ids([7, 8])
    session.num_computed_tokens = session.num_tokens + outstanding_async_tokens
    session.num_output_placeholders = outstanding_async_tokens
    session.spec_token_ids = [-1] * outstanding_async_tokens

    _run_resumable_segment_stop(session)

    assert session.async_tokens_to_discard == outstanding_async_tokens
    assert session.num_computed_tokens == session.num_tokens
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    assert session._output_token_ids == []


def test_resumable_session_terminal_is_not_marked_as_segment_boundary() -> None:
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True

    outputs = _run_resumable_segment_stop(session, session_finished=True)

    output = outputs[session.client_index].outputs[0]
    assert output.finish_reason is not None
    assert output.is_segment_finished is False


def test_update_from_output_settles_in_flight_tokens() -> None:
    """vLLM 0.26: schedule() increments num_in_flight_tokens per scheduled
    token; update_from_output must decrement it symmetrically. If the
    decrement is dropped the counter grows monotonically and both readers
    (allocate_slots, _connector_finished) clamp
    max(0, num_computed_tokens - num_in_flight_tokens) to zero forever,
    silently freezing sliding-window block freeing.
    """
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.num_in_flight_tokens = 1  # as left by schedule() for this step

    _run_resumable_segment_stop(session)

    assert session.num_in_flight_tokens == 0


def test_resumable_segment_boundary_keeps_pre_transition_send_watermark() -> None:
    """The old segment's transfer must not observe the next segment's reset.

    A queued streaming update can replace the same mutable Request while the
    stop output is still being handled. The connector needs the confirmed
    token count from before that replacement to avoid dropping the boundary.
    """
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.num_computed_tokens = 26
    adapter = MagicMock()
    adapter._confirmed_num_computed_tokens.return_value = 26
    inter_stage_output = {"codes": {"audio": [7]}}

    def replace_with_next_segment(request: Request) -> bool:
        request.num_computed_tokens = 0
        request._omni_segment_generation = 1
        request.status = RequestStatus.WAITING
        return False

    _run_resumable_segment_stop(
        session,
        handle_stopped=replace_with_next_segment,
        chunk_transfer_adapter=adapter,
        inter_stage_output=inter_stage_output,
    )

    adapter.save_async.assert_called_once_with(
        inter_stage_output,
        session,
        True,
        new_token_ids=[42],
        confirmed_num_computed_tokens=26,
        segment_generation=0,
    )


def test_running_decode_step_without_inter_stage_payload_does_not_raise() -> None:
    """A decode step that neither stops nor carries an inter-stage payload.

    ``finished`` is only assigned when the request stops, yet the async-chunk
    save condition reads it for every request, so this step used to raise
    ``UnboundLocalError: cannot access local variable 'finished'``.
    """
    session = _make_request()
    session.status = RequestStatus.RUNNING

    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False
    sched._update_request_with_output.return_value = ([42], False)
    sched._process_kv_transfer_trigger.return_value = False
    sched.chunk_transfer_adapter = MagicMock()
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.inter_stage_outputs = None

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)

    # Nothing to hand downstream: no payload, no segment boundary, not finished.
    sched.chunk_transfer_adapter.save_async.assert_not_called()


def test_queued_streaming_update_on_async_stop_fences_in_flight_once() -> None:
    """Native duplex + async: the next unit is usually already queued when the
    current one stops, so _handle_stopped_request applies the update and the
    stop site below it fences the same in-flight decode a second time. An
    accumulated residue outlives the drain and swallows the next unit's
    listen/speak, which the client sees as silence until its timeout.
    """
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.append_output_token_ids([7])
    session.num_computed_tokens = 4
    session.num_output_placeholders = 1
    # This step's frame (settled to 0 by update_from_output) plus one extra
    # async decode still in flight.
    session.num_in_flight_tokens = 2

    sched = _make_scheduler(stage_id=0)
    sched._enqueue_waiting_request = MagicMock()

    def handle_stopped(request: Request) -> bool:
        sched._update_request_as_session(request, _make_update([10, 20]))
        return False

    _run_resumable_segment_stop(session, handle_stopped=handle_stopped)

    # Exactly the one unreported decode, so the drain reaches zero before the
    # new segment's first frame arrives.
    assert session.num_stale_output_tokens == session.num_in_flight_tokens == 1


def test_stale_async_frame_is_dropped_before_output_processing() -> None:
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.num_in_flight_tokens = 2
    session.num_computed_tokens = session.num_tokens + 1
    session.num_output_placeholders = 1
    session.async_tokens_to_discard = 1
    session.sampling_params = SimpleNamespace(num_logprobs=1)
    num_computed_tokens = session.num_computed_tokens
    num_output_placeholders = session.num_output_placeholders

    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False

    def discard_stale_output(request: Request, token_ids: list[int]) -> tuple[list[int], bool]:
        request.async_tokens_to_discard = 0
        return token_ids, False

    sched._update_request_with_output.side_effect = discard_stale_output
    sched._process_kv_transfer_trigger.return_value = False
    sched.chunk_transfer_adapter = MagicMock()
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.multimodal_outputs = None
    model_runner_output.inter_stage_outputs = [{"hidden": object()}]
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)

    assert session.async_tokens_to_discard == 0
    assert session.status == RequestStatus.RUNNING
    assert session.num_computed_tokens == num_computed_tokens
    assert session.num_output_placeholders == num_output_placeholders
    sched.chunk_transfer_adapter.save_async.assert_not_called()

    session.sampling_params = SamplingParams(max_tokens=8)
    next_payload = {"hidden": object()}
    sched._update_request_with_output.side_effect = None
    sched._update_request_with_output.return_value = ([43], False)
    model_runner_output.sampled_token_ids = [[43]]
    model_runner_output.inter_stage_outputs = [next_payload]

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)

    sched.chunk_transfer_adapter.save_async.assert_called_once_with(
        next_payload,
        session,
        False,
        new_token_ids=[43],
        confirmed_num_computed_tokens=None,
    )


def test_legacy_stale_async_marker_bypasses_real_placeholder_accounting() -> None:
    request = _make_request()
    request.status = RequestStatus.RUNNING
    request.num_in_flight_tokens = 1
    request.num_output_placeholders = 0
    request.async_tokens_to_discard = 1
    request.num_stale_output_tokens = 0

    sched = OmniARAsyncScheduler.__new__(OmniARAsyncScheduler)
    sched.requests = {request.request_id: request}
    sched.perf_metrics = None
    sched.chunk_transfer_adapter = None
    sched.connector = None
    sched.ec_connector = None
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.finished_req_ids_dict = {}
    sched._new_prompt_len_snapshot = {}
    sched.kv_cache_manager = MagicMock()
    sched.kv_cache_manager.take_events.return_value = None
    sched._remove_stopped_requests_from_queues = MagicMock()
    sched._handle_failed_kv_load_outputs = MagicMock(return_value=[])
    sched._cleanup_kv_tracking = MagicMock()
    sched._aggregate_kv_connector_stats = MagicMock(return_value=None)
    sched._publish_kv_cache_events = MagicMock()
    sched._attach_finished_request_sets = MagicMock()
    sched._attach_scheduler_stats = MagicMock()
    sched._capture_omni_connector_output = MagicMock()

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.multimodal_outputs = None
    model_runner_output.inter_stage_outputs = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {request.request_id: 0}
    model_runner_output.routed_experts = None

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)

    assert request.async_tokens_to_discard == 0
    assert request.num_output_placeholders == 0
    assert request._output_token_ids == []


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_explicit_streaming_payload_replaces_placeholder_prompt() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=False,
        segment_finished_requests=set(),
    )
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    update = _make_update([10, 20])
    update.additional_information = {
        "tts_token_ids": [10, 20],
        "meta": {"replace_streaming_prompt": True},
    }
    update.model_intermediate_buffer = {
        "ids": {"tts": [41, 42, 99]},
        "meta": {"turn_eos_token_id": 99},
    }

    sched._update_request_as_session(session, update)

    assert session.prompt_token_ids == [10, 20]
    assert session.additional_information == update.additional_information
    assert session.model_intermediate_buffer == {
        "ids": {"tts": [41, 42, 99]},
        "meta": {"turn_eos_token_id": 99},
    }
    assert session.status == RequestStatus.WAITING
    sched._free_request_blocks.assert_called_once_with(session)
    sched.encoder_cache_manager.free.assert_called_once_with(session)


def test_explicit_model_intermediate_prompt_replacement_releases_cache_and_watermark() -> None:
    sched = _make_scheduler(stage_id=1)
    session = _make_request()
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=False,
        segment_finished_requests=set(),
        requests_num_chunks_sent={session.external_req_id: 59},
    )
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.prompt_token_ids = [0] * 59
    session._all_token_ids.clear()
    session._all_token_ids.extend(session.prompt_token_ids)
    session.num_prompt_tokens = 59
    session.num_computed_tokens = 59
    session.num_in_flight_tokens = 2
    update = _make_update([0] * 10)
    update.additional_information = None
    update.model_intermediate_buffer = {
        "ids": {"tts": list(range(8))},
        "hidden_states": {"tts": [[0.0]] * 8},
        "meta": {
            "next_stage_prompt_len": 10,
            "replace_streaming_prompt": True,
        },
    }

    sched._update_request_as_session(session, update)

    assert session.prompt_token_ids == [0] * 10
    assert list(session._all_token_ids) == [0] * 10
    assert session.num_prompt_tokens == 10
    assert session.num_computed_tokens == 0
    assert session.num_stale_output_tokens == 2
    assert session.additional_information is None
    assert session.model_intermediate_buffer == update.model_intermediate_buffer
    assert session.status == RequestStatus.WAITING
    assert sched.chunk_transfer_adapter.requests_num_chunks_sent == {}
    sched._free_request_blocks.assert_called_once_with(session)
    sched.encoder_cache_manager.free.assert_called_once_with(session)


def test_ready_async_chunk_prompt_replacement_releases_stale_kv_once() -> None:
    sched = _make_scheduler(stage_id=1)
    session = _make_request()
    session.external_req_id = "external-ar-streaming-test"
    session.num_in_flight_tokens = 2
    # _update_request_as_session() may have already fenced this frame before
    # the connector marks the explicit replacement ready.
    session.num_stale_output_tokens = 2
    session.num_output_placeholders = 2
    session.spec_token_ids = [-1, -1]
    sched.requests = {session.request_id: session}
    sched._inflight_prefills.add(session)
    sched.chunk_transfer_adapter = SimpleNamespace(
        replaced_streaming_prompt_ids={session.request_id},
        requests_with_ready_chunks={session.request_id},
        requests_num_chunks_sent={session.external_req_id: 4090},
    )

    sched._reset_ready_async_chunk_replacements()
    sched._reset_ready_async_chunk_replacements()

    sched._free_request_blocks.assert_called_once_with(session)
    sched.encoder_cache_manager.free.assert_called_once_with(session)
    assert session not in sched._inflight_prefills
    assert session.num_stale_output_tokens == 2
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    assert sched.chunk_transfer_adapter.replaced_streaming_prompt_ids == set()
    assert sched.chunk_transfer_adapter.requests_with_ready_chunks == {session.request_id}
    assert sched.chunk_transfer_adapter.requests_num_chunks_sent == {}


def test_chunk_segment_cleanup_keeps_requeued_resumable_receiver() -> None:
    """A WAITING_FOR_CHUNK stop must not delete its newly parked session."""
    session = _make_request()
    session.resumable = True
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    def queue(*requests):
        result = MagicMock()
        result.requests = set(requests)
        result.add_request.side_effect = result.requests.add
        result.remove_requests.side_effect = result.requests.difference_update
        return result

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(session_mode="duplex"))
    sched.running = []
    sched.waiting = queue()
    sched.skipped_waiting = queue(session)
    sched.num_waiting_for_streaming_input = 1
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=True,
        segment_finished_requests={session.request_id},
    )

    sched._resume_downstream_chunk_receiver(session)
    sched._remove_stopped_requests_from_queues(set(), {session})

    assert session.status == RequestStatus.WAITING
    assert session in sched.waiting.requests
    assert session not in sched.skipped_waiting.requests
    assert sched.num_waiting_for_streaming_input == 0
    assert session.request_id not in sched.chunk_transfer_adapter.segment_finished_requests


@pytest.mark.parametrize(
    ("receives_chunks", "session_mode"),
    [
        (False, "duplex"),
        (True, "turn"),
    ],
)
def test_chunk_segment_cleanup_keeps_explicit_update_stage_parked(
    receives_chunks: bool,
    session_mode: str,
) -> None:
    """Only duplex connector-driven receivers resume without an update."""
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(session_mode=session_mode))
    sched.num_waiting_for_streaming_input = 1
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=receives_chunks,
        segment_finished_requests={session.request_id},
    )
    sched.skipped_waiting = MagicMock()
    sched._enqueue_waiting_request = MagicMock()

    sched._resume_downstream_chunk_receiver(session)

    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert sched.num_waiting_for_streaming_input == 1
    assert session.request_id not in sched.chunk_transfer_adapter.segment_finished_requests
    sched.skipped_waiting.remove_requests.assert_not_called()
    sched._enqueue_waiting_request.assert_not_called()
