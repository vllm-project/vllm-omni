# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import RequestStatus

from tests.helpers.omni_scheduler import bind_omits_transfer_helpers
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from vllm_omni.engine.async_engine_utils import apply_omni_final_stage_metadata
from vllm_omni.engine.serialization import deserialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _omit_flags(*, force_kv_transfer: bool) -> tuple[bool, bool, dict]:
    tagged = apply_omni_final_stage_metadata(
        _make_engine_request(),
        final_stage_id=0,
        force_kv_transfer=force_kv_transfer,
    )
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    request = SimpleNamespace(
        request_id="req",
        additional_information=tagged.additional_information,
    )
    omits_kv = scheduler._request_omits_kv_transfer_to_next_stage(request)
    omits_chunk = scheduler._request_omits_chunk_transfer_to_next_stage(request)
    metadata = deserialize_additional_information(tagged.additional_information)
    return omits_kv, omits_chunk, metadata


def test_stage_zero_request_omits_kv_transfer():
    omits_kv, omits_chunk, metadata = _omit_flags(force_kv_transfer=False)

    assert omits_kv
    assert omits_chunk
    assert "omni_force_kv_transfer" not in metadata


def test_cfg_companion_forces_kv_transfer_without_downstream_payload():
    omits_kv, omits_chunk, metadata = _omit_flags(force_kv_transfer=True)

    assert not omits_kv
    assert omits_chunk
    assert metadata["omni_final_stage_id"] == 0
    assert metadata["omni_force_kv_transfer"] is True


class _ChunkRequest(SimpleNamespace):
    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        return isinstance(other, _ChunkRequest) and other.request_id == self.request_id


def _make_chunk_request(final_stage_id: int, *, force_kv_transfer: bool = False):
    tagged = apply_omni_final_stage_metadata(
        _make_engine_request(),
        final_stage_id=final_stage_id,
        force_kv_transfer=force_kv_transfer,
    )
    request = _ChunkRequest(
        request_id="req-shm-leak",
        additional_information=tagged.additional_information,
        num_in_flight_tokens=1,
        num_stale_output_tokens=0,
        sampling_params=SimpleNamespace(num_logprobs=None),
        has_encoder_inputs=False,
        pooling_params=None,
        status=RequestStatus.RUNNING,
        resumable=False,
        num_output_placeholders=0,
        spec_token_ids=[],
        _output_token_ids=[],
        num_computed_tokens=1,
        stop_reason=None,
        num_nans_in_logits=None,
        client_index=0,
        trace_headers=None,
    )
    request.is_finished = lambda: False
    request.get_finished_reason = lambda: None
    request.take_prefill_stats = lambda: None
    request.take_events = lambda: None
    return request


class _PendingKeyConnector(SharedMemoryConnector):
    """Record put() keys without touching /dev/shm."""

    def put(self, from_stage, to_stage, put_key, data):
        self._pending_keys.add(put_key)
        return True, 1, {"shm": True}


def _run_finished_save_step(mocker, request, *, inter_stage_outputs=None):
    connector = _PendingKeyConnector({})
    adapter = mocker.MagicMock()
    adapter._confirmed_num_computed_tokens.return_value = 1
    adapter.save_async.side_effect = lambda *args, **kwargs: connector.put(
        "0",
        "1",
        f"{request.request_id}_finished",
        {"finished": True},
    )

    sched = mocker.MagicMock()
    sched.requests = {request.request_id: request}
    sched.perf_metrics = None
    sched.defer_block_free = False
    sched.structured_output_manager.should_advance.return_value = False
    sched._update_request_with_output.return_value = ([42], True)
    sched._process_kv_transfer_trigger.return_value = False
    sched._handle_stopped_request.return_value = True

    def _free_request(req, delay_free_blocks=False):
        sched._omits_kv_transfer_cache.pop(req.request_id, None)
        return None, None

    sched._free_request.side_effect = _free_request
    sched.chunk_transfer_adapter = adapter
    sched.running = [request]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None
    bind_omits_transfer_helpers(sched)

    scheduler_output = mocker.MagicMock()
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = mocker.MagicMock()
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {request.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.inter_stage_outputs = inter_stage_outputs

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)
    return adapter, sched, connector


def test_stage_zero_final_finish_does_not_save_async(mocker):
    request = _make_chunk_request(0)
    adapter, _, connector = _run_finished_save_step(mocker, request)

    adapter.save_async.assert_not_called()
    assert connector._pending_keys == set()


def test_stage_zero_final_skips_inter_stage_output(mocker):
    warn = mocker.patch("vllm_omni.core.sched.omni_ar_scheduler.logger.warning")
    request = _make_chunk_request(0)
    adapter, _, connector = _run_finished_save_step(mocker, request, inter_stage_outputs=[{"codes": 1}])

    adapter.save_async.assert_not_called()
    assert connector._pending_keys == set()
    assert warn.called
    assert any("inter_stage_output is present" in str(call) for call in warn.call_args_list)


def test_cfg_companion_finish_does_not_save_async(mocker):
    request = _make_chunk_request(0, force_kv_transfer=True)
    adapter, _, connector = _run_finished_save_step(mocker, request)

    adapter.save_async.assert_not_called()
    assert connector._pending_keys == set()


def test_downstream_finish_still_saves_async(mocker):
    request = _make_chunk_request(1)
    adapter, _, connector = _run_finished_save_step(mocker, request)

    adapter.save_async.assert_called_once()
    assert f"{request.request_id}_finished" in connector._pending_keys


def test_streaming_session_update_invalidates_omits_kv_transfer_cache():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    scheduler.skipped_waiting = []
    scheduler.log_stats = False
    tagged = apply_omni_final_stage_metadata(_make_engine_request(), final_stage_id=0)
    session = SimpleNamespace(
        request_id="req",
        additional_information=tagged.additional_information,
        status=RequestStatus.WAITING,
    )
    assert scheduler._request_omits_chunk_transfer_to_next_stage(session)
    assert "req" in scheduler._omits_kv_transfer_cache
    update = SimpleNamespace(
        additional_information={"omni_final_stage_id": 1},
        model_intermediate_buffer=None,
        arrival_time=0.0,
        sampling_params=None,
    )

    OmniSchedulerMixin._finish_streaming_session_update(scheduler, session, update)

    assert scheduler._omits_kv_transfer_cache == {}
    assert session.additional_information == {"omni_final_stage_id": 1}
    assert not scheduler._request_omits_chunk_transfer_to_next_stage(session)
    assert not scheduler._request_omits_kv_transfer_to_next_stage(session)


def test_omits_cache_misses_when_payload_object_is_replaced():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    tagged = apply_omni_final_stage_metadata(_make_engine_request(), final_stage_id=0)
    request = SimpleNamespace(
        request_id="req",
        additional_information=tagged.additional_information,
    )
    assert scheduler._request_omits_chunk_transfer_to_next_stage(request)
    request.additional_information = {"omni_final_stage_id": 1}

    assert not scheduler._request_omits_chunk_transfer_to_next_stage(request)
