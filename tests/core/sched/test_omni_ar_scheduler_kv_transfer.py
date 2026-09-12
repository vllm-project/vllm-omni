# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType, SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import RequestStatus

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


def _request_omits_kv_transfer(*, force_kv_transfer: bool) -> tuple[bool, dict]:
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
    result = scheduler._request_omits_kv_transfer_to_next_stage(request)
    metadata = deserialize_additional_information(tagged.additional_information)
    return result, metadata


def test_stage_zero_request_omits_kv_transfer():
    omits_transfer, metadata = _request_omits_kv_transfer(force_kv_transfer=False)

    assert omits_transfer
    assert "omni_force_kv_transfer" not in metadata


def test_cfg_companion_forces_kv_transfer_without_downstream_payload():
    omits_transfer, metadata = _request_omits_kv_transfer(force_kv_transfer=True)

    assert not omits_transfer
    assert metadata["omni_final_stage_id"] == 0
    assert metadata["omni_force_kv_transfer"] is True


class _ChunkRequest(SimpleNamespace):
    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        return isinstance(other, _ChunkRequest) and other.request_id == self.request_id


def _make_chunk_request(final_stage_id: int):
    tagged = apply_omni_final_stage_metadata(_make_engine_request(), final_stage_id=final_stage_id)
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


def _run_finished_save_step(mocker, request):
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
    sched._omits_kv_transfer_cache = {}

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
    sched._request_omits_kv_transfer_to_next_stage = MethodType(
        OmniARScheduler._request_omits_kv_transfer_to_next_stage,
        sched,
    )

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
    model_runner_output.inter_stage_outputs = None

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)
    return adapter, sched, connector


def test_stage_zero_final_finish_does_not_save_async(mocker):
    request = _make_chunk_request(0)
    adapter, sched, connector = _run_finished_save_step(mocker, request)

    adapter.save_async.assert_not_called()
    assert request.request_id not in sched._omits_kv_transfer_cache
    assert connector._pending_keys == set()


def test_downstream_finish_still_saves_async(mocker):
    request = _make_chunk_request(1)
    adapter, sched, connector = _run_finished_save_step(mocker, request)

    adapter.save_async.assert_called_once()
    assert request.request_id not in sched._omits_kv_transfer_cache
    assert f"{request.request_id}_finished" in connector._pending_keys


def test_streaming_session_update_invalidates_omits_kv_transfer_cache():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {"req": True}
    scheduler.skipped_waiting = []
    scheduler.log_stats = False
    session = SimpleNamespace(
        request_id="req",
        additional_information="old",
        status=RequestStatus.WAITING,
    )
    update = SimpleNamespace(
        additional_information={"omni_final_stage_id": 1},
        model_intermediate_buffer=None,
        arrival_time=0.0,
        sampling_params=None,
    )

    OmniSchedulerMixin._finish_streaming_session_update(scheduler, session, update)

    assert scheduler._omits_kv_transfer_cache == {}
    assert session.additional_information == {"omni_final_stage_id": 1}
