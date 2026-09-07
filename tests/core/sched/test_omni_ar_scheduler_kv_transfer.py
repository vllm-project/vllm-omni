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


def _run_finished_save_step(mocker, request, *, inter_stage_outputs=None):
    adapter = mocker.MagicMock()
    adapter._confirmed_num_computed_tokens.return_value = 1

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
    return adapter


@pytest.mark.parametrize(
    ("final_stage_id", "force_kv_transfer", "expect_save"),
    [
        pytest.param(0, False, False, id="stage-zero-final"),
        pytest.param(1, False, True, id="downstream-final"),
    ],
)
def test_finish_saves_async_only_with_downstream_consumer(mocker, final_stage_id, force_kv_transfer, expect_save):
    request = _make_chunk_request(final_stage_id, force_kv_transfer=force_kv_transfer)
    adapter = _run_finished_save_step(mocker, request)

    assert adapter.save_async.called is expect_save


def test_stage_zero_final_skips_inter_stage_output(mocker):
    warn = mocker.patch("vllm_omni.core.sched.omni_ar_scheduler.logger.warning")
    request = _make_chunk_request(0)
    adapter = _run_finished_save_step(mocker, request, inter_stage_outputs=[{"codes": 1}])

    adapter.save_async.assert_not_called()
    assert any("inter_stage_output is present" in str(call) for call in warn.call_args_list)


def test_streaming_session_update_invalidates_omits_kv_transfer_cache():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    scheduler.deferred_waiting = set()
    scheduler.kv_holding_waiting = []
    scheduler.log_stats = False
    tagged = apply_omni_final_stage_metadata(_make_engine_request(), final_stage_id=0)

    class Session(SimpleNamespace):
        __hash__ = object.__hash__

    session = Session(
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


class _HashableNamespace(SimpleNamespace):
    """Identity-hashable request stub matching vLLM Request semantics."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _make_free_request_scheduler(status: RequestStatus):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    request = _HashableNamespace(
        request_id="req",
        client_index=0,
        status=status,
        num_computed_tokens=10,
        num_output_placeholders=0,
        additional_information=None,
        is_finished=lambda: True,
    )
    scheduler._omits_kv_transfer_cache = {}
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _request: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._new_prompt_len_snapshot = {}
    scheduler._connector_finished = lambda _request: (False, None)
    scheduler._should_transfer_kv_for_request = lambda _request_id: True
    scheduler.requests_needing_kv_transfer = {}
    scheduler.waiting_for_transfer_free = set()
    scheduler.active_kv_transfers = set()
    scheduler.pending_stop_after_extraction = set()
    scheduler.transfer_triggered_requests = set()
    scheduler.input_coordinator = None
    freed: list[str] = []
    scheduler._free_blocks = lambda req: freed.append(req.request_id)
    return scheduler, request, freed


def test_decode_to_dit_transfer_keeps_prefill_and_decode_blocks():
    """The DiT export must include blocks allocated for the imported prefix."""
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.requests_needing_kv_transfer = {}
    scheduler._should_transfer_kv_for_request = lambda _request_id: True
    scheduler.kv_cache_manager = SimpleNamespace(
        # Blocks 10 and 11 represent the imported Prefill prefix; block 12
        # contains locally generated Decode tokens. Block 13 is unused tail.
        get_block_ids=lambda _request_id: ([10, 11, 12, 13],),
    )
    scheduler.cache_config = SimpleNamespace(block_size=4)

    scheduler._mark_request_for_kv_transfer("req", seq_len=10)

    assert scheduler.requests_needing_kv_transfer["req"] == {
        "seq_len": 10,
        "block_ids": [10, 11, 12],
    }


@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.FINISHED_ABORTED,
        RequestStatus.FINISHED_ERROR,
        RequestStatus.FINISHED_IGNORED,
    ],
)
def test_failed_request_does_not_start_downstream_kv_transfer(status: RequestStatus):
    scheduler, request, freed = _make_free_request_scheduler(status)
    scheduler.requests_needing_kv_transfer["req"] = {
        "seq_len": 10,
        "block_ids": [10],
    }
    scheduler.pending_stop_after_extraction.add("req")
    scheduler.transfer_triggered_requests.add("req")

    result = OmniARScheduler._free_request(scheduler, request)

    assert result == (None, None)
    assert scheduler.requests_needing_kv_transfer == {}
    assert "req" not in scheduler.pending_stop_after_extraction
    assert "req" not in scheduler.transfer_triggered_requests
    assert freed == ["req"]


def test_successful_request_still_starts_downstream_kv_transfer():
    scheduler, request, freed = _make_free_request_scheduler(RequestStatus.FINISHED_STOPPED)

    def mark_for_transfer(request_id: str, seq_len: int):
        scheduler.requests_needing_kv_transfer[request_id] = {
            "seq_len": seq_len,
            "block_ids": [10],
        }

    scheduler._mark_request_for_kv_transfer = mark_for_transfer

    result = OmniARScheduler._free_request(scheduler, request)

    assert result == (
        {
            "past_key_values": [10],
            "kv_metadata": {"seq_len": 10, "block_ids": [10]},
        },
        None,
    )
    assert "req" in scheduler.waiting_for_transfer_free
    assert freed == []


def test_aborted_request_keeps_blocks_until_active_transfer_is_acknowledged():
    scheduler, request, freed = _make_free_request_scheduler(RequestStatus.FINISHED_ABORTED)
    scheduler.active_kv_transfers.add("req")
    scheduler.transfer_triggered_requests.add("req")

    result = OmniARScheduler._free_request(scheduler, request)

    assert result == (None, None)
    assert "req" in scheduler.active_kv_transfers
    assert "req" in scheduler.waiting_for_transfer_free
    assert "req" in scheduler.transfer_triggered_requests
    assert freed == []
