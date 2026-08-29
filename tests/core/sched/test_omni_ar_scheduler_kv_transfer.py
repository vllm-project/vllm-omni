from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
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


def _native_pd_trigger_scheduler() -> OmniARScheduler:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.kv_transfer_criteria = {"type": "prefill_finished", "stop_after_transfer": True}
    scheduler.waiting_for_transfer_free = set()
    scheduler.transfer_triggered_requests = set()
    scheduler.pending_stop_after_extraction = set()
    scheduler.active_kv_transfers = set()
    scheduler._pd_prefill_submit_ready_requests = set()
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
    )
    scheduler._request_omits_kv_transfer_to_next_stage = lambda _request: False
    return scheduler


def _trigger_request(status: RequestStatus) -> SimpleNamespace:
    request = SimpleNamespace(
        request_id="req",
        status=status,
        num_computed_tokens=70,
        num_output_placeholders=0,
        num_prompt_tokens=70,
    )
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    return request


def test_native_pd_trigger_preserves_length_capped_status_for_mooncake_send():
    scheduler = _native_pd_trigger_scheduler()
    request = _trigger_request(RequestStatus.FINISHED_LENGTH_CAPPED)

    assert scheduler._process_kv_transfer_trigger(request, [123])

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.request_id in scheduler._pd_prefill_submit_ready_requests


def test_native_pd_trigger_stops_an_unfinished_request():
    scheduler = _native_pd_trigger_scheduler()
    request = _trigger_request(RequestStatus.RUNNING)

    assert scheduler._process_kv_transfer_trigger(request, [123])

    assert request.status == RequestStatus.FINISHED_STOPPED


def test_stage_zero_request_omits_kv_transfer():
    omits_transfer, metadata = _request_omits_kv_transfer(force_kv_transfer=False)

    assert omits_transfer
    assert "omni_force_kv_transfer" not in metadata


def test_cfg_companion_forces_kv_transfer_without_downstream_payload():
    omits_transfer, metadata = _request_omits_kv_transfer(force_kv_transfer=True)

    assert not omits_transfer
    assert metadata["omni_final_stage_id"] == 0
    assert metadata["omni_force_kv_transfer"] is True
