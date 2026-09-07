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


def test_stage_zero_request_omits_kv_transfer():
    omits_transfer, metadata = _request_omits_kv_transfer(force_kv_transfer=False)

    assert omits_transfer
    assert "omni_force_kv_transfer" not in metadata


def test_cfg_companion_forces_kv_transfer_without_downstream_payload():
    omits_transfer, metadata = _request_omits_kv_transfer(force_kv_transfer=True)

    assert not omits_transfer
    assert metadata["omni_final_stage_id"] == 0
    assert metadata["omni_force_kv_transfer"] is True


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
