from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest

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
