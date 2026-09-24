from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniNewRequestData

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_new_request_data_copies_payloads():
    prompt_embeds = torch.randn(2, 3)
    additional_information = {
        "speaker": ["test"],
        "codes": torch.tensor([1, 2], dtype=torch.int64),
    }
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="ext-1",
        prompt_token_ids=[101, 102],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
        additional_information=additional_information,
    )

    data = OmniNewRequestData.from_request(request, ([0, 1],), prefill_token_ids=[101, 102])

    assert data.prompt_embeds is prompt_embeds
    assert data.additional_information is additional_information
    assert data.prefill_token_ids == [101, 102]


def test_omni_new_request_data_allows_missing_payloads():
    request = SimpleNamespace(
        request_id="req-2",
        external_req_id="ext-2",
        prompt_token_ids=[201, 202],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        additional_information=None,
    )

    data = OmniNewRequestData.from_request(request, ([0],), prefill_token_ids=None)

    assert data.prompt_embeds is None
    assert data.additional_information is None


def test_from_base_preserves_every_upstream_field_and_adds_omni_payloads():
    prompt_embeds = torch.randn(2, 3)
    base = NewRequestData(
        req_id="req-base",
        prompt_token_ids=[1, 2],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([3, 4],),
        num_computed_tokens=1,
        lora_request=None,
        prompt_embeds=prompt_embeds,
        prompt_is_token_ids=False,
        prefill_token_ids=[1],
    )
    request = SimpleNamespace(
        external_req_id="external-base",
        additional_information={"payload": "value"},
        model_intermediate_buffer={"hidden": "state"},
    )

    data = OmniNewRequestData.from_base(base, request)

    for field_name in NewRequestData.__dataclass_fields__:
        assert getattr(data, field_name) is getattr(base, field_name)
    assert data.external_req_id == "external-base"
    assert data.additional_information == {"payload": "value"}
    assert data.model_intermediate_buffer == {"hidden": "state"}


def test_rewrap_converts_fallback_entries_and_keeps_fast_path_entries():
    base = NewRequestData(
        req_id="req-fallback",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([2],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=True,
        prefill_token_ids=[1],
    )
    fast_path = OmniNewRequestData(
        req_id="req-fast",
        prompt_token_ids=[3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([4],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=True,
        prefill_token_ids=[3],
        additional_information={"fast": True},
    )
    scheduler = OmniSchedulerMixin()
    scheduler.requests = {
        "req-fallback": SimpleNamespace(
            external_req_id="external-fallback",
            additional_information={"fallback": True},
            model_intermediate_buffer=None,
        )
    }
    output = SimpleNamespace(scheduled_new_reqs=[base, fast_path])

    scheduler._rewrap_scheduled_new_reqs(output)

    fallback_data, fast_path_data = output.scheduled_new_reqs
    assert isinstance(fallback_data, OmniNewRequestData)
    assert fallback_data.prefill_token_ids == [1]
    assert fallback_data.additional_information == {"fallback": True}
    assert fast_path_data is fast_path
