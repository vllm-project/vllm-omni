from types import SimpleNamespace

import torch

from vllm_omni.core.sched.output import OmniNewRequestData


def test_omni_new_request_data_copies_payloads():
    prompt_embeds = torch.randn(2, 3)
    additional_information = {
        "speaker": ["test"],
        "codes": torch.tensor([1, 2], dtype=torch.int64),
    }
    request = SimpleNamespace(
        request_id="req-1",
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
