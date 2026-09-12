# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests guarding OmniEngineCoreRequest against upstream field drift.

``OmniEngineCoreRequest.from_request`` mirrors upstream vLLM ``EngineCoreRequest``
fields into the omni subclass by hand. When upstream vLLM adds, renames, or
removes a field, the manual copy can silently drop it. These tests pin field
parity so the drift is caught at test time instead of at runtime after a
dependency bump.
"""

from __future__ import annotations

import pytest
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import AdditionalInformationPayload, OmniEngineCoreRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BASE_FIELDS = EngineCoreRequest.__struct_fields__
OMNI_FIELDS = OmniEngineCoreRequest.__struct_fields__


def test_omni_engine_core_request_inherits_every_base_field() -> None:
    missing = [name for name in BASE_FIELDS if name not in OMNI_FIELDS]
    assert missing == []


def _build_request_with_all_fields() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt="salt",
        data_parallel_rank=0,
        prompt_embeds=None,
        prompt_is_token_ids=[True, False],
        client_index=1,
        current_wave=2,
        priority=3,
        trace_headers={"k": "v"},
        resumable=True,
        external_req_id="ext-1",
        reasoning_ended=True,
        reasoning_parser_kwargs={"a": 1},
        abort_immediately=True,
        session_id="sess-1",
    )


def test_from_request_copies_every_base_field() -> None:
    """Every upstream field must survive the manual copy in from_request."""
    request = _build_request_with_all_fields()
    copied = OmniEngineCoreRequest.from_request(request)
    for name in BASE_FIELDS:
        assert getattr(copied, name) == getattr(request, name), name


def test_from_request_keeps_omni_specific_fields() -> None:
    request = _build_request_with_all_fields()
    copied = OmniEngineCoreRequest.from_request(
        request,
        additional_information=AdditionalInformationPayload(entries={}),
        model_intermediate_buffer={"m": 1},
    )
    assert copied.additional_information is not None
    assert copied.additional_information.entries == {}
    assert copied.model_intermediate_buffer == {"m": 1}
