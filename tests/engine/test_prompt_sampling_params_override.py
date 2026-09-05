# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import (
    Orchestrator,
    build_engine_core_request_from_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_token_prompt_sampling_override_is_request_local():
    base_params = SamplingParams(max_tokens=4096, temperature=1.0)

    request = build_engine_core_request_from_tokens(
        request_id="request-a",
        prompt={
            "prompt_token_ids": [0, 0, 0],
            "sampling_params_override": {"max_tokens": 10},
        },
        params=base_params,
    )

    assert request.sampling_params.max_tokens == 10
    assert request.sampling_params.temperature == 1.0
    assert base_params.max_tokens == 4096


@pytest.mark.parametrize("max_tokens", [10, 100])
def test_resumable_next_stage_request_applies_prompt_budget(max_tokens):
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [
        None,
        SimpleNamespace(stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))),
    ]

    request = orchestrator._build_next_stage_request(
        "request-a",
        1,
        {
            "prompt_token_ids": [0, 0, 0],
            "sampling_params_override": {"max_tokens": max_tokens},
        },
        params=SamplingParams(max_tokens=4096),
        resumable=True,
    )

    assert request.resumable is True
    assert request.sampling_params.max_tokens == max_tokens
