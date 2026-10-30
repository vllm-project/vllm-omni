# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.serialization import deserialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_processed_stage_request_preserves_runner_payload_with_routing_metadata():
    orchestrator = object.__new__(Orchestrator)
    request = EngineCoreRequest(
        request_id="runner-payload",
        prompt_token_ids=[4, 5, 6],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    buffer = {"text": ["Read this sentence."], "drvexo_fixed_reply": True}
    upgraded = orchestrator._upgrade_processed_stage_request(
        request,
        {"model_intermediate_buffer": buffer, "additional_information": {"global_request_id": ["runner-payload"]}},
    )
    assert upgraded.model_intermediate_buffer == buffer
    assert deserialize_additional_information(upgraded.additional_information) == {
        "global_request_id": ["runner-payload"]
    }
