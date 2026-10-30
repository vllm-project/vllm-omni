# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage routing metadata must not erase the caller's speech payload."""

from dataclasses import dataclass, field

import pytest

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.entrypoints.openai.drvexo_speech import speech_prompt
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class RequestState:
    additional_information_cpu: dict = field(default_factory=dict)


@dataclass
class ScheduledRequest:
    req_id: str
    model_intermediate_buffer: dict
    additional_information: object
    prompt_embeds: object = None


@dataclass
class ScheduledBatch:
    scheduled_new_reqs: list[ScheduledRequest]


@pytest.mark.parametrize("language", ["Auto", "Chinese", "English"])
def test_decode_preserves_fixed_speech_payload_alongside_final_stage_metadata(language):
    runner = object.__new__(OmniGPUModelRunner)
    runner.requests = {"speech": RequestState()}
    runner.model_intermediate_buffer = {}
    buffer = speech_prompt("Read this sentence.", language)["model_intermediate_buffer"]
    batch = ScheduledBatch(
        [ScheduledRequest("speech", buffer, serialize_additional_information({"omni_final_stage_id": 2}))]
    )
    runner._decode_and_store_request_payloads(batch)
    actual = runner.model_intermediate_buffer["speech"]
    assert actual["drvexo_fixed_reply"] is True
    assert actual["text"] == ["Read this sentence."]
    assert actual["language"] == [language]
    assert actual["omni_final_stage_id"] == 2
    assert runner.requests["speech"].additional_information_cpu == actual
