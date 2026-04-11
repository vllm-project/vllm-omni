from unittest.mock import Mock

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_core_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[1, 1, 1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_build_add_request_message_preserves_additional_information():
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("speech",)

    input_processor = Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = Mock()
    engine.output_processors = [output_processor]

    prompt = {
        "prompt_token_ids": [1, 1, 1],
        "additional_information": {
            "text": ["hello world"],
            "speaker": ["vivian"],
        },
    }

    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=0.0,
    )

    request = msg["prompt"]
    assert isinstance(request, OmniEngineCoreRequest)
    assert request.external_req_id == "req-1"
    assert request.additional_information is not None
    assert request.additional_information.entries["text"].list_data == ["hello world"]
    assert request.additional_information.entries["speaker"].list_data == ["vivian"]
    output_processor.add_request.assert_called_once()


def test_build_add_request_message_with_resumable_streaming():
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("generate",)

    input_processor = Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = Mock()
    engine.output_processors = [output_processor]

    msg = engine._build_add_request_message(
        request_id="req-stream",
        prompt={"prompt_token_ids": [1, 2, 3]},
        sampling_params_list=[params],
        final_stage_id=0,
        resumable=True,
        message_type="streaming_update",
    )

    assert msg["type"] == "streaming_update"
    input_processor.process_inputs.assert_called_once()
    assert input_processor.process_inputs.call_args.kwargs["resumable"] is True
