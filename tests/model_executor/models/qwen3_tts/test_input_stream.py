# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU admission contracts for Qwen3-TTS active token input."""

from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.qwen3_tts.input_stream import Qwen3TTSTextInput
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration as Talker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request_and_config():
    request = SimpleNamespace(
        request_id="request",
        additional_information={"input_stream": {"token_ids": [31, 32]}},
        model_intermediate_buffer=None,
        resumable=False,
        max_tokens=8,
        num_prompt_tokens=5,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            final_output=True,
            engine_output_type="audio",
            use_v2_model_runner=True,
            max_model_len=32,
            hf_config=SimpleNamespace(talker_config=SimpleNamespace(text_vocab_size=64)),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        num_speculative_tokens=0,
    )
    return request, config


@pytest.mark.parametrize("channel", ["plain", "wire", "native"])
def test_initial_stream_survives_request_payload_transport(channel):
    request, config = request_and_config()
    if channel == "wire":
        request.additional_information = serialize_additional_information(request.additional_information)
    elif channel == "native":
        request.model_intermediate_buffer = request.additional_information
        request.additional_information = None
    stream = Talker.create_input_stream(request, config)
    assert stream.take_update(reset=True) == {"token_ids": [31, 32], "final": False}
    stream.append([33])
    assert stream.take_update(reset=False) == {"token_ids": [33], "final": False}


@pytest.mark.parametrize(
    "unsupported",
    ["v1", "prefix", "speculation", "tp", "dp", "resumable", "base", "layout"],
)
def test_unsupported_streams_are_rejected_during_admission(unsupported):
    request, config = request_and_config()
    if unsupported == "v1":
        config.model_config.use_v2_model_runner = False
    elif unsupported == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif unsupported == "speculation":
        config.num_speculative_tokens = 1
    elif unsupported == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif unsupported == "dp":
        config.parallel_config.data_parallel_size = 2
    elif unsupported == "resumable":
        request.resumable = True
    elif unsupported == "base":
        request.additional_information["task_type"] = ["Base"]
    else:
        request.additional_information["non_streaming_mode"] = [True]
    with pytest.raises(ValueError):
        Talker.create_input_stream(request, config)


def test_nonfinal_latent_talker_accepts_the_same_text_stream():
    request, config = request_and_config()
    config.model_config.final_output = False
    config.model_config.engine_output_type = "latent"
    stream = Talker.create_input_stream(request, config)
    assert stream.take_update(reset=True) == {"token_ids": [31, 32], "final": False}
    stream.append([33], final=True)
    assert stream.take_update(reset=False) == {"token_ids": [33], "final": True}


@pytest.mark.parametrize("max_tokens,max_model_len,capacity", [(8, 32, 7), (16, 10, 4)])
def test_append_preserves_text_eos_capacity_within_generation_and_context_limits(max_tokens, max_model_len, capacity):
    request, config = request_and_config()
    request.max_tokens = max_tokens
    config.model_config.max_model_len = max_model_len
    stream = Talker.create_input_stream(request, config)
    stream.append([33] * (capacity - 2))
    accepted = list(stream.token_ids)
    with pytest.raises(ValueError, match="capacity"):
        stream.append([34])
    assert len(stream.token_ids) == capacity
    assert stream.token_ids == accepted
    stream.append([], final=True)
    assert stream.take_update(reset=False) == {"token_ids": accepted, "final": True}


def test_input_owns_tokens_and_delivers_only_new_tokens_until_readmission():
    initial = [31]
    stream = Qwen3TTSTextInput(initial, 8, 64)
    initial.append(32)
    assert stream.take_update(reset=True)["token_ids"] == [31]
    tokens = [32, 32]
    stream.append(tokens)
    tokens.clear()
    delta = stream.take_update(reset=False)
    assert delta == {"token_ids": [32, 32], "final": False}
    delta["token_ids"].clear()
    assert stream.take_update(reset=False) is None
    assert stream.take_update(reset=True)["token_ids"] == [31, 32, 32]
    stream.append([], final=True)
    stream.append([], final=True)
    assert stream.take_update(reset=False) == {"token_ids": [], "final": True}
    assert stream.take_update(reset=False) is None
    with pytest.raises(ValueError, match="EOF"):
        stream.append([34])
