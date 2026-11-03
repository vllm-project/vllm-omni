# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from unittest.mock import Mock

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.lfm3_omni import Lfm3OmniTTSAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_adapter():
    server = Mock(spec=[])
    return Lfm3OmniTTSAdapter(SpeechServingContext(server=server, engine_client=Mock()))


def test_detection_requires_codec_stage():
    assert resolve_adapter("lfm3_omni") is Lfm3OmniTTSAdapter
    assert Lfm3OmniTTSAdapter.stage_serves_speech("lfm3_temporal", {"lfm3_temporal", "lfm3_code2wav"})
    assert not Lfm3OmniTTSAdapter.stage_serves_speech("lfm3_temporal", {"lfm3_temporal"})


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"input": " "}, "empty"),
        ({"voice": "alloy"}, "named voice"),
        ({"ref_audio": "file:///reference.wav"}, "voice cloning"),
        ({"extra_params": {"audio_temperature": -1}}, "audio_temperature"),
        ({"extra_params": {"audio_repetition_penalty": None}}, "audio_repetition_penalty"),
        ({"extra_params": {"audio_repetition_window": 1.5}}, "audio_repetition_window"),
        ({"extra_params": {"audio_top_k": -1}}, "audio_top_k"),
    ],
)
def test_unsupported_speech_requests_fail_before_inference(fields, message):
    request = OpenAICreateSpeechRequest(**{"input": "Hello", **fields})
    assert message in make_adapter().validate(request)


def test_prompt_marker_and_token_limit():
    adapter = make_adapter()
    tokenizer = adapter._tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    tokenizer.encode.return_value = [1, 2, 125012]
    request = OpenAICreateSpeechRequest(input="Hello", instructions="Speak calmly.", max_new_tokens=48)
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, [], False))
    assert prepared.prompt == {"prompt_token_ids": [1, 2, 125012]}
    tokenizer.encode.assert_called_once_with("rendered prompt<|audio_start|>", add_special_tokens=False)
    assert tokenizer.apply_chat_template.call_args.args[0] == [
        {"role": "system", "content": "Speak calmly."},
        {"role": "user", "content": "Hello"},
    ]
    params = [SamplingParams(max_tokens=512), SamplingParams(max_tokens=1)]
    updated = adapter.apply_sampling_overrides(params, request)
    assert updated[0].max_tokens == 48
    assert updated[1].max_tokens == 1
    assert params[0].max_tokens == 512
