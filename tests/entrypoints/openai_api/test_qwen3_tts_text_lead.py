# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Qwen3-TTS ``text_lead`` speech request field: typed, validated, passed to the engine."""

import asyncio

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import PRECOMPUTED_TEXT_IDS_KEY
from vllm_omni.model_executor.models.qwen3_tts.text_stream import TEXT_LEAD_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The assistant template wraps a text in 3 + 5 tokens; this text has 3.
_TEXT_IDS = list(range(8 + 3))


def _tokenize(_text, **_kwargs):
    return {"input_ids": _TEXT_IDS}


def _adapter(mocker) -> Qwen3TTSAdapter:
    server = mocker.Mock()
    server._tts_tokenizer = _tokenize
    return Qwen3TTSAdapter(SpeechServingContext(server=server))


def _request(**fields) -> OpenAICreateSpeechRequest:
    return OpenAICreateSpeechRequest(input="three word text", voice="ryan", **fields)


@pytest.mark.parametrize("value", [-1, 1.5, "two"])
def test_text_lead_is_a_non_negative_integer(value):
    with pytest.raises(ValidationError):
        _request(text_lead=value)


@pytest.mark.parametrize(
    ("task_type", "fields", "message"),
    [
        ("CustomVoice", {"text_lead": None}, None),
        ("CustomVoice", {"text_lead": 0}, None),
        ("CustomVoice", {"text_lead": 2, "non_streaming_mode": False}, None),
        # CustomVoice defaults to the non-streaming prompt, which has no lead.
        ("CustomVoice", {"text_lead": 2}, "non_streaming_mode=false"),
        ("CustomVoice", {"text_lead": 2, "non_streaming_mode": True}, "non_streaming_mode=false"),
        ("Base", {"text_lead": 2, "non_streaming_mode": False}, "only valid for CustomVoice"),
        ("VoiceDesign", {"text_lead": 2, "non_streaming_mode": False}, "only valid for CustomVoice"),
    ],
)
def test_text_lead_is_checked_against_the_prompt_layout(mocker, task_type, fields, message):
    error = _adapter(mocker)._validate_text_lead(_request(**fields), task_type)
    if message is None:
        assert error is None
    else:
        assert message in error


def test_validation_does_no_tokenizer_work_on_the_event_loop(mocker):
    """validate() runs on the request handler's event loop; loading or running a tokenizer there blocks it."""
    adapter = _adapter(mocker)
    adapter.ctx.server._tts_tokenizer = mocker.Mock(side_effect=AssertionError("tokenized on the event loop"))
    adapter._load_text_tokenizer = mocker.Mock(side_effect=AssertionError("loaded a tokenizer on the event loop"))
    assert adapter._validate_text_lead(_request(text_lead=5, non_streaming_mode=False), "CustomVoice") is None


@pytest.mark.parametrize(
    ("lead", "message"),
    # The lead rows may end with the text END: at most 3 ids + 1.
    [(0, None), (4, None), (5, "exceeds the text's 3 tokens")],
)
def test_the_text_lead_cannot_pass_the_end_of_the_tokenized_text(lead, message):
    params = {TEXT_LEAD_KEY: [lead], PRECOMPUTED_TEXT_IDS_KEY: [_TEXT_IDS]}
    error = Qwen3TTSAdapter._text_lead_bound_error(params)
    if message is None:
        assert error is None
    else:
        assert message in error


@pytest.fixture
def speech_server(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path))
    engine_client = mocker.MagicMock()
    engine_client.errored = False
    engine_client.stage_configs = []
    engine_client.tts_max_instructions_length = None
    models = mocker.MagicMock()
    models.is_base_model.return_value = True
    server = OmniOpenAIServingSpeech(engine_client=engine_client, models=models, request_logger=mocker.MagicMock())
    adapter = Qwen3TTSAdapter(SpeechServingContext(server=server, engine_client=engine_client))
    yield server, adapter
    server.shutdown()


def test_the_engine_gets_the_text_lead_only_when_it_is_set(speech_server):
    _, adapter = speech_server
    params = adapter._build_tts_params(_request(language="English", non_streaming_mode=False, text_lead=2))
    assert params[TEXT_LEAD_KEY] == [2]
    assert TEXT_LEAD_KEY not in adapter._build_tts_params(_request(language="English", text_lead=0))
    assert TEXT_LEAD_KEY not in adapter._build_tts_params(_request(language="English"))


def test_build_refuses_a_text_lead_past_the_text_end(speech_server):
    """The bound is checked on the ids the prompt length estimate tokenized off the event loop."""
    server, adapter = speech_server
    server._tts_tokenizer = _tokenize
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(adapter.build(_request(language="English", non_streaming_mode=False, text_lead=5), [], False))
    prepared = asyncio.run(
        adapter.build(_request(language="English", non_streaming_mode=False, text_lead=4), [], False)
    )
    assert prepared.tts_params[TEXT_LEAD_KEY] == [4]
