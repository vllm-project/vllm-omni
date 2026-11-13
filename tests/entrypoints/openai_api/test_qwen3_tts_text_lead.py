# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Qwen3-TTS ``text_lead`` speech request field: typed, validated, passed to the engine."""

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The assistant template wraps a text in 3 + 5 tokens; this text has 3.
_TEXT_IDS = list(range(8 + 3))


def _adapter(mocker) -> Qwen3TTSAdapter:
    server = mocker.Mock()
    server._tts_tokenizer = lambda _text, **_kwargs: {"input_ids": _TEXT_IDS}
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
        # The lead rows may end with the text END: at most 3 ids + 1.
        ("CustomVoice", {"text_lead": 4, "non_streaming_mode": False}, None),
        ("CustomVoice", {"text_lead": 5, "non_streaming_mode": False}, "exceeds"),
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
    assert params["text_lead"] == [2]
    assert "text_lead" not in adapter._build_tts_params(_request(language="English", text_lead=0))
    assert "text_lead" not in adapter._build_tts_params(_request(language="English"))
