# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Qwen3-TTS ``text_lead`` speech request field: typed, validated, passed to the engine."""

import asyncio

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
    assistant_text_ids,
    num_assistant_text_ids,
)
from vllm_omni.model_executor.models.qwen3_tts.text_stream import TEXT_LEAD_KEY, TextLeadError, text_lead_error

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
    # One helper counts the text's ids for the adapter, the estimate and the builder.
    num_text_ids = num_assistant_text_ids(len(_TEXT_IDS))
    assert num_text_ids == len(assistant_text_ids(_TEXT_IDS)) == 3
    error = text_lead_error({TEXT_LEAD_KEY: [lead], "non_streaming_mode": [False]}, num_text_ids)
    if message is None:
        assert error is None
    else:
        assert message in error


@pytest.mark.parametrize(
    "ids",
    [list(range(11)), np.arange(11), torch.arange(11), np.arange(22).reshape(2, 11), torch.arange(22).reshape(2, 11)],
)
def test_the_text_id_slice_reads_the_last_axis(ids):
    assert np.array_equal(np.asarray(assistant_text_ids(ids)), np.asarray(ids)[..., 3:-5])


def test_the_estimate_checks_a_lead_against_the_task_type_it_lays_out():
    # The estimate counts the layout of its task_type argument, whatever the request's own entry says.
    with pytest.raises(TextLeadError, match="only valid for CustomVoice"):
        Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
            additional_information={
                "text": ["three word text"],
                "instruct": ["calm"],
                "non_streaming_mode": [False],
                TEXT_LEAD_KEY: [2],
            },
            task_type="VoiceDesign",
            tokenize_prompt=lambda text: _TEXT_IDS,
            codec_language_id=None,
            spk_is_dialect=None,
        )


def _estimate(**info):
    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information={"text": ["three word text"], **{key: [value] for key, value in info.items()}},
        task_type=info.get("task_type", "CustomVoice"),
        tokenize_prompt=lambda text: _TEXT_IDS,
        codec_language_id=None,
        spk_is_dialect=None,
    )


def test_an_offline_whole_text_request_learns_its_text_lead_is_too_long_before_it_runs():
    # Offline Omni / AsyncOmni requests set text_lead in additional_information and size their prompt with this
    # estimate (examples/offline_inference/text_to_speech/qwen3_tts/end2end.py), so it refuses the lead there.
    assert _estimate(task_type="CustomVoice", non_streaming_mode=False, text_lead=4)
    with pytest.raises(TextLeadError, match="exceeds the text's 3 tokens"):
        _estimate(task_type="CustomVoice", non_streaming_mode=False, text_lead=5)
    # A streamed text has no length yet: the stage's text_stream_policy bounds its lead at admission.
    assert _estimate(task_type="CustomVoice", non_streaming_mode=False, text_lead=9, text_stream=True)


@pytest.mark.parametrize(
    ("info", "message"),
    [
        ({"task_type": "CustomVoice"}, "non_streaming_mode=false"),
        ({"task_type": "CustomVoice", "non_streaming_mode": True}, "non_streaming_mode=false"),
        ({"task_type": "VoiceDesign", "non_streaming_mode": False, "instruct": "calm"}, "only valid for CustomVoice"),
    ],
)
def test_an_offline_request_learns_its_prompt_ignores_its_text_lead(info, message):
    with pytest.raises(TextLeadError, match=message):
        _estimate(text_lead=2, **info)


def test_a_text_lead_is_not_sent_with_a_fallback_prompt_length(mocker):
    """When the prompt length estimate fails, a text lead goes unchecked and the fallback length fits no layout."""
    adapter = _adapter(mocker)
    adapter._load_text_tokenizer = mocker.Mock()
    adapter.ctx.engine_client = mocker.Mock()
    adapter.ctx.server._tts_tokenizer = mocker.Mock(side_effect=RuntimeError("tokenizer unavailable"))
    with pytest.raises(TextLeadError, match="tokenizer unavailable"):
        adapter._estimate_prompt_len({"text": ["three word text"], "task_type": ["CustomVoice"], TEXT_LEAD_KEY: [2]})
    assert adapter._estimate_prompt_len({"text": ["three word text"], "task_type": ["CustomVoice"]}) == 2048


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
