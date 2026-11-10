# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for chat endpoint speaker validation."""

import asyncio
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_omni.engine.duplex.fallback import DuplexFallbackRequest
from vllm_omni.entrypoints.duplex.chat_fallback import build_chat_request
from vllm_omni.utils.speakers import (
    get_supported_speakers_from_hf_config,
    set_prompt_speaker,
    validate_requested_speaker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance._supported_speakers = None
    instance.has_kv_connector = False
    return instance


def _make_hf_config(mocker: MockerFixture, *, speaker_id: dict | None = None, spk_id: dict | None = None):
    hf_config = mocker.MagicMock()
    talker_config = mocker.MagicMock()
    talker_config.speaker_id = speaker_id
    talker_config.spk_id = spk_id
    hf_config.talker_config = talker_config
    return hf_config


def test_validate_requested_speaker_accepts_case_insensitive_value():
    supported = {"vivian", "ethan"}
    assert validate_requested_speaker("Vivian", supported) == "vivian"
    assert validate_requested_speaker(" vivian ", supported) == "vivian"


def test_validate_requested_speaker_rejects_invalid_value_with_supported_list():
    supported = {"vivian", "ethan"}
    with pytest.raises(ValueError, match="Invalid speaker 'uncle_fu'. Supported: ethan, vivian"):
        validate_requested_speaker("uncle_fu", supported)


def test_validate_requested_speaker_skips_validation_when_supported_empty():
    assert validate_requested_speaker("anything", set()) == "anything"
    assert validate_requested_speaker("  ", {"vivian"}) is None


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ({}, {"speaker": ["vivian"]}),
        ({"additional_information": {"language": ["en"]}}, {"language": ["en"], "speaker": ["vivian"]}),
        ({"additional_information": None}, {"speaker": ["vivian"]}),
        ({"additional_information": "legacy"}, {"speaker": ["vivian"]}),
    ],
)
def test_set_prompt_speaker_writes_speaker(prompt, expected):
    set_prompt_speaker(prompt, "vivian")
    assert prompt["additional_information"] == expected


def test_get_supported_speakers_from_hf_config_uses_spk_id_fallback(mocker: MockerFixture):
    hf_config = _make_hf_config(mocker, speaker_id=None, spk_id={"Serena": 0})
    assert get_supported_speakers_from_hf_config(hf_config) == {"serena"}


def test_get_supported_speakers_caches_normalized_keys(mocker: MockerFixture, serving_chat):
    serving_chat.model_config = mocker.MagicMock()
    serving_chat.model_config.hf_config = _make_hf_config(mocker, speaker_id={"Vivian": 0, "Ethan": 1})

    assert serving_chat._get_supported_speakers() == {"vivian", "ethan"}

    # Cached value should be reused even if the config changes afterwards.
    serving_chat.model_config.hf_config.talker_config.speaker_id = {"Serena": 2}
    assert serving_chat._get_supported_speakers() == {"vivian", "ethan"}


def test_resolve_chat_speaker_uses_known_nested_audio_voice(serving_chat):
    request = SimpleNamespace(audio={"voice": "Vivian"})

    assert serving_chat._resolve_chat_speaker(request, {"vivian"}) == "Vivian"


@pytest.mark.parametrize("supported", [None, set()])
def test_resolve_chat_speaker_ignores_nested_voice_without_speaker_table(serving_chat, supported):
    request = SimpleNamespace(audio={"voice": "Vivian"})

    assert serving_chat._resolve_chat_speaker(request, supported) is None


@pytest.fixture
def preprocess_chat(mocker: MockerFixture, serving_chat):
    serving_chat.model_config = mocker.MagicMock(multimodal_config=None)
    serving_chat._supported_speakers = {"aiden", "chelsie", "ethan"}
    serving_chat._needs_multistage_multimodal_split = mocker.Mock(return_value=False)
    serving_chat._attach_minicpmo45_reference_audio = mocker.AsyncMock()
    serving_chat._attach_ming3_reference_audio = mocker.AsyncMock()
    renderer = mocker.MagicMock()
    renderer.tokenizer = object()
    renderer.render_chat_async = mocker.AsyncMock(return_value=(([],), ({"prompt_token_ids": [1]},)))
    chat_params = mocker.MagicMock()
    mocker.patch.object(ChatCompletionRequest, "build_tok_params", return_value=object())
    mocker.patch.object(ChatCompletionRequest, "build_chat_params", return_value=chat_params)

    async def preprocess(request, *, supported=None):
        if supported is not None:
            serving_chat._supported_speakers = supported
        _, prompts = await serving_chat._preprocess_chat(
            request,
            request.messages,
            default_template=None,
            default_template_content_format="string",
            renderer=renderer,
        )
        return prompts[0]

    return preprocess


def _chat_request(audio_voice: str, **extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="qwen",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["audio"],
        audio={"voice": audio_voice, "format": "wav"},
        **extra,
    )


@pytest.mark.asyncio
async def test_chat_unknown_nested_voice_uses_default_speaker(preprocess_chat):
    prompt = await preprocess_chat(_chat_request("alloy"))

    assert "speaker" not in prompt.get("additional_information", {})


@pytest.mark.asyncio
async def test_chat_unknown_nested_voice_preserves_existing_prompt_metadata(preprocess_chat):
    request = _chat_request("alloy", additional_information={"speaker": ["aiden"], "language": ["English"]})

    prompt = await preprocess_chat(request)

    assert prompt["additional_information"] == {"speaker": ["aiden"], "language": ["English"]}


@pytest.mark.asyncio
async def test_chat_known_nested_voice_selects_speaker(preprocess_chat):
    prompt = await preprocess_chat(_chat_request(" Ethan "))

    assert prompt["additional_information"]["speaker"] == ["ethan"]


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["alloy", "vivian"])
async def test_chat_nested_voice_without_speaker_list_uses_default(preprocess_chat, voice):
    prompt = await preprocess_chat(_chat_request(voice), supported=set())

    assert "speaker" not in prompt.get("additional_information", {})


@pytest.mark.asyncio
async def test_chat_missing_speaker_list_preserves_explicit_prompt_metadata(preprocess_chat):
    request = _chat_request("alloy", additional_information={"speaker": ["Ryan"], "language": ["English"]})

    prompt = await preprocess_chat(request, supported=set())

    assert prompt["additional_information"] == {"speaker": ["Ryan"], "language": ["English"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["voice", "speaker"])
async def test_chat_explicit_root_speaker_without_speaker_list_is_forwarded(preprocess_chat, field):
    prompt = await preprocess_chat(_chat_request("alloy", **{field: "Ryan"}), supported=set())

    assert prompt["additional_information"]["speaker"] == ["ryan"]


@pytest.mark.asyncio
async def test_joyai_chat_without_speaker_list_uses_valid_tts_default(mocker: MockerFixture, preprocess_chat):
    from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import (
        _build_tts_metadata,
        _validate_talker_speaker,
    )

    prompt = await preprocess_chat(_chat_request("alloy"), supported=set())
    metadata = _build_tts_metadata(prompt, "Hello")
    target_model = mocker.MagicMock()
    target_model.hf_config.talker_config.spk_id = {"vivian": 3065, "ryan": 3061}

    _validate_talker_speaker(metadata, target_model)
    assert metadata["speaker"] == ["Vivian"]


@pytest.mark.asyncio
async def test_chat_root_speaker_takes_precedence_over_nested_voice(preprocess_chat):
    prompt = await preprocess_chat(_chat_request("alloy", speaker="Chelsie"))

    assert prompt["additional_information"]["speaker"] == ["chelsie"]


@pytest.mark.asyncio
async def test_chat_root_voice_takes_precedence_over_speaker(preprocess_chat):
    prompt = await preprocess_chat(_chat_request("alloy", voice="Ethan", speaker="Chelsie"))

    assert prompt["additional_information"]["speaker"] == ["ethan"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["voice", "speaker"])
async def test_chat_unknown_root_speaker_is_still_rejected(preprocess_chat, field):
    with pytest.raises(ValueError, match="Invalid speaker 'alloy'"):
        await preprocess_chat(_chat_request("ethan", **{field: "alloy"}))


@pytest.mark.asyncio
async def test_fallback_unknown_session_voice_is_still_rejected(preprocess_chat):
    fallback_request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={"model": "qwen", "modalities": ["audio"], "voice": "alloy"},
        input_payload=None,
        policy_messages=(),
    )
    request = build_chat_request(fallback_request, model="qwen")

    with pytest.raises(ValueError, match="Invalid speaker 'alloy'"):
        await preprocess_chat(request)


@pytest.mark.asyncio
async def test_fallback_session_voice_without_speaker_list_is_forwarded(preprocess_chat):
    fallback_request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={"model": "qwen", "modalities": ["audio"], "voice": "Ryan"},
        input_payload=None,
        policy_messages=(),
    )

    prompt = await preprocess_chat(build_chat_request(fallback_request, model="qwen"), supported=set())

    assert prompt["additional_information"]["speaker"] == ["ryan"]


@pytest.mark.asyncio
async def test_fallback_existing_root_speaker_takes_precedence(preprocess_chat):
    fallback_request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={
            "model": "qwen",
            "modalities": ["audio"],
            "voice": "ethan",
            "extra_body": {"speaker": "Chelsie"},
        },
        input_payload=None,
        policy_messages=(),
    )

    prompt = await preprocess_chat(build_chat_request(fallback_request, model="qwen"))

    assert prompt["additional_information"]["speaker"] == ["chelsie"]


def test_create_chat_completion_converts_value_error_to_error_response(mocker: MockerFixture, serving_chat):
    serving_chat._diffusion_mode = False
    serving_chat._check_model = mocker.AsyncMock(return_value=None)
    serving_chat.engine_client = mocker.MagicMock(errored=False)
    serving_chat._maybe_get_adapters = mocker.MagicMock(return_value=None)
    serving_chat.models = mocker.MagicMock()
    serving_chat.models.model_name.return_value = "test-model"
    serving_chat.renderer = mocker.MagicMock()
    serving_chat.renderer.get_tokenizer.return_value = mocker.MagicMock()
    serving_chat.reasoning_parser_cls = None
    serving_chat.tool_parser = None
    serving_chat.parser_cls = None
    serving_chat.use_harmony = False
    serving_chat.enable_auto_tools = False
    serving_chat.exclude_tools_when_tool_choice_none = False
    serving_chat.trust_request_chat_template = False
    serving_chat.chat_template = None
    serving_chat.chat_template_content_format = "string"
    serving_chat.default_chat_template_kwargs = {}
    serving_chat.online_renderer = mocker.MagicMock()
    serving_chat.online_renderer.validate_chat_template.return_value = None
    serving_chat._effective_chat_template_kwargs = mocker.MagicMock(return_value={})
    serving_chat._preprocess_chat = mocker.AsyncMock(
        side_effect=ValueError("Invalid speaker 'uncle_fu'. Supported: ethan, vivian")
    )
    serving_chat.create_error_response = mocker.MagicMock(return_value="error-response")

    request = SimpleNamespace(
        tool_choice=None,
        tools=None,
        chat_template=None,
        chat_template_kwargs=None,
        reasoning_effort=None,
        messages=[],
        add_generation_prompt=False,
        continue_final_message=False,
        add_special_tokens=False,
        request_id="speaker-test",
    )

    result = asyncio.run(serving_chat.create_chat_completion(request))

    assert result == "error-response"
    serving_chat.create_error_response.assert_called_once_with("Invalid speaker 'uncle_fu'. Supported: ethan, vivian")
