# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for chat endpoint speaker validation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_omni.entrypoints.openai.utils import (
    get_supported_speakers_from_hf_config,
    validate_requested_speaker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance._supported_speakers = None
    return instance


def _make_hf_config(*, speaker_id: dict | None = None, spk_id: dict | None = None):
    hf_config = MagicMock()
    talker_config = MagicMock()
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


def test_get_supported_speakers_from_hf_config_uses_spk_id_fallback():
    hf_config = _make_hf_config(speaker_id=None, spk_id={"Serena": 0})
    assert get_supported_speakers_from_hf_config(hf_config) == {"serena"}


def test_get_supported_speakers_caches_normalized_keys(serving_chat):
    serving_chat.model_config = MagicMock()
    serving_chat.model_config.hf_config = _make_hf_config(speaker_id={"Vivian": 0, "Ethan": 1})

    assert serving_chat._get_supported_speakers() == {"vivian", "ethan"}

    # Cached value should be reused even if the config changes afterwards.
    serving_chat.model_config.hf_config.talker_config.speaker_id = {"Serena": 2}
    assert serving_chat._get_supported_speakers() == {"vivian", "ethan"}


def test_create_chat_completion_converts_value_error_to_error_response(serving_chat):
    serving_chat._diffusion_mode = False
    serving_chat._check_model = AsyncMock(return_value=None)
    serving_chat.engine_client = MagicMock(errored=False)
    serving_chat._maybe_get_adapters = MagicMock(return_value=None)
    serving_chat.models = MagicMock()
    serving_chat.models.model_name.return_value = "test-model"
    serving_chat.renderer = MagicMock()
    serving_chat.renderer.get_tokenizer.return_value = MagicMock()
    serving_chat.reasoning_parser_cls = None
    serving_chat.tool_parser = None
    serving_chat.use_harmony = False
    serving_chat.enable_auto_tools = False
    serving_chat.exclude_tools_when_tool_choice_none = False
    serving_chat.trust_request_chat_template = False
    serving_chat.chat_template = None
    serving_chat.chat_template_content_format = "string"
    serving_chat.default_chat_template_kwargs = {}
    serving_chat._validate_chat_template = MagicMock(return_value=None)
    serving_chat._prepare_extra_chat_template_kwargs = MagicMock(return_value={})
    serving_chat._preprocess_chat = AsyncMock(
        side_effect=ValueError("Invalid speaker 'uncle_fu'. Supported: ethan, vivian")
    )
    serving_chat.create_error_response = MagicMock(return_value="error-response")

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
