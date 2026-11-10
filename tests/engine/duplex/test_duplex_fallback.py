# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contracts for the engine/API chat-fallback bridge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.fallback import DuplexFallbackRequest
from vllm_omni.engine.duplex.messages import DuplexSessionFallbackOutputMessage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fallback_request_snapshots_nested_identity_and_history() -> None:
    content = [{"type": "text", "text": "hello"}]
    history = [{"role": "user", "content": content}]
    modalities = ["audio"]
    response_config = {"model": "qwen", "modalities": modalities}
    input_payload = {"format": "pcm_f32le", "audio": "AAAA", "sample_rate_hz": 16_000}
    policy_messages = [{"role": "system", "content": "policy"}]

    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp-sid-0-abcd",
        epoch=3,
        history=history,
        response_config=response_config,
        input_payload=input_payload,
        policy_messages=policy_messages,
    )

    history[0]["role"] = "assistant"
    content[0]["text"] = "changed"
    modalities.append("text")
    input_payload["audio"] = "changed"
    policy_messages[0]["content"] = "changed"

    assert request.session_id == "sid"
    assert request.epoch == 3
    assert request.history[0]["role"] == "user"
    assert request.history[0]["content"][0]["text"] == "hello"
    assert request.response_config["modalities"] == ("audio",)
    assert request.input_payload["audio"] == "AAAA"
    assert request.policy_messages[0]["content"] == "policy"
    with pytest.raises(FrozenInstanceError):
        request.epoch = 4  # type: ignore[misc]


def test_chat_fallback_capability_is_false_by_default() -> None:
    capabilities = DuplexCapabilities()

    assert capabilities.supports_chat_fallback is False
    assert capabilities.as_dict()["implementation_level"] == "turn_based_duplex"
    assert capabilities.as_dict()["input_modes"] == ["append_audio_chunk"]


def test_capabilities_as_dict_exposes_chat_fallback_fields() -> None:
    capabilities = DuplexCapabilities(
        supports_chat_completions=True,
        supports_chat_fallback=True,
        text_turn_priming_units=0,
        implementation_level="chat_fallback",
        input_modes=["turn_commit_only"],
    )

    serialized = capabilities.as_dict()

    assert serialized["supports_chat_completions"] is True
    assert serialized["supports_chat_fallback"] is True
    assert serialized["text_turn_priming_units"] == 0
    assert serialized["implementation_level"] == "chat_fallback"


@pytest.mark.parametrize("supports_text_only_turn", [False, True])
def test_public_capabilities_preserve_text_only_turn_support(supports_text_only_turn: bool) -> None:
    capabilities = DuplexCapabilities(supports_text_only_turn=supports_text_only_turn)

    assert capabilities.as_dict()["supports_text_only_turn"] is supports_text_only_turn


def test_fallback_messages_are_engine_messages_not_realtime_events() -> None:
    message = DuplexSessionFallbackOutputMessage(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        output={"text": "hello", "end_of_turn": True},
    )

    assert message.type == "duplex_session_fallback_output"
    assert message.session_id == "sid"
    assert message.output == {"text": "hello", "end_of_turn": True}
