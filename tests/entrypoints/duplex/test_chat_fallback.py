# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure request construction and output projection for duplex chat fallback."""

from __future__ import annotations

import base64
import io
import struct
import wave

import pytest

from vllm_omni.engine.duplex.fallback import FALLBACK_AUDIO_PAYLOAD_KEY, DuplexFallbackRequest
from vllm_omni.entrypoints.duplex.chat_fallback import (
    _fallback_output_sample_rate,
    build_chat_request,
    project_chat_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_chat_request_replaces_engine_audio_placeholder() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
            },
        ),
        response_config={"model": "qwen", "modalities": ["audio"], "response_format": "wav"},
        input_payload={
            "audio": base64.b64encode(struct.pack("<2f", 0.1, 0.2)).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": 16_000,
        },
        policy_messages=({"role": "system", "content": "policy"},),
    )
    chat_request = build_chat_request(request, model="qwen")

    assert chat_request.messages[0]["content"] == "policy"
    assert chat_request.messages[-1]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_build_chat_request_does_not_duplicate_current_history_audio() -> None:
    payload = {
        "audio": base64.b64encode(struct.pack("<2f", 0.1, 0.2)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
                FALLBACK_AUDIO_PAYLOAD_KEY: payload,
            },
        ),
        response_config={"model": "qwen", "modalities": ["audio"], "response_format": "wav"},
        input_payload=payload,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")
    audio_messages = [
        message
        for message in chat_request.messages
        if isinstance(message.get("content"), list)
        and message["content"]
        and message["content"][0].get("type") == "audio_url"
    ]

    assert len(audio_messages) == 1
    assert audio_messages[0]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_build_chat_request_includes_initial_user_text_once_in_user_order() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input"}}]},
        ),
        response_config={"model": "qwen", "initial_user_text": "start with this"},
        input_payload=None,
        policy_messages=({"role": "system", "content": "policy"},),
    )

    first = build_chat_request(request, model="qwen")
    second = build_chat_request(request, model="qwen")
    first_user_messages = [message for message in first.messages if message["role"] == "user"]
    second_user_messages = [message for message in second.messages if message["role"] == "user"]

    assert [message["content"] for message in first_user_messages] == [
        "start with this",
        [{"type": "audio_url", "audio_url": {"url": "native-duplex:input"}}],
    ]
    assert [message["role"] for message in first.messages] == ["system", "user", "assistant", "user"]
    assert first_user_messages == second_user_messages


def test_build_chat_request_preserves_audio_voice_and_speed() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={
            "model": "qwen",
            "modalities": ["audio"],
            "response_format": "pcm16",
            "voice": "Vivian",
            "speed": 1.25,
        },
        input_payload=None,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")

    assert chat_request.audio == {"format": "pcm", "voice": "Vivian", "speed": 1.25}


def test_build_chat_request_normalizes_frozen_tools_and_tool_choice() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={
            "model": "qwen",
            "extra_body": {
                "realtime_response_tools": [
                    {"type": "function", "function": {"name": "lookup"}},
                ],
                "realtime_response_tool_choice": {
                    "type": "function",
                    "function": {"name": "lookup"},
                },
            },
        },
        input_payload=None,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")

    serialized = chat_request.model_dump(exclude_none=True)
    assert serialized["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
    assert serialized["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_build_chat_request_keeps_each_history_audio_payload_distinct() -> None:
    first_payload = {
        "audio": base64.b64encode(struct.pack("<2f", 0.1, 0.1)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    second_payload = {
        "audio": base64.b64encode(struct.pack("<2f", 0.2, 0.2)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    placeholder = {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-2",
        response_id="resp-2",
        epoch=1,
        history=(
            {"role": "user", "content": [placeholder], "_duplex_fallback_audio": first_payload},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": [placeholder], "_duplex_fallback_audio": second_payload},
            {"role": "user", "content": "text follow-up"},
        ),
        response_config={"model": "qwen"},
        input_payload=None,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")
    audio_urls = [
        message["content"][0]["audio_url"]["url"]
        for message in chat_request.messages
        if isinstance(message.get("content"), list)
        and message["content"]
        and message["content"][0].get("type") == "audio_url"
    ]

    assert len(audio_urls) == 2
    assert audio_urls[0].startswith("data:audio/wav;base64,")
    assert audio_urls[0] != audio_urls[1]
    assert all("native-duplex:" not in url for url in audio_urls)
    assert all("_duplex_fallback_audio" not in message for message in chat_request.messages)
    assert chat_request.messages[-1]["content"] == "text follow-up"


def test_audio_history_byte_budget_prunes_complete_turns_without_mutating_the_snapshot() -> None:
    payload = {
        "audio": base64.b64encode(b"\0" * (4 * 1024 * 1024)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    audio_message = {
        "role": "user",
        "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
        FALLBACK_AUDIO_PAYLOAD_KEY: payload,
    }
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="request",
        response_id="resp",
        epoch=0,
        history=(
            {"role": "user", "content": "old question"},
            audio_message,
            {"role": "assistant", "content": "old answer"},
            audio_message,
            {"role": "user", "content": "new question"},
        ),
        response_config={"model": "qwen"},
        input_payload=payload,
        policy_messages=(),
    )

    chat = build_chat_request(request, model="qwen")

    assert [message["role"] for message in chat.messages] == ["user", "user"]
    assert chat.messages[-1]["content"] == "new question"
    assert len(request.history) == 5
    assert FALLBACK_AUDIO_PAYLOAD_KEY in request.history[1]


def test_standalone_current_audio_reserves_a_slot_without_overwriting_historical_payloads() -> None:
    def payload(value: float) -> dict[str, object]:
        return {
            "audio": base64.b64encode(struct.pack("<2f", value, value)).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": 16_000,
        }

    history: list[dict[str, object]] = []
    for index in range(4):
        history.extend(
            [
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
                    FALLBACK_AUDIO_PAYLOAD_KEY: payload((index + 1) / 8),
                },
                {"role": "assistant", "content": f"answer-{index}"},
            ]
        )
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="request",
        response_id="resp",
        epoch=0,
        history=history,
        response_config={"model": "qwen"},
        input_payload=payload(0.625),
        policy_messages=(),
    )

    chat = build_chat_request(request, model="qwen")
    audios = []
    for message in chat.messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else ():
            if part.get("type") == "audio_url":
                raw = base64.b64decode(part["audio_url"]["url"].split(",", 1)[1])
                with wave.open(io.BytesIO(raw), "rb") as wav:
                    audios.append(struct.unpack("<2h", wav.readframes(wav.getnframes())))

    assert audios == [(8192, 8192), (12288, 12288), (16384, 16384), (20480, 20480)]
    assert [message["content"] for message in chat.messages if message["role"] == "assistant"] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert len(request.history) == 8


def test_fallback_output_sample_rate_reads_realtime_session_payload() -> None:
    assert (
        _fallback_output_sample_rate(
            {
                "extra_body": {
                    "realtime_session_payload": {
                        "audio": {"output": {"format": "pcm16", "sample_rate": 24_000}},
                    },
                },
            }
        )
        == 24_000
    )


def test_project_chat_payload_returns_model_neutral_text_output() -> None:
    outputs = project_chat_payload(
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        request_id="fallback-sid-0-1",
    )

    assert outputs == [{"text": "hello", "data_plane_request_id": "fallback-sid-0-1"}]


def test_model_pcm_sample_rate_overrides_the_session_hint() -> None:
    outputs = project_chat_payload(
        {
            "modality": "audio",
            "sample_rate_hz": 16_000,
            "choices": [{"delta": {"content": base64.b64encode(b"\0\0" * 8_000).decode()}}],
        },
        request_id="fallback-sid-0-1",
        response_format="pcm16",
        sample_rate_hz=24_000,
    )

    assert outputs[0]["sample_rate_hz"] == 16_000
    assert outputs[0]["audio_duration_ms"] == 500


def test_project_chat_payload_preserves_streamed_tool_call_fragments() -> None:
    first_outputs = project_chat_payload(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"city":"'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        request_id="fallback-sid-0-1",
    )
    continuation_outputs = project_chat_payload(
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'Paris"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        request_id="fallback-sid-0-1",
    )

    assert first_outputs == [
        {
            "function_call": True,
            "_duplex_tool_call_fragment": True,
            "_duplex_tool_call_index": 0,
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"city":"',
            "data_plane_request_id": "fallback-sid-0-1",
        }
    ]
    assert continuation_outputs == [
        {
            "function_call": True,
            "_duplex_tool_call_fragment": True,
            "_duplex_tool_call_index": 0,
            "arguments": 'Paris"}',
            "data_plane_request_id": "fallback-sid-0-1",
        }
    ]


def test_project_chat_payload_returns_provider_error() -> None:
    outputs = project_chat_payload(
        {"error": {"message": "rejected", "type": "BadRequestError"}},
        request_id="fallback-sid-0-1",
    )

    assert outputs == [{"error": "rejected", "error_code": "BadRequestError"}]


def test_project_chat_payload_marks_stream_completion_as_end_of_turn() -> None:
    outputs = project_chat_payload("[DONE]", request_id="fallback-sid-0-1")

    assert outputs == [{"data_plane_request_id": "fallback-sid-0-1", "end_of_turn": True}]
