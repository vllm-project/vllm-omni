# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest
import torch
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.openai.realtime_protocol import OmniSessionUpdate
from vllm_omni.entrypoints.openai.realtime_tool_format import render_tool_preamble
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    build_realtime_prompt_token_ids,
    drain_realtime_prompt_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


class TestRealtimeConnectionTensorAndPcm:
    def test_tensor_to_numpy_none(self) -> None:
        assert RealtimeConnection._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_1d_numpy(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (2,)

    def test_tensor_to_numpy_2d_numpy_flattened(self) -> None:
        arr = np.array([[0.5], [-0.5]], dtype=np.float32)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.shape == (2,)

    def test_tensor_to_numpy_torch(self) -> None:
        t = torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        out = RealtimeConnection._tensor_to_numpy(t)
        assert out is not None
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.25, -0.25], rtol=1e-5)

    def test_pcm16_b64_roundtrip(self) -> None:
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        b64 = RealtimeConnection._pcm16_b64(audio)
        raw = base64.b64decode(b64)
        assert len(raw) == 6
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert pcm[0] == 0
        assert pcm[1] == 32767
        assert pcm[2] == -32767


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class _FakeTokenizer:
    """Encodes to one id per character, which is enough to assert that a rendered
    turn reaches the prompt and in which order."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(ch) % 1000 for ch in text]


@pytest.fixture
def tool_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn.connection_id = "ws-test"
    conn._tools = None
    conn._tool_choice = "auto"
    conn._pending_context_tokens = []
    conn._pending_tool_calls = {}
    conn._tokenizer = _FakeTokenizer()
    conn._tool_parser = None
    return conn


class TestRealtimeToolConfiguration:
    def test_no_tools_by_default(self, tool_conn: RealtimeConnection) -> None:
        assert tool_conn.tools_enabled is False

    def test_session_update_with_tools_enables_and_stages_preamble(self, tool_conn: RealtimeConnection) -> None:
        tool_conn._configure_tools({"type": "session.update", "model": "m", "tools": [_WEATHER_TOOL]})
        assert tool_conn.tools_enabled is True
        # The tool definitions must be staged for the next prompt, otherwise the
        # model never learns the tools exist.
        assert tool_conn._pending_context_tokens

    def test_preamble_declares_tool_in_qwen_format(self, tool_conn: RealtimeConnection) -> None:
        update = OmniSessionUpdate(type="session.update", model="m", tools=[_WEATHER_TOOL])
        preamble = render_tool_preamble(update.tools)
        assert "<|im_start|>system" in preamble
        assert "<tools>" in preamble and "</tools>" in preamble
        assert "get_weather" in preamble
        # The parser recognizes calls by this tag, so the instruction must name it.
        assert "<tool_call>" in preamble

    def test_tool_choice_none_disables_tools(self, tool_conn: RealtimeConnection) -> None:
        tool_conn._configure_tools(
            {"type": "session.update", "model": "m", "tools": [_WEATHER_TOOL], "tool_choice": "none"}
        )
        assert tool_conn.tools_enabled is False
        assert tool_conn._pending_context_tokens == []

    def test_session_update_without_tools_keeps_tools_disabled(self, tool_conn: RealtimeConnection) -> None:
        tool_conn._configure_tools({"type": "session.update", "model": "m"})
        assert tool_conn.tools_enabled is False
        assert tool_conn._pending_context_tokens == []


class TestRealtimeToolResults:
    @pytest.mark.asyncio
    async def test_tool_result_is_staged_as_context(self, tool_conn: RealtimeConnection) -> None:
        tool_conn._pending_tool_calls = {"call_1": "get_weather"}
        await tool_conn._handle_conversation_item_create(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"temp_c": 21}'},
            }
        )
        assert tool_conn._pending_context_tokens
        # The call is settled, so a duplicate result would be flagged as unknown.
        assert "call_1" not in tool_conn._pending_tool_calls

    @pytest.mark.asyncio
    async def test_unknown_call_id_is_tolerated(self, tool_conn: RealtimeConnection) -> None:
        # A late or duplicated result must not break the session.
        await tool_conn._handle_conversation_item_create(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "nope", "output": "{}"},
            }
        )
        assert tool_conn._pending_context_tokens


class TestRealtimeToolCallEmission:
    @pytest.mark.asyncio
    async def test_emits_delta_and_done_and_tracks_call_id(self, tool_conn: RealtimeConnection) -> None:
        sent: list[dict] = []

        async def fake_send_json(payload: dict) -> None:
            sent.append(payload)

        tool_conn.send_json = fake_send_json  # type: ignore[method-assign]
        call = ToolCall(id="call_1", function=FunctionCall(name="get_weather", arguments='{"city": "Berlin"}'))

        await tool_conn._emit_tool_calls([call])

        types = [event["type"] for event in sent]
        assert types == [
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ]
        assert sent[-1]["call_id"] == "call_1"
        assert sent[-1]["name"] == "get_weather"
        assert sent[-1]["arguments"] == '{"city": "Berlin"}'
        # Needed so the tool result the client sends back can be rendered.
        assert tool_conn._pending_tool_calls["call_1"] == "get_weather"
        # The model's own call is kept as context for the follow-up turn.
        assert tool_conn._pending_context_tokens

    def test_extract_returns_nothing_when_tools_disabled(self, tool_conn: RealtimeConnection) -> None:
        text = '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'
        tool_calls, spoken = tool_conn._extract_tool_calls(text)
        assert tool_calls == []
        assert spoken == text


class TestRealtimeToolPromptFormat:
    """Pins the rendered turns to what the model's own chat template emits.

    These strings were taken from Qwen3-Omni's ``chat_template.json``; if they
    drift the model stops recognizing the tool protocol, and it fails silently
    (the model just talks instead of calling), so assert them exactly.
    """

    def test_preamble_matches_chat_template_output(self, tool_conn: RealtimeConnection) -> None:
        update = OmniSessionUpdate(type="session.update", model="m", tools=[_WEATHER_TOOL])
        rendered = render_tool_preamble(update.tools)
        assert rendered.startswith("<|im_start|>system\n\n\n# Tools\n\n")
        assert "You may call one or more functions to assist with the user query." in rendered
        assert "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" in rendered
        assert (
            "\n</tools>\n\nFor each function call, return a json object with function name "
            'and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
            '<function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>' in rendered
        )

    def test_tool_result_matches_chat_template_output(self, tool_conn: RealtimeConnection) -> None:
        from vllm_omni.entrypoints.openai.realtime_tool_format import TOOL_RESULT_TEMPLATE

        assert TOOL_RESULT_TEMPLATE.format(output='{"temp_c": 21}') == (
            '<|im_start|>user\n<tool_response>\n{"temp_c": 21}\n</tool_response><|im_end|>\n'
        )

    def test_assistant_tool_call_embeds_arguments_as_json_object(self, tool_conn: RealtimeConnection) -> None:
        from vllm_omni.entrypoints.openai.realtime_tool_format import ASSISTANT_TOOL_CALL_TEMPLATE

        rendered = ASSISTANT_TOOL_CALL_TEMPLATE.format(name="get_weather", arguments='{"city": "Berlin"}')
        # The chat template emits a string-valued arguments field verbatim, so the
        # arguments must appear as a JSON object and not as an escaped string.
        assert rendered == (
            '<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Berlin"}}\n</tool_call><|im_end|>\n'
        )
        assert "\\" not in rendered


class TestRealtimePromptContext:
    """``buffer_realtime_audio`` prefixes each segment prompt with whatever the
    connection staged on the context channel."""

    def test_context_is_prepended(self) -> None:
        assert build_realtime_prompt_token_ids([1, 2, 3], [10, 11]) == [1, 2, 3, 10, 11]

    def test_empty_context_returns_the_prompt_itself(self) -> None:
        # A session that configures no tools must see the exact prompt it saw
        # before, so return the same object rather than a copy.
        prompt_token_ids = [10, 11]
        assert build_realtime_prompt_token_ids([], prompt_token_ids) is prompt_token_ids

    def test_drain_collects_everything_queued(self) -> None:
        queue: asyncio.Queue[list[int]] = asyncio.Queue()
        queue.put_nowait([1, 2])
        queue.put_nowait([3])
        collected: list[int] = []
        drain_realtime_prompt_context(queue, collected)
        assert collected == [1, 2, 3]
        assert queue.empty()

    def test_drain_accumulates_across_calls(self) -> None:
        # Later turns (a tool result, then the next one) append to the context
        # already in place instead of replacing it.
        queue: asyncio.Queue[list[int]] = asyncio.Queue()
        collected: list[int] = []
        queue.put_nowait([1])
        drain_realtime_prompt_context(queue, collected)
        queue.put_nowait([2])
        drain_realtime_prompt_context(queue, collected)
        assert collected == [1, 2]

    def test_drain_on_empty_queue_is_a_noop(self) -> None:
        queue: asyncio.Queue[list[int]] = asyncio.Queue()
        collected: list[int] = []
        drain_realtime_prompt_context(queue, collected)
        assert collected == []
