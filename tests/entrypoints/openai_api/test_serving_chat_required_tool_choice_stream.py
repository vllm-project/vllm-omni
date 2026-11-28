# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for streaming ``tool_choice="required"`` chat completions.

vLLM changed ``extract_required_tool_call_streaming`` twice (v0.32: dropped
``delta_text`` / ``function_name_returned``; v0.34: per-stream
``RequiredToolCallScanner``) while the streaming generator kept the v0.31
call, so every such request ended in a ``TypeError`` error event (issue #10666).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_text_omni_output,
    parse_sse_chunks,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }


def _request(n: int = 1) -> ChatCompletionRequest:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "weather and time in Paris?"}],
        tools=[_tool("get_weather"), _tool("get_time")],
        tool_choice="required",
        stream=True,
        n=n,
    )
    request.modalities = ["text"]  # type: ignore[attr-defined]
    return request


async def _stream(outputs, *, n: int = 1, reasoning_parser=None) -> list[dict]:
    async def result_generator():
        for output in outputs:
            yield output

    raw_lines = await collect_stream(
        build_serving_chat().chat_completion_stream_generator(
            request=_request(n),
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
            reasoning_parser=reasoning_parser,
        )
    )
    chunks = parse_sse_chunks(raw_lines)
    assert [chunk["error"] for chunk in chunks if "error" in chunk] == []
    return chunks


def _text_outputs(deltas: list[str], index: int = 0) -> list:
    last = len(deltas) - 1
    return [
        make_text_omni_output(
            text=delta,
            token_ids=[100 + position],
            finish_reason="stop" if position == last else None,
            index=index,
        )
        for position, delta in enumerate(deltas)
    ]


def _choices(chunks: list[dict], index: int = 0) -> list[dict]:
    return [choice for chunk in chunks for choice in chunk.get("choices", []) if choice["index"] == index]


def _assemble_tool_calls(chunks: list[dict], index: int = 0) -> list[dict]:
    """Rebuild each tool call the way an OpenAI client does: by ``index``."""
    calls: dict[int, dict] = {}
    for choice in _choices(chunks, index):
        for delta in choice["delta"].get("tool_calls") or []:
            call = calls.setdefault(delta["index"], {"ids": [], "names": [], "arguments": ""})
            function = delta.get("function") or {}
            if delta.get("id"):
                call["ids"].append(delta["id"])
            if function.get("name"):
                call["names"].append(function["name"])
            call["arguments"] += function.get("arguments") or ""
    return [calls[position] for position in sorted(calls)]


def _finish_reasons(chunks: list[dict], index: int = 0) -> list[str]:
    return [choice["finish_reason"] for choice in _choices(chunks, index) if choice.get("finish_reason")]


@pytest.mark.asyncio
async def test_required_tool_choice_streams_call_split_across_deltas():
    chunks = await _stream(_text_outputs(['[{"name": "get_', 'weather", "parameters": {"city": "Par', 'is"}}]']))

    (call,) = _assemble_tool_calls(chunks)
    assert call["names"] == ["get_weather"]
    assert len(call["ids"]) == 1
    assert call["arguments"] == '{"city": "Paris"}'
    assert _finish_reasons(chunks) == ["tool_calls"]


@pytest.mark.asyncio
async def test_required_tool_choice_counts_every_call_started_by_one_delta(monkeypatch):
    # kimi_k2 ids embed the running tool-call count, so they expose whether a
    # delta that starts two calls advances the count by two.
    monkeypatch.setattr(OmniOpenAIServingChat, "tool_call_id_type", "kimi_k2")
    chunks = await _stream(
        _text_outputs(
            [
                '[{"name": "get_weather", "parameters": {"city": "Paris"}}, {"name": "get_time", "parameters": {"ci',
                'ty": "Paris"}}, {"name": "get_weather", "parameters": {"city": "Rome"}}]',
            ]
        )
    )

    calls = _assemble_tool_calls(chunks)
    assert [call["ids"] for call in calls] == [
        ["functions.get_weather:0"],
        ["functions.get_time:1"],
        ["functions.get_weather:2"],
    ]
    assert [call["arguments"] for call in calls] == [
        '{"city": "Paris"}',
        '{"city": "Paris"}',
        '{"city": "Rome"}',
    ]
    assert _finish_reasons(chunks) == ["tool_calls"]


@pytest.mark.asyncio
async def test_required_tool_choice_keeps_scanner_state_per_choice():
    first = _text_outputs(['[{"name": "get_weather", "parameters": {"city": "Par', 'is"}}]'], index=0)
    second = _text_outputs(['[{"name": "get_time", "parameters": {"city": "Ro', 'me"}}]'], index=1)
    chunks = await _stream([first[0], second[0], first[1], second[1]], n=2)

    (weather,) = _assemble_tool_calls(chunks, index=0)
    (time_,) = _assemble_tool_calls(chunks, index=1)
    assert (weather["names"], weather["arguments"]) == (["get_weather"], '{"city": "Paris"}')
    assert (time_["names"], time_["arguments"]) == (["get_time"], '{"city": "Rome"}')
    assert _finish_reasons(chunks, index=0) == ["tool_calls"]
    assert _finish_reasons(chunks, index=1) == ["tool_calls"]


class _ThinkReasoningParser:
    """Minimal ``<think>...</think>`` parser; token id 0 closes reasoning."""

    END = "</think>"

    def is_reasoning_end(self, input_ids) -> bool:
        return 0 in input_ids

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text, *_token_ids) -> DeltaMessage:
        if self.END not in delta_text:
            return DeltaMessage(reasoning=delta_text)
        reasoning, content = delta_text.split(self.END, 1)
        return DeltaMessage(reasoning=reasoning or None, content=content or None)


@pytest.mark.asyncio
async def test_required_tool_choice_after_reasoning_keeps_text_from_the_closing_delta():
    # The delta that closes reasoning also carries the start of the tool-call
    # array. It is not parsed in that step, so the next step must hand the
    # scanner both the carried-over text and its own delta.
    outputs = [
        make_text_omni_output(text="need the weather", token_ids=[7], num_prompt_tokens=0),
        make_text_omni_output(text='</think>[{"name": "get_', token_ids=[0], num_prompt_tokens=0),
        make_text_omni_output(text='weather", "parameters": {"city": "Par', token_ids=[8], num_prompt_tokens=0),
        make_text_omni_output(text='is"}}]', token_ids=[9], finish_reason="stop", num_prompt_tokens=0),
    ]
    chunks = await _stream(outputs, reasoning_parser=_ThinkReasoningParser())

    (call,) = _assemble_tool_calls(chunks)
    assert call["names"] == ["get_weather"]
    assert call["arguments"] == '{"city": "Paris"}'
    reasoning = "".join(choice["delta"].get("reasoning") or "" for choice in _choices(chunks))
    assert reasoning == "need the weather"
    assert _finish_reasons(chunks) == ["tool_calls"]
