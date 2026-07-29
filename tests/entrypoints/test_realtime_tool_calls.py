# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the self-contained <tool_call> parser used by the
/v1/realtime tool-calling extension (see realtime_tool_calls.py)."""

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.realtime_tool_calls import ToolCallStreamState, extract_deltas

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestExtractDeltasPlainContent:
    def test_plain_text_no_tool_call_streams_as_content(self) -> None:
        state = ToolCallStreamState()
        content, deltas = extract_deltas("Hello", state)
        assert content == "Hello"
        assert deltas == []
        assert not state.has_tool_calls()

    def test_content_only_advances_incrementally(self) -> None:
        state = ToolCallStreamState()
        first, _ = extract_deltas("Hello", state)
        second, _ = extract_deltas("Hello, world", state)
        assert first == "Hello"
        assert second == ", world"

    def test_withholds_partial_start_tag_suffix(self) -> None:
        state = ToolCallStreamState()
        content, deltas = extract_deltas("Sure thing<tool_c", state)
        # "<tool_c" could still turn into "<tool_call>"; must not be sent as content yet.
        assert content == "Sure thing"
        assert deltas == []


class TestExtractDeltasToolCallLifecycle:
    def test_name_delta_emitted_once_available(self) -> None:
        state = ToolCallStreamState()
        text = '<tool_call>\n{"name": "get_weather", "arguments": '
        _, deltas = extract_deltas(text, state)
        assert len(deltas) == 1
        assert deltas[0].index == 0
        assert deltas[0].name == "get_weather"
        assert deltas[0].arguments_delta == ""
        assert state.has_tool_calls()

    def test_name_not_emitted_twice(self) -> None:
        state = ToolCallStreamState()
        extract_deltas('<tool_call>\n{"name": "get_weather", "arguments": {}', state)
        _, deltas = extract_deltas('<tool_call>\n{"name": "get_weather", "arguments": {}}', state)
        assert all(d.name is None for d in deltas)

    def test_arguments_stream_incrementally_as_suffix_diff(self) -> None:
        state = ToolCallStreamState()
        chunks = [
            '<tool_call>\n{"name": "get_weather", "arguments": {"',
            '<tool_call>\n{"name": "get_weather", "arguments": {"city',
            '<tool_call>\n{"name": "get_weather", "arguments": {"city":',
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Boston"}',
        ]
        collected_args = ""
        for chunk in chunks:
            _, deltas = extract_deltas(chunk, state)
            for d in deltas:
                collected_args += d.arguments_delta
        assert collected_args == '{"city": "Boston"}'

    def test_full_block_including_trailing_outer_brace_and_newline(self) -> None:
        """Regression: the outer {"name":..., "arguments":...} object's own
        closing brace and the newline before </tool_call> must never leak
        into the arguments value, even when they arrive in the same chunk
        as the arguments value's own closing brace."""
        state = ToolCallStreamState()
        full_text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Boston"}}\n</tool_call>'
        _, deltas = extract_deltas(full_text, state)
        args_deltas = [d.arguments_delta for d in deltas if d.arguments_delta]
        combined = "".join(args_deltas)
        assert combined == '{"city": "Boston"}'
        import json

        json.loads(combined)  # must be valid JSON, not '{"city": "Boston"}}\n'

    def test_nested_object_in_arguments_not_truncated_early(self) -> None:
        state = ToolCallStreamState()
        full_text = (
            '<tool_call>\n{"name": "book_flight", '
            '"arguments": {"from": "BOS", "to": "SFO", "options": {"nonstop": true}}}\n</tool_call>'
        )
        _, deltas = extract_deltas(full_text, state)
        combined = "".join(d.arguments_delta for d in deltas if d.arguments_delta)
        assert combined == '{"from": "BOS", "to": "SFO", "options": {"nonstop": true}}'

    def test_content_before_tool_call_is_still_streamed(self) -> None:
        state = ToolCallStreamState()
        text = 'One moment<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'
        content, _ = extract_deltas(text, state)
        assert content == "One moment"

    def test_multiple_tool_calls_tracked_by_independent_index(self) -> None:
        state = ToolCallStreamState()
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Boston"}}\n</tool_call>'
            '<tool_call>\n{"name": "get_time", "arguments": {"tz": "EST"}}\n</tool_call>'
        )
        _, deltas = extract_deltas(text, state)
        names = {d.index: d.name for d in deltas if d.name is not None}
        assert names == {0: "get_weather", 1: "get_time"}
        args_by_index: dict[int, str] = {0: "", 1: ""}
        for d in deltas:
            if d.arguments_delta:
                args_by_index[d.index] += d.arguments_delta
        assert args_by_index[0] == '{"city": "Boston"}'
        assert args_by_index[1] == '{"tz": "EST"}'
