# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration as Qwen3OmniModel,
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


class TestQwen3OmniRealtimeChatFormatting:
    """Realtime chat-template helpers on the model class.

    These keep ``RealtimeConnection`` model-agnostic — the connection
    routes all chat-template / token surface through ``self.model_cls``.
    """

    def test_constants_present(self) -> None:
        # The connection reads these directly when detecting tool calls /
        # building prompts, so they must exist as class attributes.
        assert "<|audio_pad|>" in Qwen3OmniModel.AUDIO_PLACEHOLDER
        assert Qwen3OmniModel.TOOL_CALL_OPEN == "<tool_call>"
        assert Qwen3OmniModel.TOOL_CALL_CLOSE == "</tool_call>"
        assert Qwen3OmniModel.ASSISTANT_TURN_END == "<|im_end|>"

    def test_strip_special_tokens_removes_qwen_markers(self) -> None:
        # Strips any <|...|> token (im_start/end, audio_*, endoftext, ...)
        # rather than maintaining a fixed allowlist.
        text = "<|im_start|>hi<|audio_pad|> there<|endoftext|>"
        assert Qwen3OmniModel.strip_special_tokens(text) == "hi there"

    def test_strip_special_tokens_strips_whitespace(self) -> None:
        assert Qwen3OmniModel.strip_special_tokens("  hello  ") == "hello"

    def test_strip_special_tokens_empty(self) -> None:
        assert Qwen3OmniModel.strip_special_tokens("") == ""

    def test_format_tools_schema_empty_returns_empty(self) -> None:
        assert Qwen3OmniModel.format_tools_schema([]) == ""

    def test_format_tools_schema_includes_qwen_template_markers(self) -> None:
        tools = [{"type": "function", "function": {"name": "ping", "parameters": {}}}]
        schema = Qwen3OmniModel.format_tools_schema(tools)
        # Contains the Qwen3 template skeleton.
        assert "# Tools" in schema
        assert "<tools>" in schema and "</tools>" in schema
        assert Qwen3OmniModel.TOOL_CALL_OPEN in schema
        assert Qwen3OmniModel.TOOL_CALL_CLOSE in schema
        # Tool JSON is embedded.
        assert '"name": "ping"' in schema

    def test_parse_tool_call_valid(self) -> None:
        block = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        parsed = Qwen3OmniModel.parse_tool_call(block)
        assert parsed == {"name": "get_weather", "arguments": {"city": "Paris"}}

    def test_parse_tool_call_default_arguments(self) -> None:
        # Missing "arguments" defaults to {} so callers can json.dumps without
        # a None check.
        block = '<tool_call>{"name": "noop"}</tool_call>'
        assert Qwen3OmniModel.parse_tool_call(block) == {"name": "noop", "arguments": {}}

    def test_parse_tool_call_missing_name_returns_none(self) -> None:
        # No name → not a usable tool call; return None instead of partial.
        block = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
        assert Qwen3OmniModel.parse_tool_call(block) is None

    def test_parse_tool_call_malformed_json_returns_none(self) -> None:
        block = "<tool_call>not-json</tool_call>"
        assert Qwen3OmniModel.parse_tool_call(block) is None

    def test_render_history_empty(self) -> None:
        assert Qwen3OmniModel.render_history([]) == ""

    def test_render_history_skips_user_with_none_content(self) -> None:
        # The connection reserves a None-content user slot for the in-flight
        # turn while STT is still running; render_history must skip it so
        # the prompt isn't rendered with a placeholder.
        items = [{"role": "user", "content": None}]
        assert Qwen3OmniModel.render_history(items) == ""

    def test_render_history_user_block(self) -> None:
        rendered = Qwen3OmniModel.render_history([{"role": "user", "content": "hello"}])
        assert rendered == "<|im_start|>user\nhello<|im_end|>"

    def test_render_history_tool_response_uses_user_role_wrapper(self) -> None:
        # Qwen3's chat template renders tool results as user-role blocks
        # wrapped in <tool_response>...</tool_response>.
        rendered = Qwen3OmniModel.render_history([{"role": "tool", "call_id": "c1", "content": '{"temp": 18}'}])
        assert rendered.startswith("<|im_start|>user\n<tool_response>\n")
        assert '{"temp": 18}' in rendered
        assert rendered.endswith("</tool_response><|im_end|>")

    def test_render_history_assistant_with_tool_call(self) -> None:
        items = [
            {
                "role": "assistant",
                "content": "Looking that up",
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
            }
        ]
        rendered = Qwen3OmniModel.render_history(items)
        assert rendered.startswith("<|im_start|>assistant\n")
        assert "Looking that up" in rendered
        assert "<tool_call>" in rendered and "</tool_call>" in rendered
        assert '"name": "get_weather"' in rendered

    def test_render_history_assistant_strips_special_tokens_from_content(self) -> None:
        # Stored content can contain stray <|im_end|> from greedy decoding —
        # render_history must clean it before re-inserting into the prompt
        # (otherwise the next turn sees a turn boundary mid-content).
        items = [{"role": "assistant", "content": "spoken<|im_end|>", "tool_calls": []}]
        rendered = Qwen3OmniModel.render_history(items)
        assert rendered.count("<|im_end|>") == 1  # only the closing one
        assert "spoken" in rendered

    def test_render_history_assistant_empty_content_no_tool_calls_skipped(self) -> None:
        # Empty assistant entry contributes nothing — it would otherwise
        # render as "<|im_start|>assistant\n<|im_end|>", confusing the model.
        items = [{"role": "assistant", "content": "", "tool_calls": []}]
        assert Qwen3OmniModel.render_history(items) == ""

    def test_render_history_assistant_tool_call_args_already_stringified(self) -> None:
        # Some clients pass arguments as a pre-serialized JSON string;
        # render_history should not double-encode them.
        items = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "f", "arguments": '{"x": 1}'}],
            }
        ]
        rendered = Qwen3OmniModel.render_history(items)
        # No double-quoting / nested string of the JSON.
        assert '"arguments": {"x": 1}' in rendered

    def test_build_audio_pass_prompt_audio_first_assistant_open(self) -> None:
        prompt = Qwen3OmniModel.build_audio_pass_prompt(None, "")
        # Audio user block is FIRST so the talker's hidden_projection sees
        # purely acoustic states for bootstrap.
        assert prompt.startswith("<|im_start|>user\n")
        assert Qwen3OmniModel.AUDIO_PLACEHOLDER in prompt.split("<|im_end|>")[0]
        # Ends with an open assistant turn.
        assert prompt.rstrip().endswith("<|im_start|>assistant")

    def test_build_audio_pass_prompt_includes_system_and_history(self) -> None:
        history = "<|im_start|>user\nprior question<|im_end|>"
        prompt = Qwen3OmniModel.build_audio_pass_prompt("be helpful", history)
        # System block is rendered after the audio block.
        audio_idx = prompt.index(Qwen3OmniModel.AUDIO_PLACEHOLDER)
        sys_idx = prompt.index("<|im_start|>system\nbe helpful<|im_end|>")
        hist_idx = prompt.index(history)
        assistant_idx = prompt.rindex("<|im_start|>assistant")
        assert audio_idx < sys_idx < hist_idx < assistant_idx

    def test_build_transcription_prompt_contains_placeholder_and_role(self) -> None:
        prompt = Qwen3OmniModel.build_transcription_prompt()
        assert Qwen3OmniModel.AUDIO_PLACEHOLDER in prompt
        # Ends with an open assistant turn, ready for the engine.
        assert prompt.rstrip().endswith("<|im_start|>assistant")
        # Has a system instruction nudging text-only verbatim output.
        assert "<|im_start|>system" in prompt
        assert "transcription" in prompt.lower()

    def test_render_then_parse_round_trip(self) -> None:
        # Rendering a tool call into history and re-parsing it should
        # round-trip through parse_tool_call cleanly. This guards against
        # divergence between the writer (render_history) and the reader
        # (parse_tool_call) — they share the wire format.
        original = {"name": "search", "arguments": {"q": "vllm omni", "k": 3}}
        rendered = Qwen3OmniModel.render_history([{"role": "assistant", "content": "", "tool_calls": [original]}])
        # Extract just the <tool_call>...</tool_call> block.
        start = rendered.index(Qwen3OmniModel.TOOL_CALL_OPEN)
        end = rendered.index(Qwen3OmniModel.TOOL_CALL_CLOSE) + len(Qwen3OmniModel.TOOL_CALL_CLOSE)
        parsed = Qwen3OmniModel.parse_tool_call(rendered[start:end])
        assert parsed is not None
        assert parsed["name"] == original["name"]
        # Arguments compared as JSON to ignore key ordering.
        assert json.loads(json.dumps(parsed["arguments"])) == original["arguments"]
