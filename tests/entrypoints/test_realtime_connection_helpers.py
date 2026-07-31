# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest
import torch
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection, _PendingToolCall

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


@pytest.fixture
def tool_call_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._tools = None
    conn._tool_result_queue = asyncio.Queue()
    conn._pending_tool_calls = {}
    conn._tool_rounds = 0
    return conn


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


class TestRealtimeConnectionToolCallEventRouting:
    """handle_event's tool-calling additions (session.update.tools capture,
    conversation.item.create routing) - see realtime_tool_calls.py for the
    <tool_call> text parser these events feed."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_tools_and_delegates_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        event = {"type": "session.update", "model": "qwen3-omni", "tools": tools}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == tools
        base_handle_event.assert_awaited_once_with(event)

    def test_session_update_without_tools_leaves_existing_tools_untouched(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
        tool_call_conn._tools = [{"type": "function", "function": {"name": "existing"}}]

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert tool_call_conn._tools == [{"type": "function", "function": {"name": "existing"}}]

    def test_conversation_item_create_function_call_output_is_queued(self, tool_call_conn) -> None:
        item = {"type": "function_call_output", "call_id": "call_1", "output": "sunny and 72"}

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": item}))

        assert tool_call_conn._tool_result_queue.qsize() == 1
        assert tool_call_conn._tool_result_queue.get_nowait() == item

    def test_conversation_item_create_unsupported_item_type_sends_error(self, tool_call_conn, mocker) -> None:
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": {"type": "not_a_thing"}}))

        send_error.assert_awaited_once()
        assert tool_call_conn._tool_result_queue.empty()

    def test_unrelated_event_types_still_delegate_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(tool_call_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


class TestRenderTokenPromptReattachesAudio:
    """Regression test for a real bug: the tool-call continuation re-submitted
    the engine's POST-expansion `output.prompt_token_ids` as a bare
    TokensPrompt. Those ids still contain the expanded `<|audio_pad|>` run for
    the user's spoken turn, but with no `multi_modal_data` the audio encoder
    output is gone - so the thinker saw placeholder tokens backed by nothing,
    lost the question entirely, and "answered" by emitting further tool calls
    for unrelated cities/items (and even fabricating tool results) instead of
    replying, never terminating. Fixed by splicing onto the PRE-expansion
    prompt and re-attaching its audio on every continuation."""

    def _conn(self, mocker):
        conn = RealtimeConnection.__new__(RealtimeConnection)
        conn.serving = mocker.Mock()
        conn.serving.model_config.is_encoder_decoder = False
        conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(side_effect=lambda prompts: [dict(prompts[0])])
        return conn

    @staticmethod
    def _first(gen):
        async def _run():
            return await anext(gen)

        return asyncio.run(_run())

    def test_audio_is_reattached_to_continuation_prompt(self, mocker) -> None:
        conn = self._conn(mocker)
        audio = {"audio": np.zeros(16000, dtype=np.float32)}

        result = self._first(conn._render_token_prompt([1, 2, 3], audio))

        assert result.prompt["multi_modal_data"] is audio

    def test_no_multi_modal_data_is_a_noop(self, mocker) -> None:
        conn = self._conn(mocker)

        result = self._first(conn._render_token_prompt([1, 2, 3]))

        assert "multi_modal_data" not in result.prompt

    def test_turn_prompt_capture_keeps_unexpanded_ids_and_audio(self, mocker) -> None:
        """_buffer_realtime_audio_with_tools must stash the pre-render prompt so
        the continuation has something audio-bearing to splice onto."""
        conn = self._conn(mocker)
        conn._tools = None
        conn._turn_prompt = None
        audio = {"audio": np.zeros(8000, dtype=np.float32)}
        prompt = TokensPrompt(prompt_token_ids=[10, 11], multi_modal_data=audio)

        async def _fake_buffer(*_args, **_kwargs):
            yield prompt

        conn.serving.model_cls.buffer_realtime_audio = _fake_buffer

        async def _drain():
            return [x async for x in conn._buffer_realtime_audio_with_tools(None, None)]

        asyncio.run(_drain())

        assert conn._turn_prompt["prompt_token_ids"] == [10, 11]
        assert conn._turn_prompt["multi_modal_data"] is audio


class TestCloseAssistantTurnBeforeToolResult:
    """Regression test for a real bug: the tool-result suffix opens with
    `<|im_start|>user`, but the raw generated token ids stop at the tool call
    without the `<|im_end|>` the chat template would emit. Splicing them
    directly produced `</tool_call><|im_start|>user`, leaving the assistant turn
    open. The thinker answered that malformed conversation by re-emitting the
    same tool call, looping until something bounded it - while the reference HF
    path (apply_chat_template, which closes the turn) answered the identical
    prompt correctly. Verified against a live Qwen3-Omni realtime session:
    tool results that previously looped 8+ times now answer in one round, with
    wording matching the reference implementation."""

    class _Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|im_end|>"
            return 151645

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert text == "\n"
            return [198]

    def test_appends_terminator_and_newline(self) -> None:
        out = RealtimeConnection._close_assistant_turn(self._Tok(), [1, 2, 3])
        assert out == [1, 2, 3, 151645, 198]

    def test_adds_only_newline_when_terminator_present(self) -> None:
        out = RealtimeConnection._close_assistant_turn(self._Tok(), [1, 2, 151645])
        assert out == [1, 2, 151645, 198]

    def test_is_idempotent_when_already_closed(self) -> None:
        out = RealtimeConnection._close_assistant_turn(self._Tok(), [1, 151645, 198])
        assert out == [1, 151645, 198]

    def test_does_not_mutate_caller_list(self) -> None:
        original = [1, 2, 3]
        RealtimeConnection._close_assistant_turn(self._Tok(), original)
        assert original == [1, 2, 3]

    def test_unknown_terminator_leaves_splice_untouched(self) -> None:
        class _Bad(self._Tok().__class__):
            def convert_tokens_to_ids(self, token: str) -> int:
                return 0  # == unk_token_id

        out = RealtimeConnection._close_assistant_turn(_Bad(), [1, 2, 3])
        assert out == [1, 2, 3]


class TestToolChainIsBounded:
    """A model that keeps re-emitting tool calls instead of answering must not
    recurse without bound: `_await_tool_results_and_continue` re-enters
    `_run_generation`, so an unbounded chain grows the stack and holds every
    round's result generator open. Observed during bring-up when the spliced
    prompt was malformed (see TestCloseAssistantTurnBeforeToolResult)."""

    def test_exceeding_max_rounds_reports_error_and_stops(self, tool_call_conn, mocker) -> None:
        tool_call_conn._is_connected = True
        tool_call_conn._tool_rounds = RealtimeConnection.MAX_TOOL_ROUNDS
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        run_generation = mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)

        asyncio.run(tool_call_conn._await_tool_results_and_continue([1, 2], [3]))

        send_error.assert_awaited_once()
        assert send_error.await_args.args[1] == "tool_call_loop"
        run_generation.assert_not_awaited()

    def test_rounds_below_the_cap_still_continue(self, tool_call_conn, mocker) -> None:
        """The cap must not fire early: with no pending calls the wait loop is
        skipped and generation continues."""
        tool_call_conn._is_connected = True
        tool_call_conn._tool_rounds = 0
        tool_call_conn._turn_prompt = None
        mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        run_generation = mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)
        mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.cached_tokenizer_from_config",
            return_value=mocker.Mock(
                encode=lambda text, add_special_tokens=True: [9],
                convert_tokens_to_ids=lambda token: 151645,
                unk_token_id=0,
            ),
        )
        mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.cached_processor_from_config")
        mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.safe_apply_chat_template",
            return_value="<|im_start|>user\nresult<|im_end|>\n<|im_start|>assistant\n",
        )
        tool_call_conn.serving = mocker.Mock()

        asyncio.run(tool_call_conn._await_tool_results_and_continue([1, 2], [3]))

        run_generation.assert_awaited_once()


class TestToolResultWaitReleasesOnDisconnect:
    """A client that vanishes mid-tool-call must not park the generation task
    forever. The wait is bounded so `_is_connected` is re-checked instead of
    blocking indefinitely on an empty queue."""

    def test_wait_returns_when_client_disconnects(self, tool_call_conn, mocker) -> None:
        tool_call_conn._is_connected = False  # client already gone
        tool_call_conn._pending_tool_calls = {0: _PendingToolCall(call_id="call_x", name="get_weather")}
        run_generation = mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)

        # Returns rather than hanging on the empty queue.
        asyncio.run(asyncio.wait_for(tool_call_conn._await_tool_results_and_continue([1], [2]), timeout=5))

        run_generation.assert_not_awaited()
