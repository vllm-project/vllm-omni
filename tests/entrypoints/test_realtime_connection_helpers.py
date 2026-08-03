# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from unittest import mock

import numpy as np
import pytest
import torch
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection, _PendingToolCall
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import _replay_messages

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


@dataclass
class _FakeModelConfig:
    """Only the field `_async_chunk_enabled` reads."""

    async_chunk: bool = False


@dataclass
class _FakeServing:
    model_config: _FakeModelConfig = field(default_factory=_FakeModelConfig)


@pytest.fixture
def tool_call_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._tools = None
    conn._tool_result_queue = asyncio.Queue()
    conn._pending_tool_calls = {}
    conn._tool_rounds = 0
    # Tool calling is only supported with async_chunk off, so that is the default
    # here; tests that care about the other mode set it explicitly.
    conn.serving = _FakeServing()
    # handle_event's session.update branch now also assigns these.
    conn._speaker = None
    conn._instructions = None
    return conn


@pytest.fixture
def instructions_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._instructions = None
    # ...and the siblings session.update also assigns.
    conn._tools = None
    conn._speaker = None
    # session.update now consults serving.model_config for the async_chunk gate.
    conn.serving = _FakeServing()
    return conn


@pytest.fixture
def speaker_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._speaker = None
    # ...and the siblings, so a session.update event exercises the whole branch.
    conn._tools = None
    conn._instructions = None
    # session.update now consults serving.model_config for the async_chunk gate.
    conn.serving = _FakeServing()
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


class TestRealtimeConnectionSpeakerRouting:
    """handle_event's voice/speaker-selection addition to session.update -
    threaded into buffer_realtime_audio's own `speaker` param (see
    Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio), the same
    `additional_information={"speaker": [...]}` shape /v1/chat/completions
    already uses (serving_chat.py)."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_voice_field(self, speaker_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)

        asyncio.run(speaker_conn.handle_event({"type": "session.update", "model": "qwen3-omni", "voice": "aiden"}))

        assert speaker_conn._speaker == "aiden"

    def test_session_update_captures_speaker_field(self, speaker_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)

        asyncio.run(speaker_conn.handle_event({"type": "session.update", "model": "qwen3-omni", "speaker": "ethan"}))

        assert speaker_conn._speaker == "ethan"

    def test_session_update_without_voice_or_speaker_leaves_existing_value_untouched(
        self, speaker_conn, mocker
    ) -> None:
        self._patch_base_handle_event(mocker)
        speaker_conn._speaker = "aiden"

        asyncio.run(speaker_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert speaker_conn._speaker == "aiden"

    def test_unrelated_event_types_still_delegate_to_base(self, speaker_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(speaker_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


class TestRealtimeConnectionInstructionsRouting:
    """handle_event's instructions (system prompt) addition to session.update."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_instructions(self, instructions_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
        event = {
            "type": "session.update",
            "model": "qwen3-omni",
            "instructions": "Only use tools directly relevant to what the user asked.",
        }

        asyncio.run(instructions_conn.handle_event(event))

        assert instructions_conn._instructions == "Only use tools directly relevant to what the user asked."

    def test_session_update_without_instructions_leaves_existing_value_untouched(
        self, instructions_conn, mocker
    ) -> None:
        self._patch_base_handle_event(mocker)
        instructions_conn._instructions = "existing"

        asyncio.run(instructions_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert instructions_conn._instructions == "existing"

    def test_unrelated_event_types_still_delegate_to_base(self, instructions_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(instructions_conn.handle_event(event))

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
        conn._speaker = None
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

    def test_continuation_keeps_the_selected_voice(self, mocker) -> None:
        """Without this the voice audibly changes mid-turn: the continuation is
        built as its own TokensPrompt, bypassing buffer_realtime_audio where the
        speaker is normally attached, so the spoken reply after a tool call fell
        back to the checkpoint default."""
        conn = self._conn(mocker)
        conn._speaker = "aiden"

        result = self._first(conn._render_token_prompt([1, 2, 3]))

        assert result.prompt["additional_information"] == {"speaker": ["aiden"]}

    def test_continuation_without_a_selected_voice_is_a_noop(self, mocker) -> None:
        conn = self._conn(mocker)
        conn._speaker = None

        result = self._first(conn._render_token_prompt([1, 2, 3]))

        assert "additional_information" not in result.prompt

    def test_no_multi_modal_data_is_a_noop(self, mocker) -> None:
        conn = self._conn(mocker)

        result = self._first(conn._render_token_prompt([1, 2, 3]))

        assert "multi_modal_data" not in result.prompt

    def test_turn_prompt_capture_keeps_unexpanded_ids_and_audio(self, mocker) -> None:
        """_buffer_realtime_audio_with_tools must stash the pre-render prompt so
        the continuation has something audio-bearing to splice onto."""
        conn = self._conn(mocker)
        conn._tools = None
        # _buffer_realtime_audio_with_tools also reads these now.
        conn._speaker = None
        conn._instructions = None
        conn._history = []
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


class TestParallelToolResultsRenderSeparateBlocks:
    """Review #5555: joining parallel tool results into one `role="tool"` message
    produced a single `<tool_response>` holding both outputs, so the model could
    not associate each result with its call. The chat template emits one
    `<tool_response>` per tool message (grouping consecutive tool messages under
    one user turn), so each result must be its own message, in call order.
    """

    @staticmethod
    def _capture_suffix(conn, mocker, pending: dict[int, _PendingToolCall]) -> list[dict[str, str]]:
        """Drive `_await_tool_results_and_continue` and return the message list
        handed to the chat template."""
        conn._is_connected = True
        conn._tool_rounds = 0
        conn._turn_prompt = None
        conn._pending_tool_calls = pending
        conn.serving = mocker.Mock()
        mocker.patch.object(conn, "_run_generation", new_callable=mocker.AsyncMock)
        mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.cached_tokenizer_from_config",
            return_value=mocker.Mock(
                encode=lambda text, add_special_tokens=True: [9],
                convert_tokens_to_ids=lambda token: 151645,
                unk_token_id=0,
            ),
        )
        mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.cached_processor_from_config")
        apply_tmpl = mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.safe_apply_chat_template",
            return_value="<|im_start|>user\nr<|im_end|>\n<|im_start|>assistant\n",
        )
        asyncio.run(conn._await_tool_results_and_continue([1, 2], [3]))
        return apply_tmpl.call_args.args[2]

    def test_two_results_become_two_tool_messages_in_call_order(self, tool_call_conn, mocker) -> None:
        pending = {
            0: _PendingToolCall(call_id="call_a", name="get_weather"),
            1: _PendingToolCall(call_id="call_b", name="get_weather"),
        }
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny 72"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_b", "output": "rainy 55"})

        messages = self._capture_suffix(tool_call_conn, mocker, pending)

        assert messages == [
            {"role": "tool", "content": "sunny 72"},
            {"role": "tool", "content": "rainy 55"},
        ]

    def test_call_order_is_kept_when_results_arrive_reversed(self, tool_call_conn, mocker) -> None:
        """Arrival order is the client's choice; call order is the model's."""
        pending = {
            0: _PendingToolCall(call_id="call_a", name="get_weather"),
            1: _PendingToolCall(call_id="call_b", name="get_weather"),
        }
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_b", "output": "rainy 55"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny 72"})

        messages = self._capture_suffix(tool_call_conn, mocker, pending)

        assert [m["content"] for m in messages] == ["sunny 72", "rainy 55"]

    def test_duplicate_result_for_one_call_does_not_desync_the_wait(self, tool_call_conn, mocker) -> None:
        pending = {0: _PendingToolCall(call_id="call_a", name="get_weather")}
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "first"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "second"})

        messages = self._capture_suffix(tool_call_conn, mocker, pending)

        assert messages == [{"role": "tool", "content": "first"}]


class TestToolResultValidation:
    """Review #5555: any dict with `type="function_call_output"` was enqueued, so a
    missing/non-string `call_id` or a non-string `output` was accepted and the turn
    then waited for a result that could never match - with nothing reported to the
    client. Shape problems are protocol errors."""

    @staticmethod
    def _create(item: dict) -> dict:
        return {"type": "conversation.item.create", "item": item}

    def _run(self, conn, mocker, item: dict):
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        asyncio.run(conn.handle_event(self._create(item)))
        return send_error

    def test_missing_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_non_string_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "call_id": 7, "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_non_string_output_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(
            tool_call_conn, mocker, {"type": "function_call_output", "call_id": "call_a", "output": {"a": 1}}
        )
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_well_formed_result_is_enqueued(self, tool_call_conn, mocker) -> None:
        send_error = self._run(
            tool_call_conn, mocker, {"type": "function_call_output", "call_id": "call_a", "output": "sunny"}
        )
        send_error.assert_not_awaited()
        assert tool_call_conn._tool_result_queue.qsize() == 1

    def test_unknown_call_id_is_reported_to_the_client(self, tool_call_conn, mocker) -> None:
        """Previously only logged, so a client typo produced silence."""
        tool_call_conn._is_connected = True
        tool_call_conn._tool_rounds = 0
        tool_call_conn._pending_tool_calls = {0: _PendingToolCall(call_id="call_a", name="get_weather")}
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_TYPO", "output": "sunny"})
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)

        async def _drive() -> None:
            task = asyncio.ensure_future(tool_call_conn._await_tool_results_and_continue([1], [2]))
            for _ in range(40):
                await asyncio.sleep(0.02)
                if send_error.await_count:
                    break
            tool_call_conn._is_connected = False
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(_drive())

        assert send_error.await_args.args[1] == "unknown_tool_call_id"


class TestToolsRejectedUnderAsyncChunk:
    """Review #5555: with async_chunk on, the buffer yields one TokensPrompt per
    segment, so a tool-call continuation reattached only the final segment and lost
    the start of the utterance. Aggregating the audio would not be enough - the
    generation loop also never sees one complete thinker turn to scan for a
    <tool_call> block - so tools are refused outright instead."""

    def _session_update(self, conn, mocker, async_chunk: bool):
        conn.serving = _FakeServing(_FakeModelConfig(async_chunk=async_chunk))
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        base = mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        asyncio.run(conn.handle_event({"type": "session.update", "model": "m", "tools": tools}))
        return send_error, base

    def test_tools_rejected_when_async_chunk_enabled(self, tool_call_conn, mocker) -> None:
        send_error, base = self._session_update(tool_call_conn, mocker, async_chunk=True)
        assert send_error.await_args.args[1] == "tools_require_no_async_chunk"
        assert tool_call_conn._tools is None
        base.assert_awaited_once()

    def test_tools_accepted_when_async_chunk_disabled(self, tool_call_conn, mocker) -> None:
        send_error, _ = self._session_update(tool_call_conn, mocker, async_chunk=False)
        send_error.assert_not_awaited()
        assert tool_call_conn._tools == [{"type": "function", "function": {"name": "get_weather"}}]


class TestRenderPromptPropagatesAdditionalInformation:
    """Regression test for a real bug: BaseRenderer.render_cmpl_async's
    internal pipeline (process_for_engine_async) only carries over fields
    it explicitly knows about, so `additional_information` set on the
    pre-render prompt (buffer_realtime_audio's `speaker`) was silently
    dropped and never reached the engine - the voice selection feature
    looked correct in isolation but produced no audible effect. Fixed by
    reapplying it to the rendered engine_input, mirroring how
    serving_chat.py._preprocess_chat does it for /v1/chat/completions."""

    def _conn_with_fake_renderer(self, mocker, rendered_engine_input: dict):
        conn = RealtimeConnection.__new__(RealtimeConnection)
        conn.serving = mocker.Mock()
        conn.serving.model_config.is_encoder_decoder = False
        conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(return_value=[rendered_engine_input])
        return conn

    def test_additional_information_survives_render(self, mocker) -> None:
        # render_cmpl_async's fake return simulates the real pipeline: it never
        # echoes back fields it doesn't recognize, so additional_information
        # must NOT be in here even though the input prompt has it.
        rendered = {"prompt_token_ids": [1, 2, 3]}
        conn = self._conn_with_fake_renderer(mocker, rendered)
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3], additional_information={"speaker": ["aiden"]})

        result = asyncio.run(conn._render_prompt(prompt))

        assert result.prompt["additional_information"] == {"speaker": ["aiden"]}

    def test_no_additional_information_is_a_noop(self, mocker) -> None:
        rendered = {"prompt_token_ids": [1, 2, 3]}
        conn = self._conn_with_fake_renderer(mocker, rendered)
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3])

        result = asyncio.run(conn._render_prompt(prompt))

        assert "additional_information" not in result.prompt


class TestConversationHistory:
    """Conversation history. The endpoint is audio-in and returns only the
    assistant's own reply text (no ASR of the user), so a user turn is normally
    replayed as its ORIGINAL AUDIO; text is accepted too for callers that have a
    transcript instead. Everything else is text.

    A completed TOOL turn must be replayed in full - assistant `<tool_call>`, the
    `tool` result, then the spoken answer. Replaying only the answer makes the
    history read as "the model answers these from its own knowledge" and it stops
    calling tools on later turns, confabulating instead; replaying a `<tool_call>`
    with no result reads as an unanswered call and it retries forever. Both were
    reproduced on the reference HF path, i.e. they are prompt-shape issues rather
    than serving details.
    """

    def _conn(self):
        conn = RealtimeConnection.__new__(RealtimeConnection)
        conn._history = []
        conn.send_error = mock.AsyncMock()
        return conn

    @staticmethod
    def _pcm_b64(samples: list[int]) -> str:
        return base64.b64encode(np.array(samples, dtype=np.int16).tobytes()).decode()

    def _user(self, samples):
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": self._pcm_b64(samples)}],
        }

    @staticmethod
    def _msg(role, text):
        return {"type": "message", "role": role, "content": [{"type": "text", "text": text}]}

    def test_user_audio_is_decoded_and_appended(self) -> None:
        conn = self._conn()
        asyncio.run(conn._handle_history_item(self._user([32767, -32768])))
        assert len(conn._history) == 1
        assert conn._history[0]["role"] == "user"
        np.testing.assert_allclose(conn._history[0]["audio"], [32767 / 32768.0, -1.0], rtol=1e-6)

    def test_full_tool_turn_is_preserved_in_order(self) -> None:
        """The regression this class exists for: all four messages, in order."""
        conn = self._conn()
        call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Madrid"}}\n</tool_call>'
        asyncio.run(conn._handle_history_item(self._user([1, 2])))
        asyncio.run(conn._handle_history_item(self._msg("assistant", call)))
        asyncio.run(conn._handle_history_item(self._msg("tool", "sunny and 72 degrees")))
        asyncio.run(conn._handle_history_item(self._msg("assistant", "It is sunny in Madrid.")))

        assert [m["role"] for m in conn._history] == ["user", "assistant", "tool", "assistant"]
        assert conn._history[1]["content"] == call
        assert conn._history[2]["content"] == "sunny and 72 degrees"

    def test_tool_role_is_accepted(self) -> None:
        conn = self._conn()
        asyncio.run(conn._handle_history_item(self._msg("tool", "12 units in stock")))
        assert conn._history == [{"role": "tool", "content": "12 units in stock"}]
        conn.send_error.assert_not_awaited()

    def test_user_text_is_accepted(self) -> None:
        """A caller with a text transcript and no audio (ASR elsewhere, or a
        migrated text conversation) can still seed history. Text costs a fraction
        of the tokens and bytes that replayed audio does."""
        conn = self._conn()
        asyncio.run(conn._handle_history_item(self._msg("user", "my name is Nathan")))
        assert conn._history == [{"role": "user", "content": "my name is Nathan"}]
        conn.send_error.assert_not_awaited()

    def test_user_input_text_is_accepted(self) -> None:
        """OpenAI's Realtime API uses `input_text` on a user message (and `text` on
        an assistant one), so both spellings are honoured - same tolerance the
        audio side already has for `input_audio`/`audio`."""
        conn = self._conn()
        asyncio.run(
            conn._handle_history_item(
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "my name is Nathan"}]}
            )
        )
        assert conn._history == [{"role": "user", "content": "my name is Nathan"}]
        conn.send_error.assert_not_awaited()

    def test_user_audio_and_text_are_both_kept(self) -> None:
        """A user turn may be audio PLUS text - the spoken part and a written
        instruction about it - which /v1/chat/completions already supports for this
        checkpoint. Neither part is discarded."""
        conn = self._conn()
        asyncio.run(
            conn._handle_history_item(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "audio": self._pcm_b64([16384])},
                        {"type": "input_text", "text": "translate that"},
                    ],
                }
            )
        )
        assert conn._history[0]["content"] == "translate that"
        np.testing.assert_allclose(conn._history[0]["audio"], [0.5], rtol=1e-6)

    def test_user_with_neither_audio_nor_text_is_rejected(self) -> None:
        conn = self._conn()
        asyncio.run(conn._handle_history_item({"type": "message", "role": "user", "content": []}))
        assert conn._history == []
        conn.send_error.assert_awaited_once()

    def test_user_with_blank_text_is_rejected(self) -> None:
        conn = self._conn()
        asyncio.run(conn._handle_history_item(self._msg("user", "   ")))
        assert conn._history == []
        conn.send_error.assert_awaited_once()

    def test_empty_text_is_rejected(self) -> None:
        conn = self._conn()
        asyncio.run(conn._handle_history_item(self._msg("assistant", "   ")))
        assert conn._history == []
        conn.send_error.assert_awaited_once()

    def test_unknown_role_is_rejected(self) -> None:
        conn = self._conn()
        asyncio.run(conn._handle_history_item(self._msg("system", "x")))
        assert conn._history == []
        conn.send_error.assert_awaited_once()

    def test_decode_matches_base_class_pcm16_conversion(self) -> None:
        out = RealtimeConnection._decode_pcm16(self._pcm_b64([0, 16384, -16384]))
        np.testing.assert_allclose(out, [0.0, 0.5, -0.5], rtol=1e-6)


class TestReplayMessagePairing:
    """`_replay_messages` emits the chat messages and the audio list together
    because they are matched POSITIONALLY: the Nth audio placeholder in the prompt
    binds to the Nth entry of multi_modal_data. If the two ever disagree the model
    attributes one turn's audio to a different turn - silently, with no error - so
    the count/order invariant is asserted directly here."""

    PH = "<|audio_start|><|audio_pad|><|audio_end|>"

    @staticmethod
    def _audio(n=8):
        return np.zeros(n, dtype=np.float32)

    def test_audio_turn_emits_placeholder_and_audio(self) -> None:
        msgs, audio = _replay_messages([{"role": "user", "audio": self._audio()}], self.PH, None)
        assert [m["content"] for m in msgs] == [self.PH]
        assert len(audio) == 1

    def test_text_user_turn_emits_text_and_no_audio(self) -> None:
        msgs, audio = _replay_messages([{"role": "user", "content": "my name is Nathan"}], self.PH, None)
        assert msgs == [{"role": "user", "content": "my name is Nathan"}]
        assert audio == []

    def test_mixed_audio_and_text_turn_renders_both(self) -> None:
        """Content is a plain string here, so a mixed turn is just concatenation -
        one placeholder (so one audio entry) followed by the text."""
        msgs, audio = _replay_messages(
            [{"role": "user", "audio": self._audio(), "content": "translate that"}], self.PH, None
        )
        assert msgs == [{"role": "user", "content": self.PH + "translate that"}]
        assert len(audio) == 1

    def test_placeholder_count_always_equals_audio_count(self) -> None:
        """The invariant, over a mixed history: audio user turns, text user turns,
        assistant and tool turns interleaved."""
        history = [
            {"role": "user", "content": "text one"},
            {"role": "assistant", "content": "reply one"},
            {"role": "user", "audio": self._audio()},
            {"role": "assistant", "content": "reply two"},
            {"role": "tool", "content": "sunny and 72"},
            {"role": "user", "audio": self._audio()},
            {"role": "user", "content": "text two"},
            {"role": "user", "audio": self._audio(), "content": "and this"},
        ]
        msgs, audio = _replay_messages(history, self.PH, None)
        assert len(msgs) == len(history)
        assert sum(1 for m in msgs if m["content"].startswith(self.PH)) == len(audio) == 3

    def test_order_is_preserved(self) -> None:
        history = [
            {"role": "user", "audio": np.full(4, 0.25, dtype=np.float32)},
            {"role": "assistant", "content": "a"},
            {"role": "user", "audio": np.full(4, 0.75, dtype=np.float32)},
        ]
        _, audio = _replay_messages(history, self.PH, None)
        assert [float(a[0]) for a in audio] == [0.25, 0.75]

    def test_instructions_lead_and_add_no_audio(self) -> None:
        msgs, audio = _replay_messages([{"role": "user", "audio": self._audio()}], self.PH, "be brief")
        assert msgs[0] == {"role": "system", "content": "be brief"}
        assert len(audio) == 1

    def test_empty_history_is_empty(self) -> None:
        assert _replay_messages([], self.PH, None) == ([], [])

    def test_unknown_role_defaults_to_assistant(self) -> None:
        msgs, audio = _replay_messages([{"content": "x"}], self.PH, None)
        assert msgs == [{"role": "assistant", "content": "x"}]
        assert audio == []


class TestModelStaysRegisteredAsMultimodal:
    """Guard for a real mistake: a module-level helper was added between
    `@MULTIMODAL_REGISTRY.register_processor(...)` and the model class, so the
    decorator registered the HELPER instead of the model. The file still parsed
    and every other unit test still passed - a decorator may legally decorate a
    function - but at runtime the renderer no longer saw the model as multimodal
    and every request died with `'HfRenderer' object has no attribute
    '_mm_req_counter'`. Assert the decorator is attached to the class."""

    def test_register_processor_decorates_the_model_class(self) -> None:
        import inspect

        from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni as mod

        src = inspect.getsource(mod)
        idx = src.index("@MULTIMODAL_REGISTRY.register_processor(")
        after = src[idx:]
        # The next top-level statement after the decorator block must be the class.
        following = [ln for ln in after.splitlines()[1:] if ln and not ln[0].isspace() and not ln.startswith(")")]
        assert following, "nothing follows the decorator"
        assert following[0].startswith("class Qwen3OmniMoeForConditionalGeneration"), (
            f"@MULTIMODAL_REGISTRY.register_processor decorates {following[0]!r}, "
            "not the model class - the model will not be treated as multimodal"
        )
