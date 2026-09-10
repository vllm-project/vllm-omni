from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.renderers.hf import safe_apply_chat_template
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_tool_calls import ToolCallDelta, ToolCallStreamState, extract_deltas
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)

# How long to block on a client tool result before re-checking the connection.
# Only bounds the wait between liveness checks, not the total wait: a tool may
# legitimately take a long time, but a client that has gone away must not leave
# the generation task parked forever.
_TOOL_RESULT_POLL_S = 0.5


def _text_of(contents: list[dict[str, Any]]) -> str:
    """Join the text parts of an OpenAI-Realtime content list.

    Accepts `input_text` as well as `text`: OpenAI's Realtime API uses `input_text`
    on a USER message and `text` on an assistant one, and callers port either way
    round. Mirrors the existing `input_audio`/`audio` tolerance next to it.
    """
    return " ".join(c.get("text") or "" for c in contents if c.get("type") in ("text", "input_text")).strip()


@dataclass
class _PendingToolCall:
    """A tool call being accumulated from the model's streamed output."""

    call_id: str
    name: str
    arguments: str = ""


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events, plus
    OpenAI-Realtime-shaped tool/function calling and voice (TTS speaker)
    selection.

    Reuses upstream vLLM websocket/session lifecycle and customizes
    generation output handling to emit audio deltas and tool-call events.

    Tool-calling protocol (mirrors OpenAI's Realtime API event shapes):
      - client -> server: `session.update` gains an optional `tools` field
        (list of OpenAI-style tool/function definitions).
      - server -> client, once the model starts a tool call:
        `response.output_item.added` (item.type="function_call", name, call_id)
        `response.function_call_arguments.delta` (call_id, delta)
        `response.function_call_arguments.done` (call_id, arguments)
      - client -> server, once the tool has run:
        `conversation.item.create` with
        `item = {"type": "function_call_output", "call_id": ..., "output": "..."}`
      - generation then continues automatically with the tool result appended,
        streaming the model's actual spoken reply as normal.

    Audio for a tool-call turn is not forwarded to the client once the
    `<tool_call>` tag has been parsed - the underlying 3-stage pipeline
    (thinker->talker->code2wav) still synthesizes it (there is no clean
    lower-level hook to skip talker/code2wav without changing the shared
    orchestrator - see PR description), the bytes are just dropped. Any audio
    that arrived before the tag was recognized has already been sent; in
    practice the tag appears in the thinker's text well ahead of the
    corresponding synthesized audio.

    A chain of tool calls is bounded by MAX_TOOL_ROUNDS.

    Voice selection: `session.update` gains an optional `voice` (or
    `speaker`) field, mirroring OpenAI's Realtime API. Previously there was
    no way to select a voice at all - every session silently used whichever
    key HF's `talker_config.speaker_id` lists first for the checkpoint.

    Instructions: `session.update` gains an optional `instructions` field
    (system prompt), mirroring OpenAI's Realtime API. Previously there was
    no way to set a system prompt at all for /v1/realtime.

    Cancellation: `response.cancel` aborts the turn in flight and replies with
    `response.done` (status "cancelled"), keeping the session. Previously the only
    way to stop server-side work was to close the connection, which also discarded
    the per-connection tools/voice/instructions/history.

    Scope and limitations of the tool-calling path:

    - **Non-duplex only.** This is the half-duplex `/v1/realtime` path. The
      full-duplex runtime under `experimental/fullduplex/` has no tool-calling
      support and shares no code with this.
    - **Requires `async_chunk` disabled.** `session.update` with `tools` is
      rejected when the server runs in async-chunk mode; see
      `_async_chunk_enabled` for why aggregating instead would not be enough.
      Voice selection is unaffected and works in either mode.
    - **Waits on client liveness.** Once the model has requested a tool, the turn
      blocks until a `function_call_output` arrives for every pending call. There
      is deliberately no deadline, because a slow tool is indistinguishable from
      an absent one; a client that disconnects releases the wait, and malformed
      or unknown results are reported back rather than silently dropped.
    """

    # Upper bound on consecutive tool-call rounds within one user turn.
    MAX_TOOL_ROUNDS = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None
        self._tools: list[dict[str, Any]] | None = None
        # The current turn's prompt as handed to the engine BEFORE multimodal
        # expansion: un-expanded `prompt_token_ids` (one `<|audio_pad|>`) plus the
        # audio in `multi_modal_data`. Tool-call continuations rebuild from this
        # rather than from the engine's post-expansion `output.prompt_token_ids`,
        # because those contain the expanded audio placeholder run with no way to
        # re-attach the audio - see _await_tool_results_and_continue.
        self._turn_prompt: dict[str, Any] | None = None
        # parser-assigned index (per generation) -> the call being accumulated
        self._pending_tool_calls: dict[int, _PendingToolCall] = {}
        self._tool_result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # Consecutive tool-call rounds in this turn, bounded by MAX_TOOL_ROUNDS.
        self._tool_rounds = 0
        self._speaker: str | None = None
        self._instructions: str | None = None
        # Prior conversation as an ordered message list, replayed into every
        # subsequent prompt. Entries are {"role": ..., "audio": np.ndarray} for a
        # user turn replayed as speech, and {"role": ..., "content": str} for
        # everything else (assistant, tool, or a user turn given as text). A flat
        # message list (rather than user/assistant pairs) is what lets a completed
        # TOOL turn be replayed in full - assistant `<tool_call>`, the
        # `<tool_response>` result, then the spoken answer - which the model needs
        # in order to keep calling tools on later turns. See _handle_history_item.
        self._history: list[dict[str, Any]] = []
        # Engine request id of the turn in flight, so response.cancel can abort the
        # stages directly instead of relying on connection teardown.
        self._active_request_id: str | None = None
        # Set while a cancel is being serviced, so _run_generation's terminal-event
        # path stays quiet and the client sees one `response.done` instead of a
        # spurious `response.audio.done` for a turn that never finished.
        self._cancelling = False

    @staticmethod
    def _decode_pcm16(b64_audio: str) -> np.ndarray:
        """Same PCM16 -> float32 conversion the base class applies to
        `input_audio_buffer.append`, so history audio and live audio are
        represented identically."""
        return np.frombuffer(base64.b64decode(b64_audio), dtype=np.int16).astype(np.float32) / 32768.0

    async def _handle_history_item(self, item: dict) -> None:
        """Append a prior conversation turn, OpenAI-Realtime shaped:

            {"type": "message", "role": "user",
             "content": [{"type": "input_audio", "audio": "<b64 pcm16>"}]}
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "..."}]}
            {"type": "message", "role": "user",                   # both is fine too
             "content": [{"type": "input_audio", "audio": "..."},
                         {"type": "input_text", "text": "..."}]}
            {"type": "message", "role": "assistant",
             "content": [{"type": "text", "text": "..."}]}
            {"type": "message", "role": "tool",
             "content": [{"type": "text", "text": "<tool result>"}]}

        A completed tool turn MUST be replayed in full: the assistant message
        carrying the `<tool_call>` block, then the `tool` message with its result,
        then the spoken answer. Replaying only the answer makes the history read as
        "the model answers these questions from its own knowledge", and it stops
        calling the tool on later turns and confabulates instead - reproduced on the
        reference HF path, so it is the prompt shape rather than any serving detail.
        Conversely a `<tool_call>` replayed with no matching result reads as an
        unanswered call, and the model retries it indefinitely.
        """
        role = item.get("role")
        contents = item.get("content") or []
        if role == "user":
            audio_b64 = next(
                (c.get("audio") for c in contents if c.get("type") in ("input_audio", "audio") and c.get("audio")),
                None,
            )
            text = _text_of(contents)
            if audio_b64 is None and not text:
                await self.send_error(
                    "history user message needs input_audio or text content",
                    "invalid_history_item",
                )
                return
            # Keep whichever parts were given, including both: a user turn may
            # legitimately be audio PLUS text (the spoken part and a written
            # instruction about it), which is what /v1/chat/completions already
            # supports for this checkpoint. Audio alone is the shape a live turn has;
            # text alone is what a caller with a transcript and no audio can offer.
            past: dict[str, Any] = {"role": "user"}
            if audio_b64 is not None:
                past["audio"] = self._decode_pcm16(audio_b64)
            if text:
                past["content"] = text
            self._history.append(past)
        elif role in ("assistant", "tool"):
            text = _text_of(contents)
            if not text:
                await self.send_error(f"history {role} message needs text content", "invalid_history_item")
                return
            self._history.append({"role": role, "content": text})
        else:
            await self.send_error(f"Unsupported history message role: {role!r}", "invalid_history_item")
            return
        logger.info("realtime history: %d message(s) queued for replay", len(self._history))

    async def handle_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "session.update":
            tools = event.get("tools")
            if tools is not None:
                if self._async_chunk_enabled():
                    # Refuse rather than half-work. Two independent things break
                    # under async_chunk: the buffer yields one TokensPrompt per
                    # segment, so a tool-call continuation would reattach only the
                    # final segment's audio and lose the start of the utterance;
                    # and the generation loop never sees one complete thinker turn
                    # to scan for a <tool_call> block. Aggregating the audio would
                    # fix only the first, leaving the feature looking supported
                    # while still broken -- so the limitation is explicit instead.
                    await self.send_error(
                        "Tool calling on /v1/realtime requires async_chunk to be disabled "
                        "(serve with --no-async-chunk); tools were not applied.",
                        "tools_require_no_async_chunk",
                    )
                else:
                    self._tools = tools
            # Voice selection is independent of the async_chunk gate above.
            speaker = event.get("voice") or event.get("speaker")
            if speaker is not None:
                self._speaker = speaker
            instructions = event.get("instructions")
            if instructions is not None:
                self._instructions = instructions
            await super().handle_event(event)
        elif event_type == "response.cancel":
            await self._cancel_active_generation()
        elif event_type == "conversation.item.create":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "function_call_output":
                await self._enqueue_tool_result(item)
            elif item_type == "message":
                await self._handle_history_item(item)
            else:
                await self.send_error(f"Unsupported conversation.item type: {item_type!r}", "unsupported_item")
        else:
            await super().handle_event(event)

    def _async_chunk_enabled(self) -> bool:
        """Whether the server runs in async-chunk mode.

        Read off ``model_config`` the same way ``serving_speech.py`` does
        (``:3232``, ``:3569``).
        """
        return bool(getattr(self.serving.model_config, "async_chunk", False))

    async def _enqueue_tool_result(self, item: dict) -> None:
        """Validate a `function_call_output` before it can influence generation.

        Without this, any dict carrying the right `type` was accepted: a missing
        or non-string `call_id` never matched a pending call, and `output` was
        coerced with `str()`. A client typo therefore left generation waiting
        with nothing reported back. Shape errors are protocol errors, so they are
        rejected here rather than discovered later.

        (Kept as explicit checks for now; supersede with pydantic tool-event
        models when those land.)
        """
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            await self.send_error(
                "function_call_output requires a non-empty string 'call_id'",
                "invalid_function_call_output",
            )
            return
        output = item.get("output")
        if not isinstance(output, str):
            await self.send_error(
                f"function_call_output 'output' must be a string, got {type(output).__name__}",
                "invalid_function_call_output",
            )
            return
        self._tool_result_queue.put_nowait(item)

    async def start_generation(self):
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # New user turn: reset the tool-round budget and discard any tool result
        # left over from a previous turn, which would otherwise be consumed as if
        # it answered one of this turn's calls.
        self._tool_rounds = 0
        while not self._tool_result_queue.empty():
            self._tool_result_queue.get_nowait()

        audio_stream = self.audio_stream_generator()
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._buffer_realtime_audio_with_tools(audio_stream, input_stream)
        self.generation_task = asyncio.create_task(self._run_generation(streaming_input_gen, input_stream))

    async def _cancel_active_generation(self) -> None:
        """Abort the turn in flight without dropping the session.

        Until this event the only way to stop server-side work was to close the
        connection - closing is what runs the engine-side abort in
        `_run_generation`'s finally - and that discards everything the connection
        holds (tools, voice, instructions, history), forcing a full reseed on the
        next turn. For a voice client where the user interrupts routinely, that made
        the common case the expensive one.

        The engine request is aborted FIRST so the three stages stop even if the
        reader task takes a moment to unwind; the task is then cancelled, since it
        may be parked in `_await_tool_results_and_continue` waiting on a tool result
        rather than inside the result generator, where an abort alone would not
        reach it.
        """
        task = self.generation_task
        request_id = self._active_request_id
        if (task is None or task.done()) and request_id is None:
            await self.send_error("No active response to cancel", "no_active_response")
            return

        self._cancelling = True
        try:
            if request_id is not None:
                try:
                    await self.engine.abort(request_id)
                except Exception:
                    logger.exception("Failed to abort engine request %s", request_id)
            if task is not None and not task.done():
                task.cancel()
                # We raised this CancelledError, so swallow it rather than letting it
                # propagate out of the event handler and tear down the connection.
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            self._cancelling = False

        self.generation_task = None
        self._active_request_id = None
        # A cancelled turn leaves the same debris a finished one would, and
        # start_generation only clears it on the NEXT commit: drop it now so a
        # cancel followed by a disconnect cannot strand it.
        self._pending_tool_calls.clear()
        while not self._tool_result_queue.empty():
            self._tool_result_queue.get_nowait()
        self._tool_rounds = 0
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()

        logger.info("realtime: cancelled in-flight generation (request_id=%s)", request_id)
        if self._is_connected:
            await self.send_json({"type": "response.done", "response": {"status": "cancelled"}})

    async def _render_prompt(self, prompt: PromptType) -> StreamingInput:
        model_config = self.serving.model_config
        parsed_prompt = parse_model_prompt(model_config, prompt)
        (engine_input,) = await self.serving.renderer.render_cmpl_async([parsed_prompt])
        # render_cmpl_async's internal pipeline (BaseRenderer.process_for_engine_async)
        # only carries over fields it explicitly knows about - additional_information
        # set on the pre-render prompt (buffer_realtime_audio's `speaker`) is silently
        # dropped unless reapplied to the *rendered* engine_input, exactly like
        # serving_chat.py._preprocess_chat does for /v1/chat/completions.
        additional_information = (
            parsed_prompt.get("additional_information") if isinstance(parsed_prompt, dict) else None
        )
        if additional_information:
            engine_input["additional_information"] = additional_information
        return StreamingInput(prompt=engine_input)

    async def _buffer_realtime_audio_with_tools(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Equivalent to `OpenAIServingRealtime.transcribe_realtime`, but
        threads `self._tools`/`self._speaker`/`self._instructions`/
        `self._history` through to the model's `buffer_realtime_audio`. The
        base class's `transcribe_realtime` has a fixed (audio_stream,
        input_stream, model_config) call signature with no seam for extra
        per-connection state like these, so this reimplements its (short)
        body directly rather than patching upstream vLLM."""
        stream_input_iter = self.serving.model_cls.buffer_realtime_audio(
            audio_stream,
            input_stream,
            self.serving.model_config,
            tools=self._tools,
            speaker=self._speaker,
            instructions=self._instructions,
            history=self._history,
        )
        async for prompt in stream_input_iter:
            # Remember the pre-expansion prompt so tool-call continuations can
            # re-anchor on the user's audio (see self._turn_prompt).
            if isinstance(prompt, dict):
                self._turn_prompt = dict(prompt)
            yield await self._render_prompt(prompt)

    async def _render_token_prompt(
        self,
        prompt_token_ids: list[int],
        multi_modal_data: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        # Tool-call continuation: re-attach the turn's audio. Without it the
        # `<|audio_pad|>` placeholder still sits in the token ids but has no
        # encoder output behind it, so the thinker cannot see what the user asked
        # and free-associates unrelated tool calls instead of answering.
        if multi_modal_data:
            token_prompt["multi_modal_data"] = multi_modal_data
        # Keep the selected voice for the model's actual spoken reply too, not just
        # the pre-tool-call turn. This path builds its own TokensPrompt rather than
        # going through buffer_realtime_audio, so without this the continuation
        # carries no `speaker` and silently falls back to the checkpoint default -
        # the voice audibly changes halfway through a tool-calling turn.
        if self._speaker:
            token_prompt["additional_information"] = {"speaker": [self._speaker]}
        yield await self._render_prompt(token_prompt)

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
        n = prev.shape[0]
        if n == 0:
            return True
        if curr.shape[0] < n:
            return False
        return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))

    def _raw_waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s) for the client.

        Some engine paths emit a growing cumulative waveform each step; others emit
        true per-step deltas. We support both without duplicating audio on the client.
        """
        if arr.size == 0:
            return []
        ref = self._realtime_audio_ref
        if ref is None:
            self._realtime_audio_ref = arr.copy()
            return [arr]
        if self._numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0] :]
            self._realtime_audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        # True per-step delta (not a prefix extension of what we have seen).
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            return [], 24000
        # Support both MultimodalPayload and plain dict
        if not isinstance(mm, Mapping):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        sample_rate_hz = int(sr)
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], sample_rate_hz

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.extend(self._raw_waveform_to_deltas(arr))
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.extend(self._raw_waveform_to_deltas(arr))
        return chunks, sample_rate_hz

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def _emit_tool_call_deltas(self, tool_deltas: list[ToolCallDelta]) -> None:
        for delta in tool_deltas:
            if delta.name is not None:
                call = _PendingToolCall(call_id=f"call_{uuid4().hex[:24]}", name=delta.name)
                self._pending_tool_calls[delta.index] = call
                await self.send_json(
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "function_call", "name": call.name, "call_id": call.call_id},
                    }
                )
            if delta.arguments_delta:
                call = self._pending_tool_calls.get(delta.index)
                if call is None:
                    continue  # shouldn't happen: name delta always precedes argument deltas for the same index
                call.arguments += delta.arguments_delta
                await self.send_json(
                    {
                        "type": "response.function_call_arguments.delta",
                        "call_id": call.call_id,
                        "delta": delta.arguments_delta,
                    }
                )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        self._active_request_id = request_id
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None

        request_prompt_token_ids: list[int] = []
        assistant_token_ids: list[int] = []
        tool_state = ToolCallStreamState()
        self._pending_tool_calls = {}

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        result_gen = None
        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)
                        assistant_token_ids.extend(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )
                        if not request_prompt_token_ids:
                            request_prompt_token_ids = list(output.prompt_token_ids)

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    if delta_text:
                        content_delta, tool_deltas = extract_deltas(full_text, tool_state)
                        if content_delta:
                            await self.send(TranscriptionDelta(delta=content_delta))
                        if tool_deltas:
                            await self._emit_tool_call_deltas(tool_deltas)

                audio_chunks, sample_rate = self._extract_audio_chunks(output)
                if audio_chunks and not tool_state.has_tool_calls():
                    for chunk in audio_chunks:
                        sent_audio = True
                        await self.send_json(
                            {
                                "type": "response.audio.delta",
                                "audio": self._pcm16_b64(chunk),
                                "format": "pcm16",
                                "sample_rate_hz": sample_rate,
                            }
                        )
                # else: a tool-call turn - the pipeline still synthesizes audio for the
                # raw <tool_call> text (no cheap hook to skip talker/code2wav for just
                # this turn), we just don't forward it to the client.

                if not self._is_connected:
                    break

            if tool_state.has_tool_calls():
                for call in self._pending_tool_calls.values():
                    await self.send_json(
                        {
                            "type": "response.function_call_arguments.done",
                            "call_id": call.call_id,
                            "arguments": call.arguments,
                        }
                    )
                if self._is_connected:
                    await self._await_tool_results_and_continue(request_prompt_token_ids, assistant_token_ids)
                return

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            # Close the generator explicitly so AsyncOmni.generate's cleanup
            # (input-pump cancellation and engine-side abort) runs now rather
            # than whenever the event loop garbage-collects the async
            # generator; the delay window is where a disconnected session
            # keeps cycling through the stages (issue #4271).
            if result_gen is not None:
                try:
                    await result_gen.aclose()
                except Exception:
                    logger.exception("Failed to close realtime result generator")
            self._active_request_id = None
            # Always send terminal event so clients don't hang forever - except when
            # a cancel is in flight, which sends its own `response.done`.
            if self._is_connected and not self._cancelling and not audio_done_sent and not tool_state.has_tool_calls():
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    @staticmethod
    def _close_assistant_turn(tokenizer, assistant_token_ids: list[int]) -> list[int]:
        """Terminate the model's tool-call turn with `<|im_end|>\\n` before a
        tool-result turn is appended.

        The tool-result suffix opens with `<|im_start|>user`, but the raw generated
        token ids stop at the tool call without the closing `<|im_end|>` that the
        chat template would emit. Splicing them directly yields
        `</tool_call><|im_start|>user`, leaving the assistant turn open - a
        malformed conversation the thinker responds to by re-emitting the same tool
        call instead of answering, looping until something bounds it. The reference
        HF path answers the identical prompt because `apply_chat_template` closes
        the turn. Idempotent: only the missing pieces are added.
        """
        ids = list(assistant_token_ids)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if im_end_id in (None, getattr(tokenizer, "unk_token_id", None)):
            return ids  # unexpected tokenizer; leave the splice untouched
        if im_end_id not in ids[-2:]:
            ids.append(im_end_id)
        if newline_ids and ids[-len(newline_ids) :] != newline_ids:
            ids.extend(newline_ids)
        return ids

    async def _await_tool_results_and_continue(
        self,
        prior_prompt_token_ids: list[int],
        assistant_token_ids: list[int],
    ) -> None:
        """Block until the client has submitted a `function_call_output` for
        every pending tool call from the turn that just finished, then splice
        the results onto the raw token sequence (prior prompt + the model's
        own generated tool-call tokens + a freshly-rendered tool-result turn)
        and continue generation - which streams the model's actual spoken
        reply as a normal `_run_generation` call, recursing again if the
        model chains into another tool call (bounded by MAX_TOOL_ROUNDS)."""
        self._tool_rounds += 1
        if self._tool_rounds > self.MAX_TOOL_ROUNDS:
            # A model that keeps re-emitting tool calls instead of answering
            # would otherwise recurse without bound. Observed during bring-up
            # when the spliced prompt was malformed (see _close_assistant_turn).
            logger.warning("tool-call chain exceeded %d rounds; abandoning turn", self.MAX_TOOL_ROUNDS)
            if self._is_connected:
                await self.send_error(
                    f"Tool-call chain exceeded {self.MAX_TOOL_ROUNDS} rounds without a final response",
                    "tool_call_loop",
                )
            return

        pending = dict(self._pending_tool_calls)
        call_id_to_index = {call.call_id: idx for idx, call in pending.items()}
        results_by_index: dict[int, str] = {}

        while len(results_by_index) < len(pending) and self._is_connected:
            # Bounded wait so a client that disappears mid-tool-call cannot park
            # this task forever; a slow-but-live client is unaffected.
            try:
                item = await asyncio.wait_for(self._tool_result_queue.get(), timeout=_TOOL_RESULT_POLL_S)
            except TimeoutError:
                continue
            call_id = item.get("call_id")
            idx = call_id_to_index.get(call_id)
            if idx is None:
                # Tell the client: silently dropping this would leave the turn
                # waiting for a result that is never going to match. Keep waiting
                # afterwards, since the correct result may still arrive.
                logger.warning("received function_call_output for unknown call_id=%s", call_id)
                await self.send_error(
                    f"No pending tool call with call_id={call_id!r}; expected one of {sorted(call_id_to_index)}",
                    "unknown_tool_call_id",
                )
                continue
            # `output` is validated as a string at ingress (_enqueue_tool_result).
            results_by_index[idx] = item["output"]

        if not self._is_connected:
            return

        model_config = self.serving.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        # Pass the processor's chat_template explicitly - see the matching
        # comment in Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio
        # for why relying on safe_apply_chat_template's own auto-resolution
        # is unsafe for this checkpoint.
        processor = cached_processor_from_config(model_config)
        # One `role="tool"` message PER result, in call order. The chat template
        # emits one <tool_response> block per tool message and groups consecutive
        # tool messages under a single <|im_start|>user turn, so passing separate
        # messages is what lets the model associate each result with its call.
        # Joining the results instead produces a single <tool_response> holding
        # both outputs, which breaks parallel calls.
        #
        # `sorted()` on the parser-assigned index is call order: extract_deltas
        # appends `tool_call_starts` in the order <tool_call> appears in the
        # generated text, so this is stable regardless of the order in which the
        # client returns the results.
        #
        # No `tools=` here: this continues a conversation whose token history
        # already carries the tools system preamble; it is not a fresh turn.
        tool_messages: list[dict[str, str]] = [
            {"role": "tool", "content": results_by_index[i]} for i in sorted(results_by_index)
        ]
        suffix_text = safe_apply_chat_template(
            model_config,
            tokenizer,
            tool_messages,
            chat_template=processor.chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        # Splice onto the turn's PRE-expansion prompt, not the engine's
        # post-expansion `output.prompt_token_ids`. The latter carries the expanded
        # `<|audio_pad|>` run, and re-submitting it as a bare TokensPrompt drops the
        # audio itself: the thinker then sees placeholder tokens with no encoder
        # output, loses the user's question entirely, and answers by inventing
        # further tool calls (unrelated cities/items) instead of replying. Falling
        # back to `prior_prompt_token_ids` only matters for a continuation that never
        # went through buffer_realtime_audio (no audio to lose in that case).
        base_prompt = self._turn_prompt or {}
        base_token_ids = list(base_prompt.get("prompt_token_ids") or prior_prompt_token_ids)
        multi_modal_data = base_prompt.get("multi_modal_data")
        continuation_token_ids = (
            base_token_ids + self._close_assistant_turn(tokenizer, assistant_token_ids) + tokenizer.encode(suffix_text)
        )

        # Advance the base so a chained tool call next round splices onto this
        # turn's full history while the audio stays attached exactly once, at the
        # front, still un-expanded.
        if self._turn_prompt is not None:
            self._turn_prompt = {**base_prompt, "prompt_token_ids": continuation_token_ids}

        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        await self._run_generation(self._render_token_prompt(continuation_token_ids, multi_modal_data), input_stream)

    async def send_json(self, payload: dict):
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception:
            # A failed send means the client is gone; flag it so the
            # generation loop stops instead of retrying into a dead socket.
            self._is_connected = False
            raise
