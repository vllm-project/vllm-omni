from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionToolsParam
from vllm.entrypoints.openai.engine.protocol import ToolCall, UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tool_parsers import ToolParserManager

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_protocol import (
    ConversationItemCreate,
    OmniSessionUpdate,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
)
from vllm_omni.entrypoints.openai.realtime_tool_format import (
    render_assistant_tool_call,
    render_tool_preamble,
    render_tool_result,
)
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)

# Fallback only: the tool parser is normally taken from the one the server already
# resolved for /v1/chat/completions (see _get_tool_parser), so a --tool-call-parser
# choice applies to realtime too. hermes covers the Qwen family, which is what
# this realtime path serves today.
_DEFAULT_TOOL_PARSER = "hermes"


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events.

    Reuses upstream vLLM websocket/session lifecycle and customizes generation
    output handling to emit audio deltas, plus tool/function calling: upstream's
    realtime protocol has no wire mechanism for tools, so ``session.update``
    accepts tool definitions here and function calls are surfaced using the
    OpenAI realtime event shapes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None

        # Tool-calling session state.
        self._tools: list[ChatCompletionToolsParam] | None = None
        self._tool_choice: Any = "auto"
        # Rendered turns waiting to be prepended to the next prompt. They travel
        # over ``input_stream``, the context channel upstream already threads from
        # the connection down into ``buffer_realtime_audio``.
        self._pending_context_tokens: list[int] = []
        # call_id -> function name, so a tool result coming back from the client
        # can be rendered even though the wire event only carries the call id.
        self._pending_tool_calls: dict[str, str] = {}
        self._tokenizer = None
        self._tool_parser = None

    async def start_generation(self):
        await super().start_generation()

    # ==================== tool calling ====================

    @property
    def tools_enabled(self) -> bool:
        return bool(self._tools)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = cached_tokenizer_from_config(self.serving.model_config)
        return self._tokenizer

    def _get_tool_parser(self):
        """Reuse whatever tool parser the server resolved for chat completions.

        That keeps a ``--tool-call-parser`` choice (and the model-family special
        cases that come with it) consistent between /v1/chat/completions and
        realtime, instead of pinning one syntax here.
        """
        if self._tool_parser is None:
            serving_chat = getattr(getattr(self.websocket, "app", None), "state", None)
            serving_chat = getattr(serving_chat, "openai_serving_chat", None)
            parser_cls = getattr(getattr(serving_chat, "parser_cls", None), "tool_parser_cls", None)
            if parser_cls is None:
                parser_cls = ToolParserManager.get_tool_parser(_DEFAULT_TOOL_PARSER)
            self._tool_parser = parser_cls(self._get_tokenizer())
        return self._tool_parser

    def _stage_context(self, text: str) -> None:
        """Queue a rendered turn to be prepended to the next prompt."""
        if not text:
            return
        self._pending_context_tokens.extend(self._get_tokenizer().encode(text, add_special_tokens=False))

    def _configure_tools(self, event: dict) -> None:
        """Read tool configuration out of a ``session.update`` event."""
        session_update = OmniSessionUpdate(**event)
        self._tool_choice = session_update.tool_choice
        if not session_update.tools or self._tool_choice == "none":
            self._tools = None
            return

        self._tools = session_update.tools
        self._pending_tool_calls.clear()
        self._stage_context(render_tool_preamble(self._tools))
        logger.debug(
            "Realtime session %s configured with %d tool(s)",
            self.connection_id,
            len(self._tools),
        )

    async def _handle_conversation_item_create(self, event: dict) -> None:
        """Accept a tool result and stage it as context for the next turn."""
        item = ConversationItemCreate(**event).item
        if item.call_id not in self._pending_tool_calls:
            logger.warning(
                "Realtime session %s: tool result for unknown call_id %s",
                self.connection_id,
                item.call_id,
            )
        self._pending_tool_calls.pop(item.call_id, None)
        self._stage_context(render_tool_result(item.output))

    def _tool_request(self) -> ChatCompletionRequest:
        """Minimal request object, required by the tool parser's API."""
        return ChatCompletionRequest(
            model=self.serving.model_config.served_model_name or "",
            messages=[],
            tools=self._tools,
            tool_choice=self._tool_choice,
        )

    def _extract_tool_calls(self, text: str) -> tuple[list[ToolCall], str]:
        """Split generated text into tool calls and the remaining spoken content."""
        if not text or not self.tools_enabled:
            return [], text
        try:
            extracted = self._get_tool_parser().extract_tool_calls(text, self._tool_request())
        except Exception:
            logger.exception("Realtime session %s: tool call extraction failed", self.connection_id)
            return [], text
        if not extracted.tools_called:
            return [], text
        return list(extracted.tool_calls), getattr(extracted, "content", None) or ""

    async def _emit_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Send the OpenAI realtime function-call events for each call."""
        for tool_call in tool_calls:
            call_id = tool_call.id
            name = tool_call.function.name
            arguments = tool_call.function.arguments or "{}"
            self._pending_tool_calls[call_id] = name
            item_id = f"item-{uuid4()}"
            await self.send_json(
                ResponseFunctionCallArgumentsDelta(
                    item_id=item_id,
                    call_id=call_id,
                    name=name,
                    delta=arguments,
                ).model_dump()
            )
            await self.send_json(
                ResponseFunctionCallArgumentsDone(
                    item_id=item_id,
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                ).model_dump()
            )
            # Keep the model's own call in context so the tool result the client
            # sends back has something to attach to.
            self._stage_context(render_assistant_tool_call(name, arguments))

    async def handle_event(self, event: dict):
        """Route events, adding the tool-related ones on top of upstream's."""
        event_type = event.get("type")
        if event_type == "session.update":
            await super().handle_event(event)
            if self._is_model_validated:
                self._configure_tools(event)
            return
        if event_type == "conversation.item.create":
            await self._handle_conversation_item_create(event)
            return
        await super().handle_event(event)

    # ==================== audio helpers ====================

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

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None
        tools_enabled = self.tools_enabled

        # Hand any staged context (tool definitions, a tool result, the model's
        # own previous tool call) to ``buffer_realtime_audio`` before the engine
        # starts consuming the prompt generator.
        if self._pending_context_tokens:
            input_stream.put_nowait(list(self._pending_context_tokens))
            self._pending_context_tokens.clear()

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
                    # With tools configured the connection owns the context: it
                    # stages properly rendered turns instead of echoing raw
                    # output tokens, so the two do not fight over the channel.
                    if new_token_ids and not tools_enabled:
                        input_stream.put_nowait(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    # With tools configured the text is held back until the turn
                    # ends: a tool call is only recognizable once its
                    # <tool_call>...</tool_call> block is complete, and streaming
                    # that raw markup as transcription would be wrong.
                    if delta_text and not tools_enabled:
                        await self.send(TranscriptionDelta(delta=delta_text))

                audio_chunks, sample_rate = self._extract_audio_chunks(output)

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

                if not self._is_connected:
                    break

            spoken_text = full_text
            if tools_enabled:
                tool_calls, spoken_text = self._extract_tool_calls(full_text)
                if spoken_text:
                    await self.send(TranscriptionDelta(delta=spoken_text))
                if tool_calls:
                    await self._emit_tool_calls(tool_calls)

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=spoken_text, usage=usage))

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
            # Always send terminal event so clients don't hang forever.
            if self._is_connected and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def send_json(self, payload: dict):
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception:
            # A failed send means the client is gone; flag it so the
            # generation loop stops instead of retrying into a dead socket.
            self._is_connected = False
            raise
