from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from openai.types import realtime as types
from pydantic import TypeAdapter
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams
from vllm.tool_parsers import ToolParserManager

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime.session import (
    ActiveResponse,
    AudioFullDuplexSessionState,
    ResponseUsage,
    _gen_id,
    merge_session_config,
)

logger = init_logger(__name__)

_CLIENT_EVENT_ADAPTER = TypeAdapter(types.RealtimeClientEvent)

SAMPLE_RATE_HZ = 24000
BYTES_PER_SAMPLE_PCM16 = 2
MAX_AUDIO_APPEND_BYTES = 15 * 1024 * 1024
MAX_INPUT_AUDIO_BUFFER_BYTES = 64 * 1024 * 1024

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"

# Empirically calibrated for Qwen3-Omni from 8,808 ms / 23 thinker tokens.
QWEN3_OMNI_MS_PER_TOKEN = 383.0

AUTO_TRUNCATION_TRIGGER_RATIO = 0.8
AUTO_TRUNCATION_TARGET_RATIO = 0.5


@dataclass
class _ToolParserRequest:
    """Fields consumed by vLLM tool parsers."""

    tools: list[dict] = field(default_factory=list)
    tool_choice: str = "auto"
    include_reasoning: bool = False
    skip_special_tokens: bool = True


@dataclass(slots=True)
class _ResolvedResponse:
    input: list[Any] | None
    instructions: str | None
    modalities: list[str]
    max_output_tokens: int | str
    tools: list[Any] | None
    tool_choice: Any
    metadata: Any


class OpenAIFullDuplexConnection:
    """Handle one OpenAI Realtime WebSocket session."""

    def __init__(
        self,
        websocket: WebSocket,
        engine: AsyncOmni,
        model_name: str,
        tool_call_parser: str | None = None,
        enable_auto_tool_choice: bool = False,
    ):
        self.ws = websocket
        self.engine = engine
        self.model_name = model_name
        self._tool_call_parser_name = tool_call_parser if enable_auto_tool_choice else None

        self.session = AudioFullDuplexSessionState()
        self.session.config.model = model_name

        self._connected = True
        self._response_task: asyncio.Task | None = None
        self._response_cancel_event = asyncio.Event()
        self._send_lock = asyncio.Lock()

        self._tokenizer: Any = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def handle_connection(self):
        try:
            await self.ws.accept()
            logger.info("[realtime] connection opened, session_id=%s", self.session.session_id)
            await self._send_session_created()
            await self._send_conversation_created()
            self._tokenizer = await self._resolve_tokenizer()

            while self._connected:
                try:
                    text = await self.ws.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    event = _CLIENT_EVENT_ADAPTER.validate_json(text)
                except Exception:
                    await self._send_error(
                        "Invalid or unrecognized client event",
                        "invalid_event",
                    )
                    continue
                await self._dispatch_event(event)
        except Exception:
            logger.exception("Unhandled error in realtime connection")
            await self._send_error("Internal server error", "server_error", error_type="server_error")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        self._connected = False
        await self._cancel_active_response()
        logger.info("[realtime] connection closed, session_id=%s", self.session.session_id)

    async def _resolve_tokenizer(self) -> Any:
        tokenizer = await self.engine.get_tokenizer()
        if getattr(tokenizer, "chat_template", None):
            return tokenizer
        try:
            from vllm.transformers_utils.processor import cached_processor_from_config

            preprocessor = await self.engine.get_input_preprocessor()
            model_config = preprocessor.model_config
            processor = cached_processor_from_config(model_config)
            if getattr(processor, "apply_chat_template", None):
                return processor
        except Exception:
            logger.warning("Could not load processor for chat templating")
        return tokenizer

    # ------------------------------------------------------------------ #
    #  Event dispatch                                                     #
    # ------------------------------------------------------------------ #

    async def _dispatch_event(self, event: types.RealtimeClientEvent):
        handlers = {
            types.SessionUpdateEvent: self._handle_session_update,
            types.InputAudioBufferAppendEvent: self._handle_audio_append,
            types.InputAudioBufferCommitEvent: self._handle_audio_commit,
            types.InputAudioBufferClearEvent: self._handle_audio_clear,
            types.ResponseCreateEvent: self._handle_response_create,
            types.ResponseCancelEvent: self._handle_response_cancel,
            types.ConversationItemCreateEvent: self._handle_item_create,
            types.ConversationItemDeleteEvent: self._handle_item_delete,
            types.ConversationItemRetrieveEvent: self._handle_item_retrieve,
            types.ConversationItemTruncateEvent: self._handle_item_truncate,
        }
        handler = handlers.get(type(event))
        if handler is None:
            await self._send_error(
                f"Unknown event type: {event.type}",
                "invalid_event",
                event_id=event.event_id,
            )
            return
        try:
            await handler(event)
        except Exception:
            logger.exception("Error handling event %s", event.type)
            await self._send_error(
                "Internal server error",
                "server_error",
                event_id=event.event_id,
                error_type="server_error",
            )

    # ------------------------------------------------------------------ #
    #  session.update                                                     #
    # ------------------------------------------------------------------ #

    async def _handle_session_update(self, event: types.SessionUpdateEvent):
        s = self.session
        cfg = self._sanitize_session_config(event.session)
        s.config = merge_session_config(s.config, cfg)
        await self._send_session_updated()

    def _sanitize_session_config(self, cfg: Any) -> types.RealtimeSessionCreateRequest:
        """Drop any session.update fields this server doesn't support instead
        of rejecting the whole update, so clients that always send their full
        default config (transcription, noise reduction, server-side VAD, MCP
        tools, ...) still get a working session rather than a hard error.
        """
        data = cfg.model_dump(exclude_unset=True)

        if "model" in data and data["model"] != self.model_name:
            data.pop("model")

        if "tools" in data:
            data["tools"] = [t for t in data["tools"] if (t or {}).get("type") != "mcp"]
        if (data.get("tool_choice") or {}).get("type") == "mcp" if isinstance(data.get("tool_choice"), dict) else False:
            data.pop("tool_choice")

        audio = data.get("audio")
        if audio is not None:
            inp = audio.get("input")
            if inp is not None:
                inp.pop("transcription", None)
                inp.pop("noise_reduction", None)
                if not self._is_pcm24_format(inp.get("format")):
                    inp.pop("format", None)
                td = inp.get("turn_detection")
                if isinstance(td, dict) and td.get("type") in ("server_vad", "semantic_vad"):
                    inp["turn_detection"] = None

            out = audio.get("output")
            if out is not None:
                if out.get("speed") not in (None, 1):
                    out.pop("speed", None)
                if not self._is_pcm24_format(out.get("format")):
                    out.pop("format", None)

        return types.RealtimeSessionCreateRequest.model_validate(data)

    @staticmethod
    def _unsupported_audio_option(audio: Any) -> str | None:
        if audio is None:
            return None
        audio_input = getattr(audio, "input", None)
        if audio_input is not None:
            fields = getattr(audio_input, "model_fields_set", set())
            if "transcription" in fields and audio_input.transcription is not None:
                return "Input audio transcription"
            if "noise_reduction" in fields and audio_input.noise_reduction is not None:
                return "Input audio noise reduction"
        output = getattr(audio, "output", None)
        if output is not None:
            fields = getattr(output, "model_fields_set", set())
            if "speed" in fields and output.speed is not None and output.speed != 1:
                return "Output audio speed"
        return None

    @staticmethod
    def _is_pcm24_format(audio_format: Any) -> bool:
        if audio_format is None:
            return True
        if isinstance(audio_format, str):
            return audio_format in ("audio/pcm", "pcm16")
        if isinstance(audio_format, dict):
            format_type = audio_format.get("type")
            rate = audio_format.get("rate", 24000)
        else:
            format_type = getattr(audio_format, "type", None)
            rate = getattr(audio_format, "rate", 24000)
        return format_type in ("audio/pcm", "pcm16") and rate == 24000

    @staticmethod
    def _uses_mcp(config: Any) -> bool:
        def object_type(value: Any) -> Any:
            return value.get("type") if isinstance(value, dict) else getattr(value, "type", None)

        return (
            any(object_type(tool) == "mcp" for tool in (getattr(config, "tools", None) or []))
            or object_type(getattr(config, "tool_choice", None)) == "mcp"
        )

    async def _send_unsupported_mcp(self, event_id: str | None) -> None:
        await self._send_error(
            "Remote MCP tools are not supported",
            "unsupported_feature",
            event_id=event_id,
        )

    # ------------------------------------------------------------------ #
    #  input_audio_buffer.append / .commit / .clear                       #
    # ------------------------------------------------------------------ #

    async def _handle_audio_append(self, event: types.InputAudioBufferAppendEvent):
        now = time.monotonic()
        last = getattr(self, "_last_append_wall_time", None)
        if last is not None:
            gap = now - last
            if gap > 0.5:
                logger.warning(
                    "[realtime] input_audio_buffer.append gap of %.2fs (receive loop may have stalled)",
                    gap,
                )
        self._last_append_wall_time = now

        if not event.audio:
            return
        try:
            audio_bytes = self._decode_pcm16(event.audio)
            if len(self.session.input_audio_buffer) + len(audio_bytes) > MAX_INPUT_AUDIO_BUFFER_BYTES:
                limit_mib = MAX_INPUT_AUDIO_BUFFER_BYTES // (1024 * 1024)
                raise ValueError(f"Input audio buffer exceeds the {limit_mib} MiB limit")
        except ValueError as exc:
            await self._send_error(str(exc), "invalid_request_error", event_id=event.event_id)
            return
        self.session.input_audio_buffer.extend(audio_bytes)

    @staticmethod
    def _decode_pcm16(audio: str, max_bytes: int = MAX_AUDIO_APPEND_BYTES) -> bytes:
        if len(audio) > 4 * ((max_bytes + 2) // 3):
            raise ValueError("Audio payload is too large")
        try:
            decoded = base64.b64decode(audio, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 audio data") from exc
        if len(decoded) > max_bytes:
            raise ValueError("Audio payload is too large")
        if len(decoded) % BYTES_PER_SAMPLE_PCM16:
            raise ValueError("PCM audio data must contain complete 16-bit samples")
        return decoded

    def _commit_audio_buffer(self) -> types.RealtimeConversationItemUserMessage | None:
        """Commit buffered audio as a user conversation item."""
        s = self.session
        if len(s.input_audio_buffer) == 0:
            return None
        pcm16 = np.frombuffer(bytes(s.input_audio_buffer), dtype=np.int16)
        audio_f32 = pcm16.astype(np.float32) / 32768.0
        item = types.RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            status="completed",
            content=[{"type": "input_audio", "audio": self._pcm16_b64(audio_f32)}],
        )
        s.insert_item(item)
        s.input_audio_buffer.clear()
        return item

    async def _commit_audio_buffer_and_announce(
        self,
    ) -> types.RealtimeConversationItemUserMessage | None:
        """Commit buffered audio and emit its events."""
        item = self._commit_audio_buffer()
        if item is None:
            return None
        idx = self.session.find_item_index(item.id)
        previous_item_id = self.session.items[idx - 1].id if idx else None
        await self._send_event(
            types.InputAudioBufferCommittedEvent(
                event_id=_gen_id("evt"),
                type="input_audio_buffer.committed",
                item_id=item.id,
                previous_item_id=previous_item_id,
            )
        )
        # Do not echo the client's potentially large audio payload.
        wire_item = item.model_copy(update={"content": []})
        await self._send_conversation_item_added_and_done(wire_item, previous_item_id)
        return item

    async def _handle_audio_commit(self, event: types.InputAudioBufferCommitEvent):
        s = self.session
        if len(s.input_audio_buffer) == 0:
            await self._send_error(
                "Input audio buffer is empty",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        await self._commit_audio_buffer_and_announce()

    async def _handle_audio_clear(self, event: types.InputAudioBufferClearEvent):
        self.session.input_audio_buffer.clear()
        await self._send_event(
            types.InputAudioBufferClearedEvent(
                event_id=_gen_id("evt"),
                type="input_audio_buffer.cleared",
            )
        )

    # ------------------------------------------------------------------ #
    #  response.create                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_tools_and_choice(s: AudioFullDuplexSessionState, response_cfg: Any):
        """Per-response overrides win over session.config, same pattern as
        output_modalities/max_output_tokens below."""
        tools = s.config.tools
        if response_cfg is not None and getattr(response_cfg, "tools", None) is not None:
            tools = response_cfg.tools
        tool_choice = s.config.tool_choice
        if response_cfg is not None and getattr(response_cfg, "tool_choice", None) is not None:
            tool_choice = response_cfg.tool_choice
        return tools, tool_choice

    def _resolve_response(self, response_cfg: Any) -> _ResolvedResponse:
        s = self.session
        tools, tool_choice = self._resolve_tools_and_choice(s, response_cfg)
        modalities = s.config.output_modalities
        max_output_tokens = s.config.max_output_tokens
        response_input = None
        instructions = None
        metadata = None
        if response_cfg is not None:
            if response_cfg.output_modalities is not None:
                modalities = response_cfg.output_modalities
            if response_cfg.max_output_tokens is not None:
                max_output_tokens = response_cfg.max_output_tokens
            response_input = response_cfg.input
            instructions = response_cfg.instructions
            if response_cfg.conversation == "none":
                raise ValueError("Out-of-band responses are not supported")
            metadata = response_cfg.metadata
            audio = getattr(response_cfg, "audio", None)
            unsupported = self._unsupported_audio_option(audio)
            if unsupported is not None:
                raise ValueError(f"{unsupported} is not supported")
            output = getattr(audio, "output", None) if audio is not None else None
            if output is not None and not self._is_pcm24_format(getattr(output, "format", None)):
                raise ValueError("Only 24 kHz PCM output audio is supported")

        if tools and tool_choice != "none" and self._tool_call_parser_name is None:
            raise ValueError("Function tools require --enable-auto-tool-choice and --tool-call-parser")

        resolved_input = self._resolve_response_input(response_input) if response_input is not None else None
        return _ResolvedResponse(
            input=resolved_input,
            instructions=instructions,
            modalities=list(modalities),
            max_output_tokens=max_output_tokens,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )

    async def _estimate_total_tokens(
        self,
        tools: list | None,
        *,
        instructions: str | None = None,
        items: list[Any] | None = None,
    ) -> int:
        t0 = time.monotonic()
        prompt = await self._build_full_prompt(tools=tools, instructions=instructions, items=items)
        total = len(prompt["prompt_token_ids"])
        audio_arrays = prompt.get("multi_modal_data", {}).get("audio", [])
        if audio_arrays:
            raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
            placeholder_len = len(raw_tok.encode(AUDIO_PLACEHOLDER, add_special_tokens=False))
            for arr, sr in audio_arrays:
                total += self._qwen3_omni_audio_token_count(arr.shape[0], sr) - placeholder_len
        elapsed = time.monotonic() - t0
        if elapsed > 0.05:
            logger.warning(
                "[realtime] _estimate_total_tokens took %.3fs (history=%d items, %d audio segments)",
                elapsed,
                len(items if items is not None else self.session.items),
                len(audio_arrays),
            )
        return total

    async def _maybe_truncate_history(self, response: _ResolvedResponse) -> bool:
        s = self.session
        max_model_len = getattr(self.engine.model_config, "max_model_len", None)
        if not max_model_len:
            return True

        truncation = s.config.truncation or "auto"
        ratio = 1.0
        custom_limit = None
        if truncation == "disabled":
            mode = "disabled"
        elif truncation == "auto":
            mode = "auto"
        else:
            mode = "retention_ratio"
            ratio = truncation.retention_ratio
            if truncation.token_limits is not None:
                custom_limit = truncation.token_limits.post_instructions

        reserved_output = response.max_output_tokens if isinstance(response.max_output_tokens, int) else 0
        limit = custom_limit if custom_limit is not None else max(0, max_model_len - reserved_output)

        if mode == "auto":
            trigger = int(limit * AUTO_TRUNCATION_TRIGGER_RATIO)
            target = int(limit * AUTO_TRUNCATION_TARGET_RATIO)
        elif mode == "retention_ratio":
            trigger = limit
            target = int(limit * ratio)
        else:  # disabled
            trigger = limit
            target = limit

        persistent = response.input is None
        items = s.items if persistent else response.input
        total = await self._estimate_total_tokens(
            response.tools,
            instructions=response.instructions,
            items=items,
        )
        if total <= trigger:
            return True
        if mode == "disabled":
            logger.warning(
                "[realtime] token budget exceeded (%d/%d) and truncation is disabled -- rejecting response.create",
                total,
                limit,
            )
            return False

        idx = 0
        while total > target and idx < len(items):
            item = items[idx]
            if getattr(item, "role", None) == "system":
                idx += 1
                continue
            if persistent and item.id is not None and s.item_in_progress.get(item.id, False):
                idx += 1
                continue

            remove_indexes = [idx]
            if item.type in ("function_call", "function_call_output"):
                call_id = getattr(item, "call_id", None)
                pair_idx = next(
                    (
                        other_idx
                        for other_idx, other in enumerate(items)
                        if other_idx != idx and getattr(other, "call_id", None) == call_id
                    ),
                    None,
                )
                if pair_idx is not None:
                    remove_indexes.append(pair_idx)

            for remove_idx in sorted(remove_indexes, reverse=True):
                removed = items[remove_idx]
                if persistent:
                    s.remove_item(removed.id)
                    await self._send_event(
                        types.ConversationItemDeletedEvent(
                            event_id=_gen_id("evt"),
                            type="conversation.item.deleted",
                            item_id=removed.id,
                        )
                    )
                else:
                    del items[remove_idx]
            total = await self._estimate_total_tokens(
                response.tools,
                instructions=response.instructions,
                items=items,
            )

        return total <= limit

    async def _handle_response_create(self, event: types.ResponseCreateEvent):
        s = self.session
        response_cfg = event.response

        if response_cfg is not None and self._uses_mcp(response_cfg):
            await self._send_unsupported_mcp(event.event_id)
            return

        try:
            response = self._resolve_response(response_cfg)
        except ValueError as exc:
            await self._send_error(str(exc), "invalid_request_error", event_id=event.event_id)
            return

        if not await self._maybe_truncate_history(response):
            await self._send_error(
                "The response input exceeds the model's input token limit",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        response_id = _gen_id("resp")
        await self._send_event(
            types.ResponseCreatedEvent(
                event_id=_gen_id("evt"),
                type="response.created",
                response=self._response_object(response_id, response, "in_progress"),
            )
        )

        if s.active_response is not None:
            await self._cancel_active_response()

        s.active_response = ActiveResponse(response_id=response_id, request_id=f"rt-{response_id}")

        self._response_cancel_event.clear()
        self._response_task = asyncio.create_task(self._run_response(response_id, response))

    def _response_object(
        self,
        response_id: str,
        response: _ResolvedResponse,
        status: str,
        *,
        output: list[Any] | None = None,
        status_details: Any = None,
        usage: Any = None,
    ) -> types.RealtimeResponse:
        return types.RealtimeResponse(
            id=response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,
            output=output or [],
            conversation_id=self.session.conversation_id,
            output_modalities=response.modalities,
            max_output_tokens=response.max_output_tokens,
            metadata=response.metadata,
            usage=usage,
        )

    async def _run_response(self, response_id: str, response: _ResolvedResponse):
        s = self.session
        active = s.active_response
        if active is None:
            return

        completed = False
        try:
            await self._run_response_inner(response_id, response, s, active)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("_run_response failed for %s", response_id)
        finally:
            if not completed:
                await self._fail_response(s, response_id, response, active.item_id)
            s.active_response = None

    async def _fail_response(
        self,
        s: AudioFullDuplexSessionState,
        response_id: str,
        response: _ResolvedResponse,
        item_id: str | None,
    ) -> None:
        if item_id is not None and s.item_in_progress.pop(item_id, False):
            s.pending_truncations_ms.pop(item_id, None)
            s.remove_item(item_id)
        try:
            await self._send_event(
                types.ResponseDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.done",
                    response=self._response_object(
                        response_id,
                        response,
                        "failed",
                        status_details={
                            "type": "failed",
                            "error": {"type": "server_error", "code": None},
                        },
                    ),
                )
            )
        except Exception:
            logger.debug("Failed to send failure response.done for %s", response_id, exc_info=True)

    async def _run_response_inner(self, response_id, response, s, active):
        previous_item_id = s.items[-1].id if s.items else None
        modalities = response.modalities
        is_audio = "audio" in modalities
        tools, tool_choice = response.tools, response.tool_choice
        prompt = await self._build_full_prompt(
            tools=tools,
            instructions=response.instructions,
            items=response.input,
        )

        item_id = _gen_id("item")
        active.item_id = item_id
        output_index = 0

        converted_tools = self._convert_tools(tools) if tools else []
        tool_parser = None
        structural_tag_json = None
        if converted_tools and tool_choice != "none" and self._tool_call_parser_name:
            tool_parser_cls = ToolParserManager.get_tool_parser(self._tool_call_parser_name)
            strict_tools = [ChatCompletionToolsParam(**t) for t in self._convert_tools(tools, strict=True)]
            tool_parser = tool_parser_cls(self._tokenizer, tools=strict_tools)
            structure_tag = tool_parser.get_structural_tag(
                _ToolParserRequest(tools=strict_tools, tool_choice=tool_choice or "auto"),
                reasoning=False,
            )
            if structure_tag is not None:
                structural_tag_json = json.dumps(structure_tag.model_dump())

        item_obj = types.RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            id=item_id,
            status="in_progress",
            content=[],
        )

        try:
            s.insert_item(item_obj, previous_item_id=previous_item_id or "root")
        except ValueError:
            logger.warning(
                "[realtime] previous item '%s' vanished while starting response %s; appending instead",
                previous_item_id,
                response_id,
            )
            s.insert_item(item_obj)
        s.item_in_progress[item_id] = True

        await self._send_event(
            types.ResponseOutputItemAddedEvent(
                event_id=_gen_id("evt"),
                type="response.output_item.added",
                response_id=response_id,
                output_index=output_index,
                item=item_obj,  # type: ignore[arg-type]
            )
        )

        content_index = 0
        part_type = "audio" if is_audio else "text"
        part_obj = {"type": part_type, "text": "", "audio": "", "transcript": ""}

        await self._send_event(
            types.ResponseContentPartAddedEvent(
                event_id=_gen_id("evt"),
                type="response.content_part.added",
                response_id=response_id,
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                part=part_obj,  # type: ignore[arg-type]
            )
        )

        full_text = ""
        full_transcript = ""
        full_token_ids: list[int] = []
        cancelled = False
        usage = ResponseUsage()
        total_audio_samples = 0

        previous_text = ""
        previous_token_ids: list[int] = []
        pending_tool_calls: dict[int, dict[str, Any]] = {}
        next_output_index = 1  # 0 is the message item, reserved above
        # Qwen's talker speaks tool markup, so stop audio after detecting a call.
        tool_call_seen = False

        async def emit_content_delta(piece: str) -> None:
            nonlocal full_text, full_transcript
            if not piece:
                return
            full_transcript += piece
            if is_audio:
                await self._send_event(
                    types.ResponseAudioTranscriptDeltaEvent(
                        event_id=_gen_id("evt"),
                        type="response.output_audio_transcript.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=piece,
                    )
                )
            else:
                full_text += piece
                await self._send_event(
                    types.ResponseTextDeltaEvent(
                        event_id=_gen_id("evt"),
                        type="response.output_text.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=piece,
                    )
                )

        async def handle_tool_parser_delta(delta_msg) -> None:
            nonlocal next_output_index, tool_call_seen
            if delta_msg is None:
                return
            if delta_msg.content:
                await emit_content_delta(delta_msg.content)
            for tc in delta_msg.tool_calls:
                entry = pending_tool_calls.get(tc.index)
                if entry is None:
                    tool_call_seen = True
                    entry = {
                        "item_id": _gen_id("item"),
                        "call_id": tc.id or _gen_id("call"),
                        "name": tc.function.name if tc.function else None,
                        "arguments": "",
                        "output_index": next_output_index,
                    }
                    pending_tool_calls[tc.index] = entry
                    next_output_index += 1
                    await self._send_event(
                        types.ResponseOutputItemAddedEvent(
                            event_id=_gen_id("evt"),
                            type="response.output_item.added",
                            response_id=response_id,
                            output_index=entry["output_index"],
                            item=types.RealtimeConversationItemFunctionCall(
                                type="function_call",
                                id=entry["item_id"],
                                call_id=entry["call_id"],
                                name=entry["name"] or "",
                                arguments="",
                                status="in_progress",
                            ),  # type: ignore[arg-type]
                        )
                    )
                elif not entry["name"] and tc.function and tc.function.name:
                    entry["name"] = tc.function.name

                if tc.function and tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
                    await self._send_event(
                        types.ResponseFunctionCallArgumentsDeltaEvent(
                            event_id=_gen_id("evt"),
                            type="response.function_call_arguments.delta",
                            response_id=response_id,
                            item_id=entry["item_id"],
                            output_index=entry["output_index"],
                            call_id=entry["call_id"],
                            delta=tc.function.arguments,
                        )
                    )

        # Defaults are process-wide mutable objects.
        sampling_params_list = [
            sp.clone() if isinstance(sp, SamplingParams) else sp for sp in self.engine.default_sampling_params_list
        ]
        max_output_tokens = response.max_output_tokens
        thinker_params_configured = False
        for sp in sampling_params_list:
            if isinstance(sp, SamplingParams):
                sp.output_kind = RequestOutputKind.DELTA
                if not thinker_params_configured:
                    if isinstance(max_output_tokens, int):
                        sp.max_tokens = max_output_tokens
                    if structural_tag_json is not None:
                        sp.structured_outputs = StructuredOutputsParams(structural_tag=structural_tag_json)
                    thinker_params_configured = True

        gen = self.engine.generate(
            prompt=prompt,
            request_id=active.request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=modalities,
        )

        try:
            async for output in gen:
                if not self._connected:
                    cancelled = True
                    break

                output_type = getattr(output, "final_output_type", "text")
                if output_type == "audio":
                    audio_chunks = self._extract_audio_deltas(output)
                    for chunk in audio_chunks:
                        total_audio_samples += chunk.shape[0]
                        # is_audio guard is defense-in-depth: generate() is
                        # now given output_modalities=modalities above, so
                        # the engine shouldn't produce audio-typed output
                        # for a text-only response -- but don't rely on
                        # that alone to honor the client's explicit
                        # request; never flip session state or forward
                        # audio it didn't ask for.
                        if not is_audio:
                            continue
                        if tool_call_seen:
                            continue
                        b64 = self._pcm16_b64(chunk)
                        await self._send_event(
                            types.ResponseAudioDeltaEvent(
                                event_id=_gen_id("evt"),
                                type="response.output_audio.delta",
                                response_id=response_id,
                                item_id=item_id,
                                output_index=output_index,
                                content_index=content_index,
                                delta=b64,
                            )
                        )
                    continue

                if output.outputs:
                    first_out = output.outputs[0]
                    delta_text = first_out.text or ""
                    delta_token_ids = list(first_out.token_ids)
                    usage.output_tokens += len(delta_token_ids)
                    # Raw thinker token stream, in talker-consumption order --
                    # this is what _qwen3_omni_truncate_transcript correlates
                    # against codec frames, independent of any tool-parser
                    # stripping applied to full_text/full_transcript below
                    # (the talker speaks the raw stream, tool tags included).
                    full_token_ids.extend(delta_token_ids)

                    if output.prompt_token_ids:
                        usage.input_tokens = max(usage.input_tokens, len(output.prompt_token_ids))

                    if tool_parser is not None:
                        # Additive branch: when no tools are configured for
                        # this response, tool_parser is None and this whole
                        # block is skipped -- the plain-text path below is
                        # untouched.
                        current_text = previous_text + delta_text
                        current_token_ids = previous_token_ids + delta_token_ids
                        delta_msg = None
                        if delta_text or delta_token_ids:
                            delta_msg = tool_parser.extract_tool_calls_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                delta_token_ids,
                                request=_ToolParserRequest(tools=converted_tools, tool_choice=tool_choice or "auto"),
                            )
                        previous_text = current_text
                        previous_token_ids = current_token_ids
                        await handle_tool_parser_delta(delta_msg)
                    elif delta_text:
                        await emit_content_delta(delta_text)

        except asyncio.CancelledError:
            cancelled = True
        finally:
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.debug("Error closing generator for %s", active.request_id, exc_info=True)

        if tool_parser is not None and getattr(tool_parser, "engine_based_streaming", False):
            # finish_streaming() only exists on the newer ParserEngine-based
            # parsers (engine_based_streaming=True, e.g. Qwen3EngineToolParser)
            # -- the base ToolParser class legacy regex-based parsers extend
            # (e.g. Hermes2ProToolParser) don't declare it at all and would
            # raise AttributeError here (confirmed in production logs).
            await handle_tool_parser_delta(tool_parser.finish_streaming())

        if self._response_cancel_event.is_set():
            cancelled = True

        status = "cancelled" if cancelled else "completed"

        if is_audio:
            await self._send_event(
                types.ResponseAudioDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_audio.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                )
            )
            await self._send_event(
                types.ResponseAudioTranscriptDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_audio_transcript.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                    transcript=full_transcript,
                )
            )
        else:
            await self._send_event(
                types.ResponseTextDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_text.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                    text=full_text,
                )
            )

        # Reconstruct rather than mutate item_obj.status/.content in place:
        # pydantic does not validate/coerce plain attribute assignment after
        # construction, so `item_obj.content = [{...}]` would silently leave
        # item_obj.content holding raw dicts instead of Content objects --
        # _assistant_item_text's attribute-based access (part.transcript)
        # then always returns None for those, so no assistant turn's text
        # ever actually made it into a later response's prompt.
        item_obj = types.RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            id=item_id,
            status="completed" if not cancelled else "incomplete",
            content=(
                [{"type": "output_audio", "transcript": full_transcript}]  # type: ignore[list-item]
                if is_audio
                else [{"type": "output_text", "text": full_text}]  # type: ignore[list-item]
            ),
        )

        done_part = {"type": part_type, "text": full_text, "transcript": full_transcript}
        await self._send_event(
            types.ResponseContentPartDoneEvent(
                event_id=_gen_id("evt"),
                type="response.content_part.done",
                response_id=response_id,
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                part=done_part,  # type: ignore[arg-type]
            )
        )

        await self._send_event(
            types.ResponseOutputItemDoneEvent(
                event_id=_gen_id("evt"),
                type="response.output_item.done",
                response_id=response_id,
                output_index=output_index,
                item=item_obj,  # type: ignore[arg-type]
            )
        )

        # Per spec, response.done always includes every output item that was
        # generated, regardless of final status -- so the item exists in
        # history either way. For a cancelled response we can't know how
        # much of it the user actually heard (no alignment between audio
        # timing and text/token position -- see conversation.item.truncate),
        # so rather than guess, the history copy gets empty content; the
        # wire events above already carried the real accumulated content.
        history_item = item_obj
        if cancelled:
            history_item = types.RealtimeConversationItemAssistantMessage(
                type="message",
                role="assistant",
                id=item_obj.id,
                status="incomplete",
                content=(
                    [{"type": "output_audio", "transcript": ""}]  # type: ignore[list-item]
                    if is_audio
                    else [{"type": "output_text", "text": ""}]  # type: ignore[list-item]
                ),
            )

        # A response that only called tools has no message item in `output`
        # (matches real OpenAI behavior) -- drop the placeholder rather than
        # keep an empty message. Scoped to a response that actually
        # completed: a cancelled response keeps today's empty-incomplete
        # message-item behavior regardless of any in-flight tool call, since
        # cancellation-mid-tool-call isn't handled specially here.
        #
        # total_audio_samples == 0 is required, not just empty text: the
        # tool parser classifies raw <tool_call>...</tool_call> text as
        # "not content", but the talker has no concept of that span and
        # synthesizes audio for the whole segment regardless (nothing in
        # _thinker_to_talker_prefill special-cases tool-call text) -- so a
        # "no content" tool-call response can still have real audio that was
        # actually streamed to and played by the client. Dropping the
        # message item in that case orphans that audio: a later
        # conversation.item.truncate against it fails with "not found"
        # (confirmed in production logs, audio_end_ms=6201 on an item that
        # had already been dropped here).
        drop_message_item = (
            not cancelled
            and total_audio_samples == 0
            and not (full_text or full_transcript)
            and bool(pending_tool_calls)
        )

        chain_after = previous_item_id
        if s.find_item_index(item_id) is not None:
            # Captured before remove_item below, whose own cleanup pops
            # pending_truncations_ms as part of deleting the item -- reading
            # it only after (as the item_in_progress.pop block used to)
            # would silently lose a truncate that arrived while this
            # response was still in progress and later turned out to be
            # tool-call-only (drop_message_item), with no ack or error ever
            # sent back to the client.
            pending_ms = s.pending_truncations_ms.get(item_id)
            if drop_message_item:
                s.remove_item(item_id)
                if pending_ms is not None:
                    await self._send_error(
                        f"Item '{item_id}' produced no message content to truncate",
                        "invalid_request_error",
                    )
            else:
                s.replace_item(history_item)
                if item_obj.id:
                    s.item_duration_ms[item_obj.id] = total_audio_samples / SAMPLE_RATE_HZ * 1000
                    # Skip storing for tool-call responses: full_token_ids is
                    # the raw thinker stream (tool-call tags included), but
                    # this item's transcript/text is the tool-parser-stripped
                    # content -- the two no longer line up token-for-token,
                    # so _qwen3_omni_truncate_transcript falls back to
                    # blanking (today's behavior) rather than risk splicing
                    # raw <tool_call> text into a truncated transcript.
                    if not pending_tool_calls:
                        s.item_token_ids[item_obj.id] = full_token_ids
                await self._send_conversation_item_added_and_done(history_item, previous_item_id)
                chain_after = history_item.id

            # Only now -- after item_duration_ms/item_token_ids are finally
            # populated (or the item is gone, for drop_message_item) -- is
            # it safe to resolve a conversation.item.truncate that arrived
            # while this response was still item_in_progress (see
            # _handle_item_truncate). Clearing item_in_progress first means
            # a truncate arriving from here on goes straight through
            # _handle_item_truncate's normal, non-deferred path.
            s.item_in_progress.pop(item_id, None)
            if not drop_message_item and pending_ms is not None:
                await self._do_item_truncate(
                    types.ConversationItemTruncateEvent(
                        event_id=_gen_id("evt"),
                        type="conversation.item.truncate",
                        item_id=item_id,
                        content_index=0,
                        audio_end_ms=pending_ms,
                    )
                )

        # Function calls aren't subject to the truncate-race protection the
        # message item needed above -- per spec only assistant message items
        # can ever be truncated, so there's no client action that could race
        # ahead and remove one of these before we get here.
        function_call_items: list[types.RealtimeConversationItemFunctionCall] = []
        for entry in pending_tool_calls.values():
            await self._send_event(
                types.ResponseFunctionCallArgumentsDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.function_call_arguments.done",
                    response_id=response_id,
                    item_id=entry["item_id"],
                    output_index=entry["output_index"],
                    call_id=entry["call_id"],
                    name=entry["name"] or "",
                    arguments=entry["arguments"],
                )
            )
            fc_item = types.RealtimeConversationItemFunctionCall(
                type="function_call",
                id=entry["item_id"],
                call_id=entry["call_id"],
                name=entry["name"] or "",
                arguments=entry["arguments"],
                status="completed" if not cancelled else "incomplete",
            )
            await self._send_event(
                types.ResponseOutputItemDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_item.done",
                    response_id=response_id,
                    output_index=entry["output_index"],
                    item=fc_item,  # type: ignore[arg-type]
                )
            )
            s.insert_item(fc_item, previous_item_id=chain_after or "root")
            await self._send_conversation_item_added_and_done(fc_item, chain_after)
            chain_after = fc_item.id
            function_call_items.append(fc_item)

        usage.total_tokens = usage.input_tokens + usage.output_tokens

        status_details = None
        if cancelled:
            status_details = {
                "type": "cancelled",
                "reason": "client_cancelled",
            }

        output_items: list[Any] = [] if drop_message_item else [item_obj]
        output_items.extend(function_call_items)

        done_response = self._response_object(
            response_id,
            response,
            status,
            status_details=status_details,
            output=output_items,
            usage={
                "total_tokens": usage.total_tokens,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )

        await self._send_event(
            types.ResponseDoneEvent(
                event_id=_gen_id("evt"),
                type="response.done",
                response=done_response,
            )
        )

    # ------------------------------------------------------------------ #
    #  response.cancel                                                    #
    # ------------------------------------------------------------------ #

    async def _handle_response_cancel(self, event: types.ResponseCancelEvent):
        active = self.session.active_response
        if active is None:
            await self._send_error(
                "No response is in progress",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return
        response_id = getattr(event, "response_id", None)
        if response_id is not None and response_id != active.response_id:
            await self._send_error(
                f"Response '{response_id}' is not in progress",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return
        await self._cancel_active_response()

    async def _cancel_active_response(self) -> None:
        active = self.session.active_response
        if active is None:
            return
        self._response_cancel_event.set()
        # engine.abort() tears down orchestrator/stage-pool bookkeeping for
        # the request, but never pushes a completion sentinel through the
        # per-request output queue -- so it does NOT by itself unblock
        # _run_response_inner's `async for output in gen:` loop, which would
        # otherwise wait on that queue forever. Cancelling the task is what
        # actually stops it (injects CancelledError at the current await),
        # which _run_response_inner catches to still emit response.done.
        try:
            await self.engine.abort(active.request_id)
        except Exception:
            logger.exception("Failed to abort request %s", active.request_id)
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------ #
    #  conversation.item.create                                           #
    # ------------------------------------------------------------------ #

    async def _handle_item_create(self, event: types.ConversationItemCreateEvent):
        item = event.item

        if (getattr(item, "type", None) or "").startswith("mcp_"):
            await self._send_unsupported_mcp(event.event_id)
            return

        try:
            self._validate_input_item(item)
            s = self.session
            pos = s.insert_item(item, event.previous_item_id)
        except ValueError as e:
            await self._send_error(str(e), "invalid_request_error", event_id=event.event_id)
            return
        prev_id = s.items[pos - 1].id if pos > 0 else None

        await self._send_conversation_item_added_and_done(item, prev_id)

    def _validate_input_item(self, item: Any) -> None:
        for part in getattr(item, "content", None) or []:
            part_type = getattr(part, "type", None)
            if part_type == "input_image":
                raise ValueError("Image input is not supported")
            if part_type == "input_audio":
                audio = getattr(part, "audio", None)
                if audio:
                    self._decode_pcm16(audio)
            if part_type == "output_audio" and getattr(part, "audio", None):
                raise ValueError("Client-provided assistant audio is not supported")

    # ------------------------------------------------------------------ #
    #  conversation.item.delete                                           #
    # ------------------------------------------------------------------ #

    async def _handle_item_delete(self, event: types.ConversationItemDeleteEvent):
        item_id = event.item_id

        # Per spec: "Send this event when you want to remove any item from
        # the conversation history" -- no positional restriction, the only
        # failure case is the item not existing.
        removed = self.session.remove_item(item_id)
        if removed is None:
            await self._send_error(
                f"Item '{item_id}' not found",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        await self._send_event(
            types.ConversationItemDeletedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.deleted",
                item_id=item_id,
            )
        )

    # ------------------------------------------------------------------ #
    #  conversation.item.retrieve                                         #
    # ------------------------------------------------------------------ #

    async def _handle_item_retrieve(self, event: types.ConversationItemRetrieveEvent):
        item_id = event.item_id

        item = self.session.find_item(item_id)
        if item is None:
            await self._send_error(
                f"Item '{item_id}' not found",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        await self._send_json(
            {
                "event_id": _gen_id("evt"),
                "type": "conversation.item.retrieved",
                "item": item.model_dump(exclude_none=True),
            }
        )

    # ------------------------------------------------------------------ #
    #  conversation.item.truncate                                         #
    # ------------------------------------------------------------------ #

    async def _handle_item_truncate(self, event: types.ConversationItemTruncateEvent):
        item_id = event.item_id
        s = self.session

        item = s.find_item(item_id)
        if item is None:
            await self._send_error(
                f"Item '{item_id}' not found",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        if not isinstance(item, types.RealtimeConversationItemAssistantMessage):
            await self._send_error(
                "Can only truncate assistant messages",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        # Record intent unconditionally. If the item's own response is still
        # generating (item_in_progress), item_duration_ms/item_token_ids
        # don't exist yet and the placeholder's content is still empty --
        # applying the truncation now would be a silent no-op, and
        # _run_response_inner's own finalization write would clobber it
        # afterward regardless (blank-on-cancel, or the full untruncated
        # content if the response isn't cancelled and keeps generating past
        # this point). Deferring to finalization -- the one place
        # item_token_ids is finally complete -- closes that race regardless
        # of which order the two arrive in; see _run_response_inner's own
        # check of this dict at finalization time.
        s.pending_truncations_ms[item_id] = event.audio_end_ms

        if not s.item_in_progress.get(item_id, False):
            await self._do_item_truncate(event)

    async def _do_item_truncate(self, event: types.ConversationItemTruncateEvent) -> None:
        """Apply a conversation.item.truncate once the target item is no
        longer in progress -- called either directly from
        _handle_item_truncate (item already finalized) or from
        _run_response_inner's finalization (item was still in progress when
        the truncate first arrived)."""
        s = self.session
        item_id = event.item_id
        content_index = event.content_index
        # A second truncate may have arrived (and overwritten
        # pending_truncations_ms) while the first was still waiting on this
        # same in-progress item -- always honor the latest recorded value
        # rather than the one on the event that happened to trigger this call.
        audio_end_ms = s.pending_truncations_ms.pop(item_id, event.audio_end_ms)

        item = s.find_item(item_id)
        if item is None:
            # Deleted/removed between the truncate request and its
            # resolution (e.g. conversation.item.delete raced ahead, or
            # drop_message_item removed it in _run_response_inner) --
            # nothing left to truncate.
            return
        if not isinstance(item, types.RealtimeConversationItemAssistantMessage):
            return

        duration_ms = s.item_duration_ms.get(item_id)
        if duration_ms is not None and audio_end_ms > duration_ms:
            await self._send_error(
                f"audio_end_ms ({audio_end_ms}) is greater than the actual audio duration",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        # Per spec, truncating audio must not leave text in context the user
        # never heard -- but rather than blanking the transcript entirely
        # (the previous, spec-minimum behavior), reconstruct the prefix that
        # *was* actually heard using Qwen3-Omni's fixed talker text/frame
        # correlation (see _qwen3_omni_truncate_transcript). This keeps the
        # model's own memory of what it said in sync with what the user
        # actually heard, instead of wiping it and confusing later turns.
        # Falls back to "" (today's behavior) when we don't have a captured
        # token stream for this item (e.g. it included a tool call).
        truncated_text = self._qwen3_omni_truncate_transcript(item_id, audio_end_ms)
        new_content = list(item.content)
        if not 0 <= content_index < len(new_content):
            # Previously fell through unmodified and still sent
            # conversation.item.truncated below, claiming success on a
            # no-op -- report the bad index instead of silently keeping the
            # full, untruncated content while telling the client otherwise.
            await self._send_error(
                f"content_index {content_index} is out of range for item '{item_id}'",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return
        part = new_content[content_index]
        if hasattr(part, "transcript"):
            new_content[content_index] = {
                "type": "output_audio",
                "transcript": truncated_text,
            }
        elif hasattr(part, "text"):
            new_content[content_index] = {"type": "output_text", "text": truncated_text}
        truncated_item = types.RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            id=item_id,
            status=item.status,
            content=new_content,  # type: ignore[arg-type]
        )
        s.replace_item(truncated_item)

        await self._send_event(
            types.ConversationItemTruncatedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.truncated",
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
        )

    # ------------------------------------------------------------------ #
    #  History -> prompt serialization                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert_tools(tools: list, *, strict: bool = False) -> list[dict]:
        """RealtimeFunctionTool is flat (type/name/description/parameters);
        apply_chat_template(tools=...) and ChatCompletionToolsParam both
        expect the chat-completions nested shape. MCP tools
        (RealtimeResponseCreateMcpTool) are not handled -- skipped.

        strict=True marks every tool as OpenAI chat-completions "strict"
        function calling -- RealtimeFunctionTool has no such field of its
        own, but vLLM's structural-tag builder
        (get_model_structural_tag/_any_tool_strict) only builds a
        tool_choice="auto" grammar when at least one tool is strict, so
        callers building guided-decoding tools need this set. Left False
        for the copy handed to apply_chat_template, which shouldn't show
        clients an OpenAI-specific field their own tool declaration never
        asked for.
        """
        converted = []
        for tool in tools:
            if getattr(tool, "type", None) != "function":
                continue
            function: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            if strict:
                function["strict"] = True
            converted.append({"type": "function", "function": function})
        return converted

    def _assistant_item_text(self, item: types.RealtimeConversationItemAssistantMessage) -> str:
        """Return the assistant item's stored transcript or text."""
        for part in item.content:
            text = getattr(part, "transcript", None) or getattr(part, "text", None)
            if text:
                return text
        return ""

    def _resolve_response_input(self, items: list[Any]) -> list[Any]:
        resolved = []
        for item in items:
            if getattr(item, "type", None) != "item_reference":
                self._validate_input_item(item)
                resolved.append(item)
                continue
            item_id = getattr(item, "id", None)
            referenced = self.session.find_item(item_id) if item_id else None
            if referenced is None:
                raise ValueError(f"Item '{item_id}' not found")
            self._validate_input_item(referenced)
            resolved.append(referenced)
        return resolved

    def _qwen3_omni_truncate_transcript(self, item_id: str, audio_end_ms: float) -> str:
        """Approximate the transcript heard before ``audio_end_ms``."""
        token_ids = self.session.item_token_ids.get(item_id)
        if not token_ids:
            return ""
        tokens_heard = round(audio_end_ms / QWEN3_OMNI_MS_PER_TOKEN)
        tokens_heard = min(tokens_heard, len(token_ids))
        if tokens_heard <= 0:
            return ""
        raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        return raw_tok.decode(token_ids[:tokens_heard], skip_special_tokens=True)

    @staticmethod
    def _qwen3_omni_audio_token_count(num_samples: int, sample_rate: int) -> int:
        """Return Qwen3-Omni's expanded token count for an audio segment."""
        from vllm.model_executor.models.qwen3_omni_moe_thinker import (
            _get_feat_extract_output_lengths,
        )

        num_samples_16k = round(num_samples * 16000 / sample_rate)
        input_lengths = num_samples_16k // 160
        output_lengths = _get_feat_extract_output_lengths(input_lengths)
        return int(output_lengths) + 2  # + <|audio_start|> + <|audio_end|>

    async def _build_full_prompt(
        self,
        tools: list | None = None,
        *,
        instructions: str | None = None,
        items: list | None = None,
    ) -> TokensPrompt:
        """Render the effective instructions and items into an engine prompt."""
        from vllm.inputs import TokensPrompt

        s = self.session
        effective_instructions = instructions if instructions is not None else s.config.instructions
        effective_items = items if items is not None else s.items
        messages: list[dict[str, Any]] = []
        audio_arrays: list[tuple[np.ndarray, int]] = []
        converted_tools = self._convert_tools(tools) if tools else None

        if effective_instructions:
            messages.append({"role": "system", "content": effective_instructions})

        for item in effective_items:
            if item.type == "function_call":
                # "" not None: this model's chat template only handles
                # message.content as a string or a list (line 55/59 of its
                # Jinja template unconditionally iterates non-string content
                # assuming it's a list) -- None crashes with "'NoneType'
                # object is not iterable" (confirmed in production logs).
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.call_id,
                                "type": "function",
                                "function": {"name": item.name, "arguments": item.arguments},
                            }
                        ],
                    }
                )
                continue
            if item.type == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item.call_id, "content": item.output})
                continue

            role = getattr(item, "role", None)
            if role == "user":
                parts_text = []
                for part in item.content:
                    if part.type == "input_audio" and part.audio:
                        parts_text.append(AUDIO_PLACEHOLDER)
                        audio_bytes = self._decode_pcm16(part.audio, MAX_INPUT_AUDIO_BUFFER_BYTES)
                        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_arrays.append((pcm16.astype(np.float32) / 32768.0, SAMPLE_RATE_HZ))
                    elif part.type == "input_text" and part.text:
                        parts_text.append(part.text)
                if parts_text:
                    messages.append({"role": "user", "content": "".join(parts_text)})
            elif role == "assistant":
                text = self._assistant_item_text(item)
                if text:
                    messages.append({"role": "assistant", "content": text})
            elif role == "system":
                text = "".join(part.text for part in item.content if part.type == "input_text" and part.text)
                if text:
                    messages.append({"role": "system", "content": text})

        chat_template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
        if converted_tools:
            chat_template_kwargs["tools"] = converted_tools
        text = self._tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        token_ids = raw_tok.encode(text, add_special_tokens=False)

        prompt_data = TokensPrompt(prompt_token_ids=token_ids)
        if audio_arrays:
            prompt_data["multi_modal_data"] = {"audio": audio_arrays}
        return prompt_data

    # ------------------------------------------------------------------ #
    #  Audio output processing                                            #
    # ------------------------------------------------------------------ #

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

    def _extract_audio_deltas(self, output) -> list[np.ndarray]:
        """Return this engine step's new audio samples, as-is.

        Qwen3-Omni's code2wav already returns non-overlapping increments per
        step -- both chunked_decode and chunked_decode_streaming
        (qwen3_omni_code2wav.py) explicitly slice the left-context overlap
        *out* of their own output before returning it (see
        `wav_chunk[..., context_size * self.total_upsample:]` and the
        `start = left_context_size * total_upsample - tail` slice
        respectively), specifically so nothing downstream has to re-derive
        what's new. This used to also diff each array against a stored
        reference of the previous one (np.allclose with a loose tolerance)
        as a defensive guard -- removed because it was solving a problem
        the model already solves, and had a real failure mode: two
        independent (already-correct) increments landing within tolerance
        of each other by coincidence -- most likely on quiet/near-silent
        passages -- would make it wrongly treat the second one as
        "old prefix + new tail" and silently drop the portion it mistook
        for overlap, corrupting playback for the rest of that response.
        """
        from collections.abc import Mapping

        mm = getattr(output, "multimodal_output", None)
        if mm is None or not isinstance(mm, Mapping):
            return []

        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return []

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []

        if isinstance(raw_audio, (list, tuple)):
            if raw_audio:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.append(arr)
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.append(arr)
        return chunks

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode()

    # ------------------------------------------------------------------ #
    #  Server event emission                                              #
    # ------------------------------------------------------------------ #

    async def _send_conversation_item_added_and_done(self, item: Any, previous_item_id: str | None) -> None:
        try:
            item_data = self._dump_model(item)
        except Exception:
            logger.exception("[realtime] failed to serialize conversation item")
            self._connected = False
            return

        for event_type in ("conversation.item.added", "conversation.item.done"):
            await self._send_json(
                {
                    "event_id": _gen_id("evt"),
                    "type": event_type,
                    "previous_item_id": previous_item_id,
                    "item": item_data,
                }
            )

    async def _send_event(self, event) -> None:
        try:
            data = self._dump_model(event) if hasattr(event, "model_dump") else event
        except Exception:
            logger.exception("[realtime] failed to serialize %s", getattr(event, "type", None))
            self._connected = False
            return
        await self._send_payload(data, getattr(event, "type", None))

    @staticmethod
    def _dump_model(model: Any) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            return model.model_dump(mode="json", exclude_none=True)

    async def _send_json(self, payload: dict) -> None:
        await self._send_payload(payload, payload.get("type"))

    async def _send_payload(self, payload: Any, event_type: str | None) -> None:
        if not self._connected:
            return
        try:
            async with self._send_lock:
                await self.ws.send_text(json.dumps(payload))
        except Exception:
            logger.warning("[realtime] send failed, marking connection dead: %s", event_type, exc_info=True)
            self._connected = False

    async def _send_error(
        self,
        message: str,
        code: str = "server_error",
        event_id: str | None = None,
        *,
        error_type: str = "invalid_request_error",
    ) -> None:
        error_data: dict[str, Any] = {
            "type": error_type,
            "code": code,
            "message": message,
            "param": None,
            "event_id": event_id,
        }
        await self._send_event(
            types.RealtimeErrorEvent(
                event_id=_gen_id("evt"),
                type="error",
                error=error_data,  # type: ignore[arg-type]
            )
        )

    async def _send_session_created(self) -> None:
        session_obj = self._build_session_object()
        await self._send_event(
            types.SessionCreatedEvent(
                event_id=_gen_id("evt"),
                type="session.created",
                session=session_obj,  # type: ignore[arg-type]
            )
        )

    async def _send_session_updated(self) -> None:
        session_obj = self._build_session_object()
        await self._send_event(
            types.SessionUpdatedEvent(
                event_id=_gen_id("evt"),
                type="session.updated",
                session=session_obj,  # type: ignore[arg-type]
            )
        )

    async def _send_conversation_created(self) -> None:
        await self._send_event(
            types.ConversationCreatedEvent(
                event_id=_gen_id("evt"),
                type="conversation.created",
                conversation={  # type: ignore[arg-type]
                    "id": self.session.conversation_id,
                    "object": "realtime.conversation",
                },
            )
        )

    def _build_session_object(self) -> dict[str, Any]:
        s = self.session
        obj = s.config.model_dump(exclude_none=True)
        obj["object"] = "realtime.session"
        obj["id"] = s.session_id
        obj["expires_at"] = int(s.expires_at)
        return obj
