# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import binascii
import io
import json
import warnings
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64 as base64
from fastapi import WebSocket, WebSocketDisconnect
from openai.types import realtime as types
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from pydantic import TypeAdapter
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams
from vllm.tool_parsers import ToolParserManager

if TYPE_CHECKING:
    from vllm.inputs import EngineInput

from vllm_omni.engine.duplex.turn_detection import (
    ServerTurnDetector,
    ServerVADUnavailableError,
    SileroVADBackendProvider,
    TurnDetectionConfig,
    TurnDetectionResult,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime.session import (
    ActiveResponse,
    AudioFullDuplexSessionState,
    HistoryLimitError,
    ResponseUsage,
    _gen_id,
    merge_session_config,
)
from vllm_omni.utils.audio import audio_chunk_sample_rate
from vllm_omni.utils.audio_resample import StreamingAudioResampler

logger = init_logger(__name__)

_CLIENT_EVENT_ADAPTER = TypeAdapter(types.RealtimeClientEvent)

# The Realtime PCM contract is mono signed PCM16 at 24 kHz.
SAMPLE_RATE_HZ = 24000
BYTES_PER_SAMPLE_PCM16 = 2
# Application safety caps for one append and the complete pending input turn.
MAX_AUDIO_APPEND_BYTES = 15 * 1024 * 1024
MAX_INPUT_AUDIO_BUFFER_BYTES = 64 * 1024 * 1024

AUTO_TRUNCATION_TRIGGER_RATIO = 0.8
AUTO_TRUNCATION_TARGET_RATIO = 0.5


def _is_prompt_limit_error(exc: VLLMValidationError) -> bool:
    return exc.parameter in {"input_text", "input_tokens"}


class _UnsupportedAudioFormatError(ValueError):
    """The Realtime endpoint only accepts mono PCM16 audio at 24 kHz."""


@dataclass(slots=True)
class _ResolvedResponse:
    input: list[types.ConversationItem] | None
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
        chat_handler: Any,
        tool_call_parser: str | None = None,
        enable_auto_tool_choice: bool = False,
        vad_backend_provider: SileroVADBackendProvider | None = None,
    ):
        self.ws = websocket
        self.engine = engine
        self.model_name = model_name
        self.chat_handler = chat_handler
        self._tokenizer = chat_handler.renderer.get_tokenizer()
        self._tool_call_parser_name = tool_call_parser if enable_auto_tool_choice else None
        self._vad_backend_provider = vad_backend_provider
        self._turn_detector: ServerTurnDetector | None = None
        self._speech_item_id: str | None = None

        self.session = AudioFullDuplexSessionState()
        self.session.config.model = model_name

        self._connected = True
        self._response_task: asyncio.Task | None = None
        self._response_cancel_event = asyncio.Event()
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def handle_connection(self):
        try:
            await self.ws.accept()
            logger.info("[realtime] connection opened, session_id=%s", self.session.session_id)
            await self._send_session_created()
            await self._send_conversation_created()

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

    # ------------------------------------------------------------------ #
    #  Event dispatch                                                     #
    # ------------------------------------------------------------------ #

    async def _dispatch_event(self, event: types.RealtimeClientEvent):
        match event:
            case types.SessionUpdateEvent():
                handler = self._handle_session_update
            case types.InputAudioBufferAppendEvent():
                handler = self._handle_audio_append
            case types.InputAudioBufferCommitEvent():
                handler = self._handle_audio_commit
            case types.InputAudioBufferClearEvent():
                handler = self._handle_audio_clear
            case types.ResponseCreateEvent():
                handler = self._handle_response_create
            case types.ResponseCancelEvent():
                handler = self._handle_response_cancel
            case types.ConversationItemCreateEvent():
                handler = self._handle_item_create
            case types.ConversationItemDeleteEvent():
                handler = self._handle_item_delete
            case types.ConversationItemRetrieveEvent():
                handler = self._handle_item_retrieve
            case types.ConversationItemTruncateEvent():
                handler = self._handle_item_truncate
            case _:
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
        candidate_config = merge_session_config(s.config, event.session)
        try:
            candidate_config = self._sanitize_session_config(candidate_config, s.config)
        except _UnsupportedAudioFormatError as exc:
            await self._send_error(str(exc), "unsupported_audio_format", event_id=event.event_id)
            return
        try:
            turn_detector = self._build_turn_detector(candidate_config)
        except ValueError as exc:
            await self._send_error(str(exc), "unsupported_turn_detection", event_id=event.event_id)
            return

        s.config = candidate_config
        if turn_detector is None:
            self._reset_turn_detection()
        self._turn_detector = turn_detector
        await self._send_session_updated()

    def _sanitize_session_config(
        self,
        config: types.RealtimeSessionCreateRequest,
        previous: types.RealtimeSessionCreateRequest,
    ) -> types.RealtimeSessionCreateRequest:
        """Normalize supported session settings while keeping Pydantic models intact."""
        sanitized = config.model_copy(deep=True)

        if sanitized.model != self.model_name:
            sanitized.model = previous.model

        if sanitized.tools is not None:
            sanitized.tools = [tool for tool in sanitized.tools if getattr(tool, "type", None) != "mcp"]
        if getattr(sanitized.tool_choice, "type", None) == "mcp":
            sanitized.tool_choice = previous.tool_choice

        audio = sanitized.audio
        if audio is None:
            return sanitized

        audio_input = audio.input
        if audio_input is not None:
            # These options are accepted for compatibility but are not implemented.
            audio_input.transcription = None
            audio_input.noise_reduction = None
            turn_detection = audio_input.turn_detection
            if isinstance(turn_detection, ServerVad):
                # The OpenAI SDK model allows extra keys; omit them from effective config.
                extra_fields = turn_detection.model_fields_set - ServerVad.model_fields.keys()
                for field_name in extra_fields:
                    delattr(turn_detection, field_name)
            if not self._is_pcm16_24khz_format(audio_input.format):
                raise _UnsupportedAudioFormatError("Only 24 kHz PCM16 input audio is supported")

        audio_output = audio.output
        if audio_output is not None:
            if audio_output.speed not in (None, 1):
                previous_output = previous.audio.output if previous.audio is not None else None
                audio_output.speed = previous_output.speed if previous_output is not None else None
            if not self._is_pcm16_24khz_format(audio_output.format):
                raise _UnsupportedAudioFormatError("Only 24 kHz PCM16 output audio is supported")

        return sanitized

    def _build_turn_detector(self, config: types.RealtimeSessionCreateRequest) -> ServerTurnDetector | None:
        audio_input = config.audio.input if config.audio is not None else None
        turn_detection = audio_input.turn_detection if audio_input is not None else None
        if turn_detection is None:
            return None
        if not isinstance(turn_detection, ServerVad):
            raise ValueError("Only server_vad turn detection is supported")

        detector_config = TurnDetectionConfig(
            threshold=0.5 if turn_detection.threshold is None else float(turn_detection.threshold),
            prefix_padding_ms=(
                300 if turn_detection.prefix_padding_ms is None else int(turn_detection.prefix_padding_ms)
            ),
            silence_duration_ms=(
                500 if turn_detection.silence_duration_ms is None else int(turn_detection.silence_duration_ms)
            ),
            create_response=True if turn_detection.create_response is None else turn_detection.create_response,
            interrupt_response=(
                True if turn_detection.interrupt_response is None else turn_detection.interrupt_response
            ),
        )
        return detector_config.build_detector(self._vad_backend_provider)

    def _reset_turn_detection(self) -> None:
        if self._turn_detector is not None:
            self._turn_detector.reset()
        self._speech_item_id = None

    @staticmethod
    def _unsupported_audio_option(audio: Any) -> str | None:
        if audio is None:
            return None
        audio_input = getattr(audio, "input", None)
        if audio_input is not None:
            fields: set[str] = getattr(audio_input, "model_fields_set", set())
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
    def _is_pcm16_24khz_format(audio_format: Any) -> bool:
        if audio_format is None:
            return True
        if isinstance(audio_format, str):
            return audio_format in ("audio/pcm", "pcm16")
        format_type = getattr(audio_format, "type", None)
        rate = getattr(audio_format, "rate", SAMPLE_RATE_HZ)
        return format_type in ("audio/pcm", "pcm16") and rate == SAMPLE_RATE_HZ

    @staticmethod
    def _uses_mcp(config: Any) -> bool:
        return (
            any(getattr(tool, "type", None) == "mcp" for tool in (getattr(config, "tools", None) or []))
            or getattr(getattr(config, "tool_choice", None), "type", None) == "mcp"
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

        detector = self._turn_detector
        result: TurnDetectionResult | None = None
        if detector is not None:
            try:
                result = await asyncio.to_thread(
                    detector.process,
                    event.audio,
                    fmt="pcm16",
                    sample_rate_hz=SAMPLE_RATE_HZ,
                )
            except ServerVADUnavailableError as exc:
                self._turn_detector = None
                self._speech_item_id = None
                await self._send_error(str(exc), "server_vad_unavailable", event_id=event.event_id)
                return
            except ValueError as exc:
                await self._send_error(str(exc), "bad_audio", event_id=event.event_id)
                return

        # Limitation: this buffer retains all appended audio until commit.
        # prefix_padding_ms only adjusts speech_started.audio_start_ms; it does
        # not trim earlier silence from the audio item.
        self.session.input_audio_buffer.extend(audio_bytes)
        if result is not None:
            await self._handle_turn_detection_result(result)

    async def _handle_turn_detection_result(self, result: TurnDetectionResult) -> None:
        detector = self._turn_detector
        assert detector is not None, "turn detection results require an active detector"

        if result.speech_started and result.speech_stopped and result.speech_active:
            # Coalesce a stop and restart detected within one append. From the
            # client's perspective, keep the current speech item open and let
            # a later standalone stop finish and commit it.
            if self._speech_item_id is None:
                if detector.config.interrupt_response:
                    await self._cancel_active_response()
            return

        if result.speech_started:
            if self._speech_item_id is None:
                self._speech_item_id = _gen_id("item")
            await self._send_event(
                types.InputAudioBufferSpeechStartedEvent(
                    event_id=_gen_id("evt"),
                    type="input_audio_buffer.speech_started",
                    audio_start_ms=result.audio_start_ms or 0,
                    item_id=self._speech_item_id,
                )
            )
            if detector.config.interrupt_response:
                await self._cancel_active_response()

        if not result.speech_stopped:
            return

        item_id = self._speech_item_id or _gen_id("item")
        await self._send_event(
            types.InputAudioBufferSpeechStoppedEvent(
                event_id=_gen_id("evt"),
                type="input_audio_buffer.speech_stopped",
                audio_end_ms=result.audio_end_ms or 0,
                item_id=item_id,
            )
        )
        if not result.should_commit:
            return

        item = await self._commit_audio_buffer(item_id=item_id)
        if item is None:
            return
        if result.create_response:
            if self.session.active_response is not None and not detector.config.interrupt_response:
                await self._send_error(
                    "Cannot create an automatic response while another response is still in progress",
                    "invalid_request_error",
                )
                return
            await self._handle_response_create(
                types.ResponseCreateEvent(
                    type="response.create",
                    event_id=None,
                    response=None,
                )
            )

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

    @staticmethod
    def _pcm16_wav_b64(audio: bytes) -> str:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(BYTES_PER_SAMPLE_PCM16)
                wav.setframerate(SAMPLE_RATE_HZ)
                wav.writeframes(audio)
            return base64.b64encode(buffer.getvalue()).decode("ascii")

    async def _commit_audio_buffer(
        self,
        event_id: str | None = None,
        item_id: str | None = None,
    ) -> types.RealtimeConversationItemUserMessage | None:
        """Commit buffered audio and announce the conversation item."""
        try:
            s = self.session
            if len(s.input_audio_buffer) == 0:
                return None
            audio = base64.b64encode(s.input_audio_buffer).decode("ascii")
            item = types.RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                status="completed",
                id=item_id,
                content=[{"type": "input_audio", "audio": audio}],
            )
            try:
                s.insert_item(item)
            except HistoryLimitError as exc:
                s.input_audio_buffer.clear()
                await self._send_error(str(exc), "invalid_request_error", event_id=event_id)
                return None
            s.input_audio_buffer.clear()
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
        finally:
            self._reset_turn_detection()

    async def _handle_audio_commit(self, event: types.InputAudioBufferCommitEvent):
        s = self.session
        if len(s.input_audio_buffer) == 0:
            await self._send_error(
                "Input audio buffer is empty",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        await self._commit_audio_buffer(event.event_id, item_id=self._speech_item_id)

    async def _handle_audio_clear(self, event: types.InputAudioBufferClearEvent):
        self.session.input_audio_buffer.clear()
        self._reset_turn_detection()
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
            if output is not None and not self._is_pcm16_24khz_format(getattr(output, "format", None)):
                raise _UnsupportedAudioFormatError("Only 24 kHz PCM16 output audio is supported")

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

    async def _truncate_prompt_items(self, response: _ResolvedResponse) -> list[types.ConversationItem] | None:
        s = self.session
        max_model_len = self.engine.model_config.max_model_len
        if response.input is None:
            items = s.model_context_items()
        else:
            items = response.input

        async def probe_prompt(
            current_items: list[types.ConversationItem],
        ) -> tuple[EngineInput | None, int]:
            # Budget checks are speculative and may render repeatedly while
            # truncating, so keep them off the shared sender cache.
            try:
                prompt = await self._build_full_prompt(
                    tools=response.tools,
                    instructions=response.instructions,
                    items=current_items,
                    skip_mm_cache=True,
                )
            except VLLMValidationError as exc:
                if not _is_prompt_limit_error(exc):
                    raise
                return None, max_model_len + 1
            return prompt, len(prompt["prompt_token_ids"])

        truncation = s.config.truncation or "auto"
        ratio = 1.0
        custom_limit = None
        if isinstance(truncation, str):
            mode = "disabled" if truncation == "disabled" else "auto"
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

        engine_input, total = await probe_prompt(items)
        if total <= trigger:
            return items
        if mode == "disabled":
            logger.warning(
                "[realtime] token budget exceeded (%d/%d) and truncation is disabled -- rejecting response.create",
                total,
                limit,
            )
            return None

        all_items = items
        if not all_items:
            return None if total > limit else all_items

        low = 1
        high = len(all_items)
        feasible_index: int | None = None
        while low < high:
            middle = (low + high) // 2
            items = all_items[middle:]
            engine_input, total = await probe_prompt(items)
            if total <= target:
                high = middle
                feasible_index = middle
            else:
                low = middle + 1

        if feasible_index != low:
            items = all_items[low:]
            engine_input, total = await probe_prompt(items)

        if total > limit:
            return None
        return items

    async def _handle_response_create(self, event: types.ResponseCreateEvent):
        s = self.session
        response_cfg = event.response

        if response_cfg is not None and self._uses_mcp(response_cfg):
            await self._send_unsupported_mcp(event.event_id)
            return

        try:
            response = self._resolve_response(response_cfg)
        except _UnsupportedAudioFormatError as exc:
            await self._send_error(str(exc), "unsupported_audio_format", event_id=event.event_id)
            return
        except ValueError as exc:
            await self._send_error(str(exc), "invalid_request_error", event_id=event.event_id)
            return

        had_active_response = s.active_response is not None
        try:
            preflight_items = await self._truncate_prompt_items(response)
        except VLLMValidationError as exc:
            await self._send_error(str(exc), "invalid_request_error", event_id=event.event_id)
            return
        if preflight_items is None:
            await self._send_error(
                "The response input exceeds the model's input token limit",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        if had_active_response:
            if s.active_response is not None:
                await self._cancel_active_response()
            try:
                preflight_items = await self._truncate_prompt_items(response)
            except VLLMValidationError as exc:
                await self._send_error(str(exc), "invalid_request_error", event_id=event.event_id)
                return
            if preflight_items is None:
                await self._send_error(
                    "The response input exceeds the model's input token limit",
                    "invalid_request_error",
                    event_id=event.event_id,
                )
                return

        try:
            engine_input = await self._build_full_prompt(
                tools=response.tools,
                instructions=response.instructions,
                items=preflight_items,
            )
        except VLLMValidationError as exc:
            await self._send_error(str(exc), "invalid_request_error", event_id=event.event_id)
            return
        if response.input is None:
            s.commit_model_context_items(preflight_items)

        response_id = _gen_id("resp")
        await self._send_event(
            types.ResponseCreatedEvent(
                event_id=_gen_id("evt"),
                type="response.created",
                response=self._response_object(response_id, response, "in_progress"),
            )
        )

        s.active_response = ActiveResponse(response_id=response_id, request_id=f"rt-{response_id}")

        self._response_cancel_event.clear()
        self._response_task = asyncio.create_task(self._run_response(response_id, response, engine_input))

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

    async def _run_response(
        self,
        response_id: str,
        response: _ResolvedResponse,
        engine_input: EngineInput,
    ):
        s = self.session
        active = s.active_response
        if active is None:
            return

        completed = False
        try:
            await self._run_response_inner(response_id, response, s, active, engine_input)
            completed = True
        except asyncio.CancelledError:
            if self._response_cancel_event.is_set():
                await self._finish_cancelled_response(s, response_id, response, active)
                completed = True
            else:
                raise
        except Exception:
            logger.exception("_run_response failed for %s", response_id)
        finally:
            if not completed:
                await self._fail_response(s, response_id, response, active.item_id)
            s.active_response = None

    def _mark_terminal_response(self, response_id: str) -> bool:
        active = self.session.active_response
        if active is None or active.response_id != response_id or active.terminal_event_sent:
            return False
        active.terminal_event_sent = True
        return True

    async def _finish_cancelled_response(
        self,
        s: AudioFullDuplexSessionState,
        response_id: str,
        response: _ResolvedResponse,
        active: ActiveResponse,
    ) -> None:
        if active.item_id is not None and s.item_in_progress.pop(active.item_id, False):
            s.pending_truncations_ms.pop(active.item_id, None)
            s.remove_item(active.item_id)
        if not self._mark_terminal_response(response_id):
            return
        await self._send_event(
            types.ResponseDoneEvent(
                event_id=_gen_id("evt"),
                type="response.done",
                response=self._response_object(
                    response_id,
                    response,
                    "cancelled",
                    status_details={"type": "cancelled", "reason": "client_cancelled"},
                ),
            )
        )

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
        if not self._mark_terminal_response(response_id):
            return
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

    async def _run_response_inner(self, response_id, response, s, active, engine_input: EngineInput):
        previous_item_id = s.items[-1].id if s.items else None
        modalities = response.modalities
        is_audio = "audio" in modalities
        tools, tool_choice = response.tools, response.tool_choice
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
            parser_request = ChatCompletionRequest(
                messages=[],
                tools=strict_tools,
                tool_choice=self._convert_tool_choice(tool_choice),
                include_reasoning=False,
                skip_special_tokens=True,
            )
            structure_tag = tool_parser.get_structural_tag(
                parser_request,
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
        limit_reached = False
        usage = ResponseUsage()
        total_audio_samples = 0
        audio_resampler: StreamingAudioResampler | None = None

        previous_text = ""
        previous_token_ids: list[int] = []
        pending_tool_calls: dict[int, dict[str, Any]] = {}
        next_output_index = 1  # 0 is the message item, reserved above
        # Tool-call markup is control output, so suppress its audio after detection.
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

        async def emit_audio_delta(chunk: np.ndarray) -> None:
            nonlocal total_audio_samples
            if not chunk.size:
                return
            # Honor the requested modality and suppress audio after a tool call.
            if not is_audio or tool_call_seen:
                return
            total_audio_samples += chunk.shape[0]
            await self._send_event(
                types.ResponseAudioDeltaEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_audio.delta",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                    delta=self._pcm16_b64(chunk),
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

        sampling_params_list = self.chat_handler._fix_minicpmo45_audio_stream_output_kinds(
            sampling_params_list, modalities
        )
        gen = self.engine.generate(
            prompt=engine_input,
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
                    if audio_chunks:
                        if audio_resampler is None:
                            audio_resampler = StreamingAudioResampler(
                                audio_chunk_sample_rate(output),
                                SAMPLE_RATE_HZ,
                            )
                        for chunk in audio_chunks:
                            await emit_audio_delta(audio_resampler.process(chunk))
                    continue

                if output.outputs:
                    first_out = output.outputs[0]
                    finish_reason = getattr(first_out, "finish_reason", None)
                    finish_reason_value = getattr(finish_reason, "value", finish_reason)
                    if finish_reason_value == "length" or str(finish_reason_value).lower() in {
                        "finishreason.length",
                        "length",
                    }:
                        limit_reached = True
                    delta_text = first_out.text or ""
                    delta_token_ids = list(first_out.token_ids)
                    usage.output_tokens += len(delta_token_ids)
                    # Keep the raw token stream for approximate transcript
                    # truncation, independent of any tool-parser stripping.
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
                                request=parser_request,
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

        if audio_resampler is not None and not cancelled:
            tail = audio_resampler.process(np.empty(0, dtype=np.float32), final=True)
            await emit_audio_delta(tail)

        status = "cancelled" if cancelled else "incomplete" if limit_reached else "completed"

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
            status="completed" if not cancelled and not limit_reached else "incomplete",
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
            pending_event = s.pending_truncations_ms.get(item_id)
            if drop_message_item:
                s.remove_item(item_id)
                if pending_event is not None:
                    await self._send_error(
                        f"Item '{item_id}' produced no message content to truncate",
                        "invalid_request_error",
                        event_id=pending_event.event_id,
                    )
            else:
                s.replace_item(history_item)
                if item_obj.id:
                    s.item_duration_ms[item_obj.id] = total_audio_samples / SAMPLE_RATE_HZ * 1000
                    # Skip storing for tool-call responses: full_token_ids is
                    # the raw thinker stream (tool-call tags included), but
                    # this item's transcript/text is tool-parser-stripped,
                    # so truncation falls back to a blank transcript rather
                    # than splice raw tool markup into it.
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
            if not drop_message_item and pending_event is not None:
                await self._do_item_truncate(pending_event)

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
                status="completed" if not cancelled and not limit_reached else "incomplete",
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
        elif limit_reached:
            status_details = {
                "type": "incomplete",
                "reason": "max_output_tokens",
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

        if not self._mark_terminal_response(response_id):
            return
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

    def _validate_input_item(self, item: types.ConversationItem) -> None:
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
        s.pending_truncations_ms[item_id] = event

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
        # A second truncate may have arrived (and overwritten
        # pending_truncations_ms) while the first was still waiting on this
        # same in-progress item -- always honor the latest recorded value
        # rather than the one on the event that happened to trigger this call.
        effective_event = s.pending_truncations_ms.pop(item_id, None) or event
        content_index = effective_event.content_index
        audio_end_ms = effective_event.audio_end_ms

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
                event_id=effective_event.event_id,
            )
            return

        new_content = list(item.content)
        if not 0 <= content_index < len(new_content):
            # Reject an invalid content index before changing the history.
            await self._send_error(
                f"content_index {content_index} is out of range for item '{item_id}'",
                "invalid_request_error",
                event_id=effective_event.event_id,
            )
            return

        # Preserve only the transcript prefix proportional to the audio heard.
        # Exact token/audio alignment is model-specific, so fall back to an
        # empty transcript when token IDs or duration are unavailable.
        truncated_text = self._truncate_transcript(item_id, audio_end_ms)
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

    @staticmethod
    def _convert_tool_choice(
        tool_choice: Any,
    ) -> str | ChatCompletionNamedToolChoiceParam:
        if isinstance(tool_choice, str):
            return tool_choice
        choice_type = getattr(tool_choice, "type", None)
        name = getattr(tool_choice, "name", None)
        if choice_type == "function" and name:
            return ChatCompletionNamedToolChoiceParam(
                type="function",
                function={"name": name},
            )
        return "auto"

    def _assistant_item_text(self, item: types.RealtimeConversationItemAssistantMessage) -> str:
        """Return the assistant item's stored transcript or text."""
        for part in item.content:
            text = getattr(part, "transcript", None) or getattr(part, "text", None)
            if text:
                return text
        return ""

    def _resolve_response_input(self, items: list[Any]) -> list[types.ConversationItem]:
        resolved: list[types.ConversationItem] = []
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

    def _truncate_transcript(self, item_id: str, audio_end_ms: float) -> str:
        """Estimate the transcript prefix heard before ``audio_end_ms``."""
        token_ids = self.session.item_token_ids.get(item_id)
        duration_ms = self.session.item_duration_ms.get(item_id)
        if not token_ids or duration_ms is None or duration_ms <= 0:
            return ""
        tokens_heard = int(len(token_ids) * audio_end_ms / duration_ms)
        if tokens_heard <= 0:
            return ""
        raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        return raw_tok.decode(token_ids[:tokens_heard], skip_special_tokens=True)

    async def _build_full_prompt(
        self,
        tools: list | None = None,
        *,
        instructions: str | None = None,
        items: list[types.ConversationItem] | None = None,
        skip_mm_cache: bool = False,
    ) -> EngineInput:
        """Render the effective conversation through normal chat preprocessing."""
        chat_handler = self.chat_handler

        s = self.session
        effective_instructions = instructions if instructions is not None else s.config.instructions
        effective_items = items if items is not None else s.items
        messages: list[dict[str, Any]] = []
        converted_tools = self._convert_tools(tools) if tools else None

        if effective_instructions:
            messages.append({"role": "system", "content": effective_instructions})

        for item in effective_items:
            if item.type == "function_call":
                # Some chat templates iterate message content unconditionally.
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
                content = []
                for part in item.content:
                    if part.type == "input_audio" and part.audio:
                        audio_bytes = self._decode_pcm16(part.audio, MAX_INPUT_AUDIO_BUFFER_BYTES)
                        content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": self._pcm16_wav_b64(audio_bytes),
                                    "format": "wav",
                                },
                            }
                        )
                    elif part.type == "input_text" and part.text:
                        content.append({"type": "text", "text": part.text})
                if content:
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                text = self._assistant_item_text(item)
                if text:
                    messages.append({"role": "assistant", "content": text})
            elif role == "system":
                text = "".join(part.text for part in item.content if part.type == "input_text" and part.text)
                if text:
                    messages.append({"role": "system", "content": text})

        request = ChatCompletionRequest(model=self.model_name, messages=messages)
        _, (engine_input,) = await chat_handler._preprocess_chat(
            request,
            messages,
            default_template=request.chat_template or chat_handler.chat_template,
            default_template_content_format=chat_handler.chat_template_content_format,
            default_template_kwargs=chat_handler._effective_chat_template_kwargs(request),
            tool_dicts=converted_tools,
            skip_mm_cache=skip_mm_cache,
        )
        if engine_input.get("prompt_token_ids") is None:
            raise RuntimeError("Realtime renderer did not return prompt token IDs")
        return engine_input

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
