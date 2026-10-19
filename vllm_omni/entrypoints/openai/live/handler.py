# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``LiveSessionHandler``: one OpenAI Live session over a WebSocket.

Native full-duplex models only: the model decides, unit by unit, when to
listen and when to speak. Input audio is cut into units by the model's
processor and appended to one resumable request, one segment at a time (each
update re-emits the previous segment's last sampled token).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from vllm.logger import init_logger

from vllm_omni.config.live_session import LiveSessionConfig, LiveSessionDeployConfig
from vllm_omni.entrypoints.openai.live import protocol as proto
from vllm_omni.entrypoints.openai.live.audio import (
    InputAudioDecoder,
    OutputAudioEncoder,
    ResampleStream,
    parse_audio_format,
    resample,
)
from vllm_omni.entrypoints.openai.live.driver import ResumableRequestDriver, finish_reason_str
from vllm_omni.entrypoints.openai.live.processor import LiveSessionProcessor, ProcessorContext
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    LiveSessionState,
    NativeUnitItem,
    SystemItem,
    UserTextItem,
)
from vllm_omni.metrics.live import LiveSessionMetrics

logger = init_logger(__name__)

USAGE_INTERVAL_MS = 60_000
# Units queued behind the open segment before the session is reported as
# falling behind real time.
BACKLOG_WARNING_UNITS = 3


class SessionSlots:
    """Live sessions admitted on this API server, bounded by ``max_sessions``."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.limit is not None and self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)


# OmniARScheduler finishes a streaming request whose next update would not fit
# max_model_len; AsyncOmni.generate raises it as a RuntimeError with this prefix.
CONTEXT_OVERFLOW_PREFIX = "context_length_exceeded"


def is_context_overflow(exc: BaseException) -> bool:
    return str(exc).startswith(CONTEXT_OVERFLOW_PREFIX)


class LiveSessionHandler:
    def __init__(
        self,
        websocket: WebSocket,
        *,
        engine: Any,
        model_name: str,
        live_config: LiveSessionConfig,
        deploy_config: LiveSessionDeployConfig,
        processor_cls: type[LiveSessionProcessor],
        processor_context: ProcessorContext,
        slots: SessionSlots | None = None,
        metrics: LiveSessionMetrics | None = None,
    ) -> None:
        self.ws = websocket
        self.engine = engine
        self.model_name = model_name
        self.live_config = live_config
        self.deploy = deploy_config
        self.processor_cls = processor_cls
        self.processor_context = processor_context
        self.slots = slots or SessionSlots(None)
        self.metrics = metrics or LiveSessionMetrics(model_name, log_stats=False)

        self.state: LiveSessionState | None = None
        self.processor: LiveSessionProcessor | None = None
        self.driver: ResumableRequestDriver | None = None

        self._send_lock = asyncio.Lock()
        self._ws_open = False
        self._closing = False
        self._holds_slot = False
        self._recv_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._decoder: InputAudioDecoder | None = None
        self._encoder: OutputAudioEncoder | None = None
        self._to_model: ResampleStream | None = None
        self._next_usage_ms = USAGE_INTERVAL_MS

        # Units ready for the model, and the unit whose segment is open.
        self._pending: deque[NativeUnitItem] = deque()
        self._inflight: NativeUnitItem | None = None
        self._backlog_warned = 0
        # The request's prefix segment is open (``render_prefix``); after it,
        # either the next unit or the resubmitted history follows.
        self._prefix_open = False
        self._resume_after_prefix = False
        self._after_prefix = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        await self.ws.accept()
        self._ws_open = True
        self._recv_task = asyncio.current_task()
        try:
            while not self._closing:
                try:
                    text = await self.ws.receive_text()
                except WebSocketDisconnect:
                    self._ws_open = False
                    await self._close("connection_lost", send_closed=False)
                    return
                await self._dispatch(text)
        except asyncio.CancelledError:
            if not self._closing:
                raise
        except Exception as exc:
            logger.exception("Live session %s failed", self._session_id)
            await self._fail(exc)
        finally:
            await self._close("connection_lost", send_closed=False)

    @property
    def _session_id(self) -> str:
        return self.state.id if self.state is not None else "-"

    def _spawn(self, coro: Any, name: str) -> asyncio.Task:
        task = asyncio.create_task(self._guard(coro), name=f"live-{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _guard(self, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                logger.exception("Live session %s background task failed", self._session_id)
                await self._fail(exc)

    async def _fail(self, exc: BaseException, reason: str = "internal") -> None:
        """Generic ``internal_server_error``, cleanup, close with 1011."""
        if self._closing:
            return
        self.metrics.error(reason)
        await self._send_error(proto.internal_error())
        await self._close(None, send_closed=False, code=proto.CLOSE_INTERNAL_ERROR)

    async def _close(
        self,
        reason: str | None,
        *,
        client_event_id: str | None = None,
        send_closed: bool = True,
        code: int = proto.CLOSE_NORMAL,
    ) -> None:
        """Run the session cleanup once; optionally reply ``session.closed``."""
        if self._closing:
            return
        self._closing = True
        self.metrics.session_finished()
        if self._holds_slot:
            self._holds_slot = False
            self.slots.release()
        if reason is not None:
            logger.info("Live session %s closing: %s", self._session_id, reason)
        if self.driver is not None:
            await self.driver.close()
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()
        if send_closed and self.state is not None and reason is not None:
            await self._send(proto.usage_updated(self.state.usage_seconds))
            await self._send(
                proto.session_closed(self.state.to_resource(), reason, self.state.usage_seconds, client_event_id)
            )
        if self._ws_open:
            self._ws_open = False
            with contextlib.suppress(Exception):
                await self.ws.close(code=code)
        if self._recv_task is not None and self._recv_task is not current and not self._recv_task.done():
            self._recv_task.cancel()

    # ------------------------------------------------------------------ #
    # Wire                                                               #
    # ------------------------------------------------------------------ #

    async def _send(self, event: BaseModel) -> None:
        if not self._ws_open:
            return
        payload = proto.dump_event(event)
        try:
            async with self._send_lock:
                await self.ws.send_text(payload)
        except Exception:
            self._ws_open = False
            if not self._closing:
                self._spawn(self._close("connection_lost", send_closed=False), "drop")

    async def _send_error(self, err: LiveProtocolError, client_event_id: str | None = None) -> None:
        await self._send(proto.error_event(err, client_event_id))

    async def _dispatch(self, text: str) -> None:
        client_event_id = None
        try:
            payload = proto.parse_client_json(text)
            client_event_id = payload.get("event_id")
            event_type = payload["type"]
            if event_type in proto.SERVER_IGNORED_EVENTS:
                return
            if (
                event_type in proto.SERVER_UNSUPPORTED_EVENTS
                or event_type in self.live_config.unsupported_features.events
            ):
                raise LiveProtocolError(
                    "unsupported_event", f"'{event_type}' is not supported by this server or model."
                )
            if event_type not in proto.CLIENT_EVENT_MODELS:
                raise LiveProtocolError("unsupported_event", f"Unsupported event type '{event_type}'.")
            if event_type == "session.start":
                if self.state is not None:
                    raise LiveProtocolError("session_already_started", "The session has already started.")
                await self.handle_session_start(payload, client_event_id)
                return
            if self.state is None:
                raise LiveProtocolError("session_not_started", "Send session.start before other events.")
            await self._handlers[event_type](self, payload, client_event_id)
        except LiveProtocolError as err:
            await self._send_error(err, client_event_id)

    # ------------------------------------------------------------------ #
    # session.start                                                      #
    # ------------------------------------------------------------------ #

    async def handle_session_start(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        if not self.slots.acquire():
            logger.warning("Live session.start refused: %d/%s sessions active", self.slots.active, self.slots.limit)
            self.metrics.session_rejected()
            await self._send_error(proto.session_limit_reached(), client_event_id)
            await self._close(None, send_closed=False, code=proto.CLOSE_TRY_AGAIN_LATER)
            return
        self._holds_slot = True

        try:
            state, processor = self._validate_session_start(payload)
            self._seed_history(state)
            processor.configure(state)
        except LiveProtocolError as err:
            logger.warning("Live session.start rejected: %s (%s, param=%s)", err.message, err.code, err.param)
            await self._send_error(err, client_event_id)
            await self._close(None, send_closed=False, code=proto.CLOSE_POLICY_VIOLATION)
            return
        self.state, self.processor = state, processor

        self._decoder = InputAudioDecoder(state.audio_format)
        self._encoder = OutputAudioEncoder(state.audio_format, processor.output_sample_rate)
        self._to_model = ResampleStream(state.audio_format.rate, processor.input_sample_rate)
        self.driver = ResumableRequestDriver(
            self.engine,
            state.id,
            processor.sampling_params_list(self.engine.default_sampling_params_list),
            on_output=self._on_engine_output,
            on_error=self._on_engine_error,
        )
        self._spawn(self._expiry_loop(), "expiry")
        self.metrics.session_started()
        await self._send(proto.session_started(state.to_resource(), client_event_id))
        await self._send(proto.info("capabilities", json.dumps(self._capabilities())))

    def _capabilities(self) -> dict[str, Any]:
        unsupported = self.live_config.unsupported_features
        return {
            "vad": "native",
            "unsupported_features": {
                "events": sorted({*proto.SERVER_UNSUPPORTED_EVENTS, *unsupported.events}),
                "session_configs": sorted({*proto.SERVER_UNSUPPORTED_SESSION_CONFIGS, *unsupported.session_configs}),
            },
        }

    @staticmethod
    def _lookup(raw: Mapping[str, Any], path: str) -> Any:
        value: Any = raw
        for part in path.split("."):
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value

    def _check_unsupported_configs(self, raw: Mapping[str, Any]) -> None:
        if raw.get("store") not in (None, False):
            raise proto.unsupported_session_config("store")
        for path in ("client", *self.live_config.unsupported_features.session_configs):
            value = self._lookup(raw, path)
            if value is not None and value != [] and value != {}:
                raise proto.unsupported_session_config(path)

    def _validate_delegation(self, delegation: Any) -> dict[str, Any] | None:
        """Accept a ``responses`` delegation that delegates nothing."""
        if delegation is None:
            return None
        if not isinstance(delegation, Mapping) or delegation.get("type") != "responses":
            raise proto.unsupported_session_config("delegation")
        responses = delegation.get("responses") or {}
        if not isinstance(responses, Mapping):
            raise proto.invalid_value("'delegation.responses' must be an object.", "delegation.responses")
        for key, value in responses.items():
            if value is None or (key == "tools" and value == []):
                continue
            if key == "model":
                if value != self.model_name:
                    raise proto.invalid_value(
                        f"delegation.responses.model must be the served model '{self.model_name}'.",
                        "delegation.responses.model",
                    )
                continue
            raise proto.unsupported_session_config(f"delegation.responses.{key}")
        return {"type": "responses", "responses": {"model": self.model_name}}

    def _validate_session_start(self, payload: dict[str, Any]) -> tuple[LiveSessionState, LiveSessionProcessor]:
        raw = payload.get("session")
        if not isinstance(raw, Mapping):
            raise proto.invalid_value("'session' must be an object.", "session")
        self._check_unsupported_configs(raw)
        delegation = self._validate_delegation(raw.get("delegation"))
        audio = raw.get("audio") or {}
        if not isinstance(audio, Mapping):
            raise proto.invalid_value("'session.audio' must be an object.", "audio")
        audio_format = parse_audio_format(audio.get("format"))
        event = proto.validate_client_event(payload)
        session = event.session  # type: ignore[attr-defined]
        if session.model != self.model_name:
            raise proto.invalid_value(f"Model '{session.model}' is not served here.", "model")

        processor = self.processor_cls(self.processor_context)
        output = audio.get("output")
        voice = output.get("voice") if isinstance(output, Mapping) else None
        voice_name = voice.get("id") if isinstance(voice, Mapping) else voice
        if isinstance(voice_name, str) and voice_name in proto.OPENAI_BUILT_IN_VOICES:
            raise proto.invalid_value(
                f"OpenAI voice '{voice_name}' is not available; use a voice id of the served model.",
                "audio.output.voice",
            )
        processor.resolve_voice(voice)

        instructions = raw.get("instructions")
        if instructions is not None and processor.count_tokens(instructions) > proto.MAX_INSTRUCTIONS_TOKENS:
            raise proto.invalid_value(f"'instructions' exceeds {proto.MAX_INSTRUCTIONS_TOKENS} tokens.", "instructions")
        items = raw.get("input")
        if items is not None:
            if len(items) > proto.MAX_INPUT_MESSAGES:
                raise proto.invalid_value(f"'input' exceeds {proto.MAX_INPUT_MESSAGES} messages.", "input")
            total = 0
            for index, item in enumerate(items):
                content = item.get("content") if isinstance(item, Mapping) else None
                if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0].get("text"), str):
                    raise proto.invalid_value("Each 'input' message must have exactly one text part.", f"input.{index}")
                total += processor.count_tokens(content[0]["text"])
            if total > proto.MAX_INPUT_TOKENS:
                raise proto.invalid_value(f"'input' exceeds {proto.MAX_INPUT_TOKENS} tokens.", "input")

        state = LiveSessionState(
            model=self.model_name,
            audio_format=audio_format,
            expires_at=int(time.time()) + self.deploy.session_lifetime_s,
            voice=voice,
            instructions=instructions,
            input=list(items) if items is not None else None,
            delegation=delegation,
        )
        return state, processor

    @staticmethod
    def _seed_history(state: LiveSessionState) -> None:
        if state.instructions and state.instructions.strip():
            state.history.append(SystemItem(state.instructions))
        for item in state.input or []:
            text = item["content"][0]["text"]
            role = item.get("role")
            if role == "developer":
                state.history.append(SystemItem(text))
            elif role == "user":
                state.history.append(UserTextItem(text))
            elif role == "assistant":
                state.history.append(AssistantItem(text))

    def _started(self) -> tuple[LiveSessionState, LiveSessionProcessor, ResumableRequestDriver]:
        """The session objects, which exist once session.start succeeded."""
        assert self.state is not None and self.processor is not None and self.driver is not None
        return self.state, self.processor, self.driver

    # ------------------------------------------------------------------ #
    # Client events                                                      #
    # ------------------------------------------------------------------ #

    async def handle_session_update(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        raw = payload.get("session")
        if not isinstance(raw, Mapping):
            raise proto.invalid_value("'session' must be an object.", "session")
        for key, value in raw.items():
            if key != "delegation" and value is not None:
                raise proto.unsupported_session_config(key)
        proto.validate_client_event(payload)
        if raw.get("delegation") is not None:
            self._validate_delegation(raw["delegation"])
        state, _, _ = self._started()
        await self._send(proto.session_updated(state.to_resource(), client_event_id))

    async def handle_input_audio_append(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        event = proto.validate_client_event(payload)
        state, processor, _ = self._started()
        assert self._decoder is not None and self._to_model is not None
        samples, duration_ms = self._decoder.decode(event.audio)  # type: ignore[attr-defined]
        start_ms = state.session_ms
        state.session_ms += duration_ms
        state.input_audio_ms += duration_ms
        self.metrics.input_audio(duration_ms / 1000.0)
        while state.session_ms >= self._next_usage_ms:
            self._next_usage_ms += USAGE_INTERVAL_MS
            await self._send(proto.usage_updated(state.usage_seconds))

        model_samples = self._to_model.process(samples)
        if state.muted:
            processor.push_silence(model_samples.size, start_ms)
        else:
            processor.push_audio(model_samples, start_ms)
        while (unit := processor.next_unit()) is not None:
            self._pending.append(unit)
        if len(self._pending) > max(BACKLOG_WARNING_UNITS, self._backlog_warned):
            self._backlog_warned = len(self._pending)
            logger.warning("Live session %s: model is %d units behind real time", state.id, len(self._pending))
        await self._pump()

    async def handle_input_audio_mute(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        self._started()[0].muted = True
        await self._send(proto.input_audio_muted(client_event_id))

    async def handle_input_audio_unmute(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        self._started()[0].muted = False
        await self._send(proto.input_audio_unmuted(client_event_id))

    async def handle_session_close(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        await self._close("close_requested", client_event_id=client_event_id)

    # ------------------------------------------------------------------ #
    # Units                                                              #
    # ------------------------------------------------------------------ #

    async def _open_request(self, *, resume: bool) -> bool:
        """Open a new request with the processor's prefix segment, if it has
        one. Returns whether the prefix is now open."""
        state, processor, driver = self._started()
        prefix = processor.render_prefix(state)
        if prefix is None:
            return False
        params = processor.prefix_sampling_params(state, driver.stage0_params)
        await driver.start(prefix, params)
        self._prefix_open = True
        self._resume_after_prefix = resume
        return True

    async def _pump(self) -> None:
        """Submit the next ready unit once the previous segment finished."""
        if self._inflight is not None or self._prefix_open or not self._pending or self._closing:
            return
        state, processor, driver = self._started()
        if not driver.active and await self._open_request(resume=False):
            return
        unit = self._pending.popleft()
        state.history.append(unit)
        self._inflight = unit
        params = processor.stage0_sampling_params(state, driver.stage0_params)
        if not driver.active:
            await driver.start(processor.render_unit(state, unit, first=True, last_sampled_token=None), params)
        elif self._after_prefix:
            self._after_prefix = False
            driver.append(processor.render_unit(state, unit, first=True, last_sampled_token=None), params)
        else:
            prompt = processor.render_unit(state, unit, first=False, last_sampled_token=driver.last_sampled_token)
            driver.append(prompt, params)

    async def _on_prefix_finished(self) -> None:
        self._prefix_open = False
        if self._resume_after_prefix:
            self._resume_after_prefix = False
            state, processor, driver = self._started()
            params = processor.stage0_sampling_params(state, driver.stage0_params)
            driver.append(processor.render_resume(state), params)
            return
        self._after_prefix = True
        await self._pump()

    async def _resubmit_after_overflow(self) -> None:
        """The context is full: trim the history and resubmit it, the unit
        whose append overflowed included, as a new request."""
        state, processor, driver = self._started()
        if self._inflight is None or len(state.units) <= 1:
            await self._fail(RuntimeError("context_length_exceeded on the session's first unit"), "context")
            return
        dropped = processor.trim_context(state)
        if not dropped:
            await self._fail(RuntimeError("context_length_exceeded with nothing left to trim"), "context")
            return
        self.metrics.context_trim()
        logger.info(
            "Live session %s: context full; dropped %d units, resubmitting %d",
            state.id,
            dropped,
            len(state.units),
        )
        self._after_prefix = False
        if await self._open_request(resume=True):
            return
        params = processor.stage0_sampling_params(state, driver.stage0_params)
        await driver.start(processor.render_resume(state), params)

    # ------------------------------------------------------------------ #
    # Engine output                                                      #
    # ------------------------------------------------------------------ #

    async def _on_engine_error(self, epoch: int, exc: BaseException) -> None:
        if self._closing or self.driver is None or epoch != self.driver.epoch:
            return
        if is_context_overflow(exc):
            await self._resubmit_after_overflow()
            return
        logger.error("Live session %s: engine error: %s", self._session_id, exc)
        await self._fail(exc)

    async def _on_engine_output(self, epoch: int, output: Any) -> None:
        if self._closing or self.driver is None or epoch != self.driver.epoch:
            return
        if getattr(output, "error", None):
            await self._fail(RuntimeError(f"engine error: {output.error}"))
            return
        state, processor, _ = self._started()
        unit = self._inflight or (state.units[-1] if state.units else None)
        start_ms, end_ms = (unit.start_ms, unit.end_ms) if unit is not None else (state.now_ms,) * 2
        if getattr(output, "stage_id", None) == 0 and getattr(output, "outputs", None):
            completion = output.outputs[0]
            if self._prefix_open:
                if finish_reason_str(getattr(completion, "finish_reason", None)) is not None:
                    await self._on_prefix_finished()
                return
            token_ids = list(getattr(completion, "token_ids", None) or ())
            processor.observe_stage0_output(state, token_ids, output)
            text = processor.spoken_text(token_ids, getattr(completion, "text", "") or "")
            if text:
                await self._send(proto.output_transcript_delta(text, start_ms, end_ms))
            if finish_reason_str(getattr(completion, "finish_reason", None)) is not None:
                self._inflight = None
                await self._pump()
            return
        stream_text = processor.observe_multimodal_output(state, output)
        if stream_text:
            await self._send(proto.output_transcript_delta(stream_text, start_ms, end_ms))
        if getattr(output, "final_output_type", None) != "audio":
            return
        chunks, sample_rate = self._audio_chunks(output)
        assert self._encoder is not None
        rate = processor.output_sample_rate
        for chunk in chunks:
            if sample_rate and sample_rate != rate:
                chunk = resample(chunk, sample_rate, rate)
            await self._send(proto.output_audio_delta(self._encoder.encode(chunk)))
            state.output_audio_ms += chunk.size * 1000.0 / rate
            self.metrics.output_audio(chunk.size / rate)

    @staticmethod
    def _audio_chunks(output: Any) -> tuple[list[np.ndarray], int | None]:
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, Mapping):
            return [], None
        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate")
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], None
        raw = mm.get(key)
        # Per-step increments; code2wav already slices its left context out.
        if isinstance(raw, (list, tuple)):
            values = [raw[-1]] if raw else []
        else:
            values = [raw]
        chunks = []
        for value in values:
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size:
                chunks.append(arr)
        return chunks, int(sr) if sr else None

    # ------------------------------------------------------------------ #
    # Timers                                                             #
    # ------------------------------------------------------------------ #

    async def _expiry_loop(self) -> None:
        delay = self._started()[0].expires_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        await self._close("expired")

    _handlers = {
        "session.update": handle_session_update,
        "session.input_audio.append": handle_input_audio_append,
        "session.input_audio.mute": handle_input_audio_mute,
        "session.input_audio.unmute": handle_input_audio_unmute,
        "session.close": handle_session_close,
    }
