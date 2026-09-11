# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""DuplexOmni: the Python API for full-duplex models.

Sessions run inside the engine (``DuplexSessionRunner`` on the orchestrator
loop of ``DuplexOrchestrator``). This class opens / resumes / closes them and
pipes typed ``DuplexCommand`` objects in and typed ``DuplexEvent`` objects out
through a ``DuplexSessionHandle``. It holds no session state beyond the
handle registry, so the websocket handler and ``InlineDuplexClient`` are both
thin consumers of the same surface.

Example::

    omni = DuplexOmni(model="openbmb/MiniCPM-o-4_5", trust_remote_code=True)
    async with await omni.open_session({"modalities": ["audio", "text"]}) as session:
        async def consume():
            async for event in session.events():
                if isinstance(event, AudioDelta):
                    play(event.audio)
        task = asyncio.create_task(consume())
        await session.submit(AppendAudio(audio=pcm_bytes, format="pcm16", sample_rate_hz=16000))
        await session.submit(Commit())
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from uuid import uuid4

from vllm.logger import init_logger

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands as duplex_commands
from vllm_omni.engine.duplex.commands import DuplexCommand
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig, ResponseCreateOptions
from vllm_omni.engine.duplex.events import DuplexEvent, SessionClosed
from vllm_omni.engine.duplex.messages import (
    DuplexControlResultMessage,
    DuplexSessionError,
    DuplexSessionEventMessage,
)
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.entrypoints.async_omni_base import AsyncOmniBase

logger = init_logger(__name__)

_DEFAULT_CONTROL_TIMEOUT_S = 10.0


class DuplexSessionHandle:
    """Client-side view of one engine-resident duplex session.

    Connection-independent: a websocket may detach and a later one resume the
    same handle. ``events()`` is single-consumer at any given time but may be
    re-entered after the previous iterator was closed (resume).
    """

    def __init__(self, omni: DuplexOmni, session_id: str) -> None:
        self._omni = omni
        #: Allocated by ``DuplexOmni.open_session``; unique for the engine's lifetime.
        self.session_id = session_id
        self.capabilities: DuplexCapabilities = DuplexCapabilities()
        self.public_session: dict[str, object] = {}
        self.lease_generation: int = 0
        self._outbox: asyncio.Queue[DuplexEvent | None] = asyncio.Queue()
        self._closed = False
        self._close_reason: str | None = None
        self._closed_event = asyncio.Event()
        self._consumer_active = False

    # ---- state ----

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_reason(self) -> str | None:
        return self._close_reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DuplexSessionHandle(session_id={self.session_id!r}, closed={self._closed})"

    # ---- input ----

    async def submit(self, command: DuplexCommand) -> None:
        """Enqueue one command in caller order; rejections arrive as ``ErrorEvent`` on ``events()``."""
        if self._closed:
            raise DuplexSessionError(
                f"duplex session {self.session_id} is closed", code="session_closed", session_id=self.session_id
            )
        await self._omni.engine.submit_command_async(self.session_id, command)

    async def append_audio(
        self,
        audio: bytes,
        *,
        format: str = "pcm16",
        sample_rate_hz: int | None = None,
        is_speech: bool | None = None,
        video_frames: Sequence[str] | None = None,
        duration_ms: int | None = None,
        audio_end_ms: int | None = None,
        hints: Mapping[str, object] | None = None,
    ) -> None:
        await self.submit(
            duplex_commands.AppendAudio(
                audio=audio,
                format=format,
                sample_rate_hz=sample_rate_hz,
                is_speech=is_speech,
                video_frames=tuple(video_frames or ()),
                duration_ms=duration_ms,
                audio_end_ms=audio_end_ms,
                hints=dict(hints or {}),
            )
        )

    async def append_text(self, text: str) -> None:
        await self.submit(duplex_commands.AppendText(text=text))

    async def commit(
        self,
        *,
        final: bool = True,
        create_response: bool | None = None,
        is_speech: bool | None = None,
    ) -> None:
        await self.submit(duplex_commands.Commit(final=final, create_response=create_response, is_speech=is_speech))

    async def create_response(self, options: ResponseCreateOptions | Mapping[str, object] | None = None) -> None:
        if isinstance(options, ResponseCreateOptions):
            payload: dict[str, object] = {
                key: value
                for key, value in (
                    ("instructions", options.instructions),
                    ("voice", options.voice),
                    ("output_audio_format", options.response_format),
                    ("temperature", options.temperature),
                    ("max_output_tokens", options.max_tokens),
                    ("speed", options.speed),
                    ("modalities", list(options.modalities) if options.modalities is not None else None),
                )
                if value is not None
            }
            if options.extra_body:
                payload["extra_body"] = dict(options.extra_body)
        else:
            payload = dict(options or {})
        await self.submit(duplex_commands.CreateResponse(options=payload))

    async def clear_input(self) -> None:
        await self.submit(duplex_commands.ClearInput())

    async def cancel_input(self) -> None:
        await self.submit(duplex_commands.CancelInput())

    async def cancel_response(self, response_id: str | None = None) -> None:
        await self.submit(duplex_commands.CancelResponse(response_id=response_id))

    async def barge_in(self) -> None:
        await self.submit(duplex_commands.BargeIn())

    async def clear_output_audio(self, response_id: str | None = None) -> None:
        await self.submit(duplex_commands.ClearOutputAudio(response_id=response_id))

    async def signal_turn(self, event: str, payload: Mapping[str, object] | None = None) -> None:
        await self.submit(duplex_commands.SignalTurn(event=event, signal_payload=dict(payload or {})))

    async def update(self, session_patch: Mapping[str, object]) -> None:
        await self.submit(duplex_commands.UpdateSession(patch=dict(session_patch)))

    async def ack_playback(
        self,
        played_ms: int,
        *,
        response_id: str | None = None,
        item_id: str | None = None,
        committed_ms: int | None = None,
    ) -> None:
        await self.submit(
            duplex_commands.AckPlayback(
                played_ms=int(played_ms),
                committed_ms=int(committed_ms) if committed_ms is not None else None,
                response_id=response_id,
                item_id=item_id,
            )
        )

    async def heartbeat(self) -> None:
        await self.submit(duplex_commands.Heartbeat())

    async def create_item(self, item: Mapping[str, object], *, previous_item_id: str | None = None) -> None:
        await self.submit(duplex_commands.CreateItem(item=dict(item), previous_item_id=previous_item_id))

    async def delete_item(self, item_id: str) -> None:
        await self.submit(duplex_commands.DeleteItem(item_id=item_id))

    async def truncate_item(self, item_id: str, *, audio_end_ms: int, content_index: int = 0) -> None:
        await self.submit(
            duplex_commands.TruncateItem(item_id=item_id, audio_end_ms=int(audio_end_ms), content_index=content_index)
        )

    async def close(self, *, reason: str = "client_close", timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S) -> None:
        if self._closed:
            return
        await self._omni.close_session(self.session_id, reason=reason, timeout=timeout)

    # ---- output ----

    async def events(self) -> AsyncIterator[DuplexEvent]:
        """Ordered public events; ends after ``session.closed`` / ``session.expired``."""
        if self._consumer_active:
            raise RuntimeError(f"duplex session {self.session_id} already has an active events() consumer")
        self._consumer_active = True
        try:
            while True:
                if self._closed and self._outbox.empty():
                    return
                event = await self._outbox.get()
                if event is None:
                    if self._closed and self._outbox.empty():
                        return
                    continue
                yield event
                if event.is_terminal:
                    return
        finally:
            self._consumer_active = False

    async def wait_closed(self) -> str:
        await self._closed_event.wait()
        return self._close_reason or "closed"

    async def __aenter__(self) -> DuplexSessionHandle:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            await self.close()
        except DuplexSessionError:
            pass

    # ---- internals used by DuplexOmni ----

    def _adopt(self, result: DuplexControlResultMessage) -> None:
        if result.capabilities is not None:
            self.capabilities = result.capabilities
        if result.public_session:
            self.public_session = dict(result.public_session)
        if result.lease_generation is not None:
            self.lease_generation = int(result.lease_generation)

    def _deliver(self, event: DuplexEvent) -> None:
        self._outbox.put_nowait(event)
        if isinstance(event, SessionClosed):  # SessionExpired is a SessionClosed
            self._mark_closed(event.reason or event.type)

    def _mark_closed(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_reason = reason
        self._closed_event.set()
        self._outbox.put_nowait(None)


class DuplexOmni(AsyncOmniBase):
    """Async Python API for full-duplex models (see module docstring).

    Construct it like ``AsyncOmni``. The pipeline must declare
    ``duplex_plugin`` and the deploy config ``session_mode: duplex``; the
    engine raises at startup otherwise. There is no ``generate()``: duplex
    serving is session-only.
    """

    engine: DuplexOmniEngine

    def _create_engine(self, **engine_kwargs: Any) -> DuplexOmniEngine:
        from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio

        return DuplexOmniEngine(duplex_audio_encoder=encode_audio, **engine_kwargs)

    def __init__(self, model: str = "", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, model=model, **kwargs)
        self._handles: dict[str, DuplexSessionHandle] = {}

    # ---- deployment facts ----

    @property
    def duplex_session_config(self) -> DuplexSessionRuntimeConfig:
        return self.engine.duplex_session_config

    @property
    def duplex_capabilities(self) -> DuplexCapabilities:
        return self.engine.duplex_capabilities

    @property
    def sessions(self) -> Mapping[str, DuplexSessionHandle]:
        return dict(self._handles)

    def active_session_count(self) -> int:
        return sum(1 for handle in self._handles.values() if not handle.closed)

    # ---- session lifecycle ----

    @staticmethod
    def _resolve_session_config(
        config: DuplexSessionConfig | Mapping[str, object] | None,
        *,
        model: str,
    ) -> DuplexSessionConfig:
        """Build the normalized session config the engine will own (the one parse/normalize site).

        A ``session_id`` / ``id`` key inside a Realtime session object is
        ignored: Realtime clients echo the session object back, and the server
        allocates every session id.
        """
        if config is None:
            resolved = DuplexSessionConfig(model=model)
        elif isinstance(config, DuplexSessionConfig):
            resolved = config
        elif isinstance(config, Mapping):
            resolved = DuplexSessionConfig.from_realtime(config, model=model)
        else:
            raise TypeError(f"unsupported duplex session config: {type(config).__name__}")
        if resolved.model is None:
            resolved.model = model
        return resolved.normalized()

    async def open_session(
        self,
        config: DuplexSessionConfig | Mapping[str, object] | None = None,
        *,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexSessionHandle:
        """Open an engine-resident session and return its handle.

        The session id is always allocated here (``duplex-<uuid4 hex>``; never
        reused, so the id alone identifies a session) and the handle is
        registered before the open RPC, so the first event (``session.created``,
        which announces the id) can never arrive before it exists.
        """
        session_id = f"duplex-{uuid4().hex}"
        session_config = self._resolve_session_config(config, model=self.model)
        handle = DuplexSessionHandle(self, session_id)
        self._handles[session_id] = handle
        self._final_output_handler()
        try:
            result = await self.engine.open_session_async(session_id, session_config, timeout=timeout)
        except BaseException:
            if self._handles.get(session_id) is handle:
                self._handles.pop(session_id, None)
            raise
        handle._adopt(result)
        return handle

    def get_session(self, session_id: str) -> DuplexSessionHandle | None:
        return self._handles.get(session_id)

    async def resume_session(
        self,
        session_id: str,
        *,
        expected_lease_generation: int,
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> DuplexSessionHandle:
        """Engine lease resume (CAS on the lease generation); returns the existing handle."""
        handle = self._require_handle(session_id)
        result = await self.engine.resume_session_async(
            session_id,
            expected_lease_generation=expected_lease_generation,
            timeout=timeout,
        )
        # The result carries the engine's current public session (state,
        # epoch, turn ...), which a resumed client must see, not the open-time snapshot.
        handle._adopt(result)
        return handle

    async def touch_session(
        self,
        session_id: str,
        *,
        activity: str = "heartbeat",
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> None:
        self._require_handle(session_id)
        await self.engine.touch_session_async(session_id, activity=activity, timeout=timeout)

    async def detach_session(self, session_id: str, *, timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S) -> None:
        """Start the engine-owned disconnect grace; expiry arrives as ``session.expired``."""
        await self.touch_session(session_id, activity="detach", timeout=timeout)

    async def close_session(
        self,
        session_id: str,
        *,
        reason: str = "client_close",
        timeout: float | None = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> None:
        handle = self._handles.get(session_id)
        if handle is None or handle.closed:
            return
        try:
            await self.engine.close_session_async(session_id, reason=reason, timeout=timeout)
        except Exception:
            # The engine refused or lost the close: the handle is unusable
            # either way, and it must not shadow the id in ``sessions``.
            handle._mark_closed(reason)
            self._handles.pop(session_id, None)
            raise
        # The manager emits session.closed after the stage cleanup, but the
        # event travels on the output queue: wait for it so ``events()`` ends
        # with the typed SessionClosed rather than an abrupt stop.
        try:
            await asyncio.wait_for(handle.wait_closed(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[DuplexOmni] session.closed for %s did not arrive in time", session_id)
            handle._mark_closed(reason)
            self._handles.pop(session_id, None)

    async def close_all_sessions(self, *, reason: str = "shutdown") -> None:
        for session_id in list(self._handles):
            try:
                await self.close_session(session_id, reason=reason)
            except Exception:
                logger.exception("[DuplexOmni] failed to close session %s", session_id)

    def _require_handle(self, session_id: str) -> DuplexSessionHandle:
        handle = self._handles.get(session_id)
        if handle is None or handle.closed:
            raise DuplexSessionError(
                f"unknown or closed duplex session: {session_id}", code="unknown_session", session_id=session_id
            )
        return handle

    # ---- engine output routing ----

    def _route_engine_message(self, msg: object) -> bool:
        if not isinstance(msg, DuplexSessionEventMessage):
            return False
        handle = self._handles.get(msg.session_id)
        if handle is None:
            logger.debug("[DuplexOmni] dropping event for unknown session %s", msg.session_id)
            return True
        handle._deliver(msg.event)
        if handle.closed:
            self._handles.pop(msg.session_id, None)
        return True

    def _on_engine_dead(self, error: str) -> None:
        for handle in list(self._handles.values()):
            handle._mark_closed(f"engine_dead: {error}")
        self._handles.clear()

    def shutdown(self, timeout: float | None = None) -> None:
        for handle in list(self._handles.values()):
            handle._mark_closed("shutdown")
        self._handles.clear()
        super().shutdown(timeout)


__all__ = ["DuplexOmni", "DuplexSessionHandle"]
