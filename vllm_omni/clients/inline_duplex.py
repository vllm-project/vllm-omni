# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""In-process duplex client: the :class:`DuplexClient` usage over a ``DuplexOmni``.

Example::

    from vllm_omni.clients.inline_duplex import InlineDuplexClient
    from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    omni = DuplexOmni(model="openbmb/MiniCPM-o-4_5", trust_remote_code=True)
    cfg = create_duplex_session_config(ref_audio=audio_data_url(wav_bytes))
    async with InlineDuplexClient(omni, model="openbmb/MiniCPM-o-4_5", config=cfg) as client:
        await client.stream_pcm(pcm16)
        await client.commit()
        async for response in client.responses():
            async for chunk in response.audio():
                play(chunk)
            break

There is no transport to drop, so this client has no reconnect, resume,
heartbeat or event-acknowledgement machinery. The ``DuplexOmni`` instance is
passed in by the caller; this module never imports server runtime code at
import time.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from vllm_omni.clients.duplex import DuplexClientBase, DuplexConnectionError, SessionConfig

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.realtime_commands import RealtimeInputDefaults
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle

__all__ = ["InlineDuplexClient"]

#: Acknowledgements are a websocket-journal concern; heartbeats are lease touches and go through.
_TRANSPORT_ONLY_EVENTS = frozenset({"session.event_ack"})


class InlineDuplexClient(DuplexClientBase):
    """Async client for one duplex session on an in-process :class:`DuplexOmni`."""

    def __init__(
        self,
        omni: DuplexOmni,
        *,
        model: str,
        config: SessionConfig | None = None,
        handshake_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(model=model, config=config, handshake_timeout_s=handshake_timeout_s)
        self.omni = omni
        self._handle: DuplexSessionHandle | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._handshake_pending = False

    # -- transport hooks ------------------------------------------------------

    async def _open(self) -> None:
        # The handle's first event is ``session.created``; the pump delivers
        # it through ``_dispatch`` where the base handshake is waiting.
        self._handle = await self.omni.open_session(self.config.to_session_payload(model=self.model))
        # The base class follows up with the ``session.update`` handshake; the
        # open above already carried that session object, so it is not resent.
        self._handshake_pending = True
        self._pump_task = asyncio.create_task(self._pump_events(), name="inline-duplex-client-pump")

    def _input_defaults(self) -> RealtimeInputDefaults:
        """Audio format / sample-rate / VAD defaults declared by this session's object."""
        from vllm_omni.engine.duplex.realtime_commands import RealtimeInputDefaults

        session_payload = self.session_info or self.config.to_session_payload(model=self.model)
        return RealtimeInputDefaults().with_session_payload(session_payload)

    async def _send_command(self, payload: dict[str, object]) -> None:
        handle = self._handle
        if handle is None or handle.closed:
            raise DuplexConnectionError("send failed: session is not open")
        event_type = payload.get("type")
        if event_type == "session.update" and self._handshake_pending:
            self._handshake_pending = False
            return
        if event_type == "session.close":
            await handle.close()
            return
        if event_type in _TRANSPORT_ONLY_EVENTS:
            return
        # Lazy import keeps ``vllm_omni.clients`` free of runtime imports at
        # module load; the caller already holds a live DuplexOmni.
        from vllm_omni.engine.duplex.commands import DuplexCommandError, command_from_realtime
        from vllm_omni.engine.duplex.events import error_event

        try:
            command = command_from_realtime(payload, defaults=self._input_defaults())
        except DuplexCommandError as exc:
            await self._dispatch(
                error_event(exc.code, str(exc), event_id=exc.event_id or payload.get("event_id")).to_realtime()
            )
            return
        await handle.submit(command)

    async def _teardown(self) -> None:
        pump = self._pump_task
        if pump is not None and not pump.done():
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        handle = self._handle
        if handle is not None and not handle.closed:
            try:
                await handle.close()
            except Exception:  # noqa: BLE001
                pass

    # -- internals ---------------------------------------------------------------

    async def _pump_events(self) -> None:
        handle = self._handle
        assert handle is not None
        reason = "closed"
        try:
            async for event in handle.events():
                await self._dispatch(event.to_realtime())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            reason = f"session failed: {exc}"
            self._finalize(reason, expected=False)
            return
        self._finalize(reason, expected=True)
