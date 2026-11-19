# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client for the OpenAI Live ``/v1/live/sessions`` endpoint (native full duplex).

A Live session has no turns or responses: input audio streams in and the model
streams audio (and, for some models, transcript) back. Received events are
folded into an :class:`~vllm_omni.clients.duplex.EventCollector` under one
stream identity (the Live session id), so the duplex benchmark metrics read a
Live session the same way they read a Realtime one.
"""

from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlsplit, urlunsplit

import pybase64 as base64

from vllm_omni.clients.duplex import EventCollector

__all__ = [
    "LIVE_PATH",
    "LIVE_PCM_RATES_HZ",
    "LiveClient",
    "LiveSessionError",
    "build_live_url",
]

LIVE_PATH = "/v1/live/sessions"
LIVE_PCM_RATES_HZ = (16_000, 24_000)
_PCM16_BYTES_PER_SAMPLE = 2
_MAX_FRAME_BYTES = 64 * 1024 * 1024
# Live server events renamed to the Realtime event types EventCollector and the
# duplex metric/clock helpers read.
_COLLECTOR_EVENT_TYPES = {
    "session.output_audio.delta": "response.output_audio.delta",
    "session.output_transcript.delta": "response.output_audio_transcript.delta",
}


class LiveSessionError(Exception):
    """The server refused or ended the session before the client asked."""


def build_live_url(url: str) -> str:
    """Normalize an http(s)/ws(s) base or full URL to the Live WebSocket URL."""
    parts = urlsplit(url)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    if scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError(f"Unsupported Live URL: {url!r}")
    path = parts.path if parts.path not in {"", "/"} else LIVE_PATH
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


class LiveClient:
    """One Live session over WebSocket; use as ``async with``."""

    def __init__(
        self,
        url: str,
        *,
        sample_rate_hz: int = 16_000,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        if sample_rate_hz not in LIVE_PCM_RATES_HZ:
            raise ValueError(f"Live audio/pcm rate must be one of {LIVE_PCM_RATES_HZ}")
        self.url = build_live_url(url)
        self.sample_rate_hz = sample_rate_hz
        self.additional_headers = additional_headers
        self.events = EventCollector()
        self.events.output_sample_rate_hz = sample_rate_hz
        self.session_id: str | None = None
        self.close_code: int | None = None
        self.closed_by_server = False
        self._ws = None
        self._reader: asyncio.Task[None] | None = None
        self._closing = False
        self._started = asyncio.Event()
        self._closed = asyncio.Event()

    @property
    def stream_id(self) -> str:
        """The identity Live output carries in :attr:`events`."""
        return self.session_id or "live"

    async def __aenter__(self) -> LiveClient:
        import websockets

        self._ws = await websockets.connect(
            self.url, additional_headers=self.additional_headers, max_size=_MAX_FRAME_BYTES
        )
        self._reader = asyncio.create_task(self._read_events())
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self._closing = True
        if self._ws is not None:
            await self._ws.close()
        if self._reader is not None:
            await self._reader

    async def _read_events(self) -> None:
        from websockets.exceptions import ConnectionClosed

        assert self._ws is not None
        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                if isinstance(event, dict):
                    self._add(event)
        except ConnectionClosed:
            pass
        self.close_code = self._ws.close_code
        if not self._closing:
            self.closed_by_server = True
        self._started.set()
        self._closed.set()

    def _add(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        if event_type == "session.started":
            session = event.get("session")
            if isinstance(session, dict) and isinstance(session.get("id"), str):
                self.session_id = session["id"]
            self._started.set()
        elif event_type == "session.closed":
            self._closed.set()
        if isinstance(event_type, str) and event_type in _COLLECTOR_EVENT_TYPES:
            event = {**event, "type": _COLLECTOR_EVENT_TYPES[event_type], "response_id": self.stream_id}
            if "end_ms" in event:
                event["audio_end_ms"] = event["end_ms"]
        self.events.add(event)

    async def send(self, event: dict[str, object]) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(event))

    async def start(
        self,
        model: str,
        *,
        instructions: str | None = None,
        voice: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        audio: dict[str, object] = {"format": {"type": "audio/pcm", "rate": self.sample_rate_hz}}
        if voice is not None:
            audio["output"] = {"voice": voice}
        session: dict[str, object] = {"model": model, "audio": audio}
        if instructions is not None:
            session["instructions"] = instructions
        await self.send({"type": "session.start", "session": session})
        await asyncio.wait_for(self._started.wait(), timeout=timeout_s)
        if self.session_id is None:
            raise LiveSessionError(f"session.start was refused: {self.errors() or f'close code {self.close_code}'}")

    async def stream_pcm16(self, pcm16: bytes, *, chunk_ms: int, realtime: bool) -> None:
        """Append mono PCM16 at the session rate; stops early if the server ends the session."""
        bytes_per_s = self.sample_rate_hz * _PCM16_BYTES_PER_SAMPLE
        chunk_bytes = max(_PCM16_BYTES_PER_SAMPLE, bytes_per_s * chunk_ms // 1000 // 2 * 2)
        started_at = time.monotonic()
        sent = 0
        for offset in range(0, len(pcm16), chunk_bytes):
            if self.closed_by_server:
                return
            chunk = pcm16[offset : offset + chunk_bytes]
            await self.send({"type": "session.input_audio.append", "audio": base64.b64encode(chunk).decode("ascii")})
            sent += len(chunk)
            if realtime:
                await asyncio.sleep(max(0.0, started_at + sent / bytes_per_s - time.monotonic()))

    async def wait_quiet(self, *, settle_s: float, timeout_s: float) -> None:
        """Wait until no server event arrived for ``settle_s`` or the server ended the session."""
        deadline = time.monotonic() + timeout_s
        quiet_since = time.monotonic()
        while not self.closed_by_server:
            now = time.monotonic()
            received = self.events.event_received_at_s
            last = max(quiet_since, received[-1]) if received else quiet_since
            if now - last >= settle_s:
                return
            if now >= deadline:
                raise TimeoutError("Timed out waiting for Live output to settle")
            await asyncio.sleep(0.05)

    async def close(self, *, timeout_s: float = 20.0) -> None:
        if self.closed_by_server or self._closed.is_set():
            return
        self._closing = True
        await self.send({"type": "session.close"})
        await asyncio.wait_for(self._closed.wait(), timeout=timeout_s)

    def errors(self) -> list[dict[str, object]]:
        return self.events.errors()

    def audio_bytes(self) -> bytes:
        return self.events.audio_bytes(self.stream_id)
