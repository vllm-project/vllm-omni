# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime wire-protocol probes for benchmarks and server-VAD smoke tests.

Applications should use ``vllm_omni.clients.duplex.DuplexClient``. This probe
keeps its explicit handshake and media clock for reproducible workloads."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pybase64 as base64

from vllm_omni.clients.duplex import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    DuplexEvent,
    DuplexProtocolError,
    EventCollector,
    duplex_unit_boundary_ms,
    wait_for_condition,
)

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover - example dependency
    raise SystemExit("Install websockets first: pip install websockets") from exc

_BACKEND_UNAVAILABLE_CODES = frozenset({"server_vad_unavailable", "semantic_vad_unavailable"})


def build_realtime_url(
    url: str,
    model: str | None,
    *,
    autostart: bool | None = None,
) -> str:
    """Select the duplex route; new session IDs are assigned by the server."""
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"}:
        parts = parts._replace(scheme="ws" if parts.scheme == "http" else "wss")
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError(f"Unsupported Realtime URL: {url!r}")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.pop("native_duplex", None)
    query.pop("session_id", None)
    query["duplex"] = "1"
    if model:
        query["model"] = model
    if autostart is not None:
        query["autostart"] = "1" if autostart else "0"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _session_stats_error_message(event: Mapping[str, object]) -> str:
    error = event.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    message = event.get("message")
    if isinstance(message, str) and message:
        return message
    return "session.stats request failed"


class RealtimeEventCollector(EventCollector):
    """Probe event collector that remembers unavailable VAD backends."""

    def __init__(self) -> None:
        super().__init__()
        self._backend_error: dict[str, object] | None = None

    def add(self, event: dict[str, object] | DuplexEvent, *, received_at_s: float | None = None) -> None:
        super().add(event, received_at_s=received_at_s)
        event = self.events[-1]
        error = event.get("error")
        if (
            self._backend_error is None
            and event.get("type") == "error"
            and isinstance(error, dict)
            and isinstance(error.get("code"), str)
            and error.get("code") in _BACKEND_UNAVAILABLE_CODES
        ):
            self._backend_error = self.events[-1]


class RealtimeDuplexClient:
    """Small async client used by the user demo and reusable smoke probes."""

    def __init__(
        self,
        url: str,
        *,
        max_size: int = 64 * 1024 * 1024,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.max_size = max_size
        self.additional_headers = additional_headers
        self.events = RealtimeEventCollector()
        self._ws: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._media_clock_ms = 0

    async def __aenter__(self) -> RealtimeDuplexClient:
        self._ws = await websockets.connect(
            self.url,
            max_size=self.max_size,
            additional_headers=self.additional_headers,
        )
        self._reader_task = asyncio.create_task(self._read_events())
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    async def _read_events(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                if isinstance(event, dict):
                    event.setdefault("_media_clock_ms", self._media_clock_ms)
                    self.events.add(event)
        except ConnectionClosed:
            return

    async def get_session_stats(
        self,
        *,
        timeout_s: float = 10.0,
        event_id: str | None = None,
    ) -> dict[str, object]:
        """Fetch the server's per-turn stage work and audio (``session.stats``, RFC #8325)."""
        request_id = event_id or f"evt_{uuid4().hex}"
        from_index = len(self.events.events)
        deadline = time.monotonic() + timeout_s
        try:
            await asyncio.wait_for(
                self.send({"type": "session.stats.get", "event_id": request_id}),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            raise TimeoutError("Timed out sending session.stats.get") from exc

        def arrived() -> bool:
            for event in self.events.events[from_index:]:
                if event.get("type") == "session.stats" and event.get("request_event_id") == request_id:
                    return True
                if event.get("type") == "error" and (
                    event.get("event_id") == request_id or event.get("related_event_id") == request_id
                ):
                    return True
            self.raise_if_reader_stopped()
            return False

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out sending session.stats.get")
        await wait_for_condition(arrived, timeout_s=remaining, label="session.stats")
        for event in self.events.events[from_index:]:
            if event.get("type") == "session.stats" and event.get("request_event_id") == request_id:
                stats = event.get("stats")
                return stats if isinstance(stats, dict) else {}
            if event.get("type") == "error" and (
                event.get("event_id") == request_id or event.get("related_event_id") == request_id
            ):
                raise ConnectionError(_session_stats_error_message(event))
        return {}

    async def send(self, event: dict[str, object]) -> None:
        await self._ws.send(json.dumps(event))
        if event.get("type") != "input_audio_buffer.append":
            return
        event_id = event.get("event_id")
        self.events.note_client_append(
            {
                "sent_at_s": time.monotonic(),
                "event_id": event_id if isinstance(event_id, str) else "",
            }
        )

    def raise_if_reader_stopped(self) -> None:
        """Stop waiting after a transport failure or an unavailable backend."""

        event = self.events._backend_error
        if event is not None:
            self._raise_protocol_error(event)

        task = self._reader_task
        if task is None or not task.done():
            return
        if task.cancelled():
            raise ConnectionError("Realtime WebSocket reader was cancelled")
        error = task.exception()
        if error is not None:
            raise ConnectionError("Realtime WebSocket reader failed") from error
        raise ConnectionError("Realtime WebSocket closed before the requested event arrived")

    @staticmethod
    def _raise_protocol_error(event: dict[str, object]) -> None:
        error = event["error"]
        assert isinstance(error, dict)
        raise DuplexProtocolError(
            str(error.get("message") or error["code"]),
            code=str(error["code"]),
            event_id=str(event["event_id"]) if event.get("event_id") else None,
            raw=event,
        )

    async def configure(
        self,
        model: str,
        *,
        output_audio_format: str = "pcm16",
        ref_audio: str | None = None,
        instructions: str | None = None,
        initial_user_text: str | None = None,
        auto_response: bool = True,
        temperature: float | None = None,
        extra_body: dict[str, object] | None = None,
        turn_detection: dict[str, object] | None = None,
        idle_timeout_s: float | None = None,
        timeout_s: float = 20.0,
        modalities: Sequence[str] | None = ("audio", "text"),
    ) -> None:
        session_extra_body = dict(extra_body or {})
        session_extra_body.pop("native_duplex", None)
        session_extra_body.update(auto_response=auto_response, force_listen_count=0)
        session: dict[str, object] = {
            "model": model,
            "input_audio_format": "pcm16",
            "output_audio_format": output_audio_format,
            "turn_detection": dict(turn_detection) if turn_detection is not None else None,
            "overlap_policy": (
                "barge_in_on_speech"
                if turn_detection is not None and turn_detection.get("interrupt_response", True) is True
                else "listen_only"
            ),
            "playback_commit_policy": "ack_only",
            "extra_body": session_extra_body,
        }
        if modalities is not None:
            session["modalities"] = list(modalities)
        if temperature is not None:
            session["temperature"] = float(temperature)
        if ref_audio is not None:
            session["ref_audio"] = ref_audio
        if instructions is not None:
            session["instructions"] = instructions
        if initial_user_text is not None:
            session_extra = session["extra_body"]
            assert isinstance(session_extra, dict)
            session_extra["duplex_initial_user_text"] = initial_user_text
        if idle_timeout_s is not None:
            session["idle_timeout_s"] = idle_timeout_s
        event_offset = len(self.events.events)
        await self.send({"type": "session.update", "session": session})

        def session_created() -> bool:
            # Admission errors arrive before the server closes the socket.
            # Preserve their code and retryability instead of replacing them
            # with the reader's generic connection-closed error.
            for event in self.events.events[event_offset:]:
                error = event.get("error")
                if event.get("type") == "error" and isinstance(error, dict) and isinstance(error.get("code"), str):
                    self._raise_protocol_error(event)
            self.raise_if_reader_stopped()
            if self.events.count("session.created") > 0:
                return True
            return False

        await wait_for_condition(
            session_created,
            timeout_s=timeout_s,
            label="session.created",
        )

    async def stream_pcm16(
        self,
        pcm16: bytes,
        *,
        chunk_ms: int = 200,
        realtime: bool = True,
        video_frames: Sequence[str] | None = None,
        stacked_video_frames: Sequence[str | None] | None = None,
    ) -> int:
        """Append PCM16 audio, optionally interleaving omni camera frames.

        ``video_frames`` holds base64 JPEG/PNG frames in capture order, one per
        second of the clip. Frame ``k`` rides the append that closes model unit
        ``k`` (see ``duplex_unit_boundary_ms``), which reproduces the official
        ``streaming_prefill(audio_waveform=<1 s>, frame_list=[frame])`` pairing:
        a second of audio and the picture captured during it enter the same
        unit. Sending on whole-second boundaries instead would strand frame 0 on
        an append that cannot close a unit yet, and shift every later frame one
        unit ahead of its audio.

        ``stacked_video_frames`` is the optional parallel track of composites
        built by ``video_stacking.concat_frames``: entry ``k`` tiles the
        sub-frames captured *inside* unit ``k``, and rides the same append right
        after the base frame, giving the official ``frame_list=[base,
        composite]``. The audio is untouched — a unit stays one second however
        many sub-frames the composite carries. ``None`` entries send the base
        frame alone.

        Returns the number of base frames actually sent (a clip shorter than the
        frame list leaves the tail unsent).
        """
        chunk_bytes = max(
            PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000,
            PCM16_BYTES_PER_SAMPLE,
        )
        frames = list(video_frames or [])
        stacked = list(stacked_video_frames or [])

        def units() -> Iterator[tuple[bytes, list[str] | None]]:
            audio_end_ms = 0
            frames_sent = 0
            for offset in range(0, len(pcm16), chunk_bytes):
                chunk = pcm16[offset : offset + chunk_bytes]
                audio_end_ms += len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
                if not frames or audio_end_ms < duplex_unit_boundary_ms(frames_sent):
                    yield chunk, None
                    continue
                # A video advances one frame per unit and holds its last frame
                # when the audio outlives the clip; a still image is a
                # one-element list and therefore repeats.
                index = min(frames_sent, len(frames) - 1)
                composite = stacked[index] if index < len(stacked) else None
                frames_sent += 1
                yield chunk, [frames[index]] if composite is None else [frames[index], composite]

        return await self.stream_av_units(units(), realtime=realtime)

    async def stream_av_units(
        self,
        units: Any,
        *,
        realtime: bool = True,
        frame_sink: Callable[[list[str]], Awaitable[object]] | None = None,
    ) -> int:
        """Stream PCM16 units, optionally attaching camera frames to each unit.

        A unit's frame slot takes a single JPEG/PNG (raw bytes or base64) or a
        sequence of them, which the Realtime wire caps at two per append.
        ``frame_sink`` receives a unit's frames instead of the append, just
        before it is sent, for sessions that take images as conversation items.
        Returns the number of appends that carried at least one frame.
        """
        audio_end_ms = 0
        frames_sent = 0
        started_at = time.monotonic()
        for chunk, frame in units:
            self.raise_if_reader_stopped()
            if not chunk:
                continue
            duration_ms = len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
            audio_end_ms += duration_ms
            self._media_clock_ms = audio_end_ms
            event: dict[str, object] = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
                "input_audio_format": "pcm16",
                "sample_rate_hz": PCM16_SAMPLE_RATE,
                "duration_ms": duration_ms,
                "audio_end_ms": audio_end_ms,
            }
            encoded_frames = [
                base64.b64encode(item).decode("ascii") if isinstance(item, bytes | bytearray) else item
                for item in (frame if isinstance(frame, list | tuple) else [frame])
                if item is not None
            ]
            if encoded_frames:
                if frame_sink is not None:
                    await frame_sink(encoded_frames)
                else:
                    event["video_frames"] = encoded_frames
                frames_sent += 1
            await self.send(event)
            if realtime:
                # Encoding, frame delivery and send time consume this unit's
                # budget; sleeping a full unit after them accumulates drift.
                await asyncio.sleep(max(0.0, started_at + audio_end_ms / 1000 - time.monotonic()))
        return frames_sent

    async def append_silence(self, duration_ms: int) -> None:
        """Append one zeroed PCM16 unit, e.g. drain padding after a commit.

        The same ``input_audio_buffer.append`` event shape ``stream_av_units``
        sends — ``duration_ms``/``audio_end_ms`` included, ``_media_clock_ms``
        advanced — so the server's step loop keeps stepping on it as if the
        clip were still streaming.
        """
        duration_ms = max(0, int(duration_ms))
        if duration_ms == 0:
            return
        self._media_clock_ms += duration_ms
        silence = bytes(round(PCM16_SAMPLE_RATE * duration_ms / 1000) * PCM16_BYTES_PER_SAMPLE)
        await self.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(silence).decode("ascii"),
                "input_audio_format": "pcm16",
                "sample_rate_hz": PCM16_SAMPLE_RATE,
                "duration_ms": duration_ms,
                "audio_end_ms": self._media_clock_ms,
            }
        )

    async def commit(self) -> None:
        await self.send({"type": "input_audio_buffer.commit", "final": True})

    async def acknowledge_playback(self) -> None:
        session = next((e.get("session") for e in self.events.events if e.get("type") == "session.created"), None)
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        if isinstance(capabilities, dict) and capabilities.get("supports_playback_ack") is False:
            return
        for response_id in self.events.response_ids:
            pcm16 = self.events.audio_bytes(response_id)
            if not pcm16 and self.events.response_is_done(response_id):
                continue
            played_ms = len(pcm16) * 1000 // (self.events.output_sample_rate_hz * PCM16_BYTES_PER_SAMPLE)
            await self.send_playback_ack(response_id, played_ms)

    async def send_playback_ack(self, response_id: str, played_ms: int) -> None:
        await self.send(
            {
                "type": "playback.ack",
                "response_id": response_id,
                "item_id": f"item_{response_id}",
                "played_ms": played_ms,
                "committed_ms": played_ms,
            }
        )

    async def close_session(self, *, timeout_s: float = 20.0) -> None:
        from_index = len(self.events.events)
        await self.send({"type": "session.close"})

        def session_closed() -> bool:
            if any(event.get("type") == "session.closed" for event in self.events.events[from_index:]):
                return True
            self.raise_if_reader_stopped()
            return False

        await wait_for_condition(
            session_closed,
            timeout_s=timeout_s,
            label="session.closed",
        )
