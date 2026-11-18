# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reusable Realtime WebSocket, PCM, and event helpers for MiniCPM-o demos."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover - example dependency
    raise SystemExit("Install websockets first: pip install websockets") from exc

PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2

# Server-side model unit boundaries, in cumulative appended audio.
# Stage0 configures the streaming mel processor with first_chunk_ms=1035 and
# chunk_ms=1000; the processor aligns the first chunk down to a hop_length (160
# samples) multiple, so unit 0 closes at 16480 samples and every later unit
# closes 16000 samples after it. Camera frames must ride the append that closes
# a unit, otherwise Stage0 cannot bind them to that unit's audio.
DUPLEX_FIRST_UNIT_MS = 1030
DUPLEX_UNIT_MS = 1000


def duplex_unit_boundary_ms(unit_index: int) -> int:
    """Cumulative appended audio, in ms, that closes model unit ``unit_index``."""
    return DUPLEX_FIRST_UNIT_MS + max(0, int(unit_index)) * DUPLEX_UNIT_MS


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


def read_pcm16_wav(path: Path) -> bytes:
    """Read a mono, uncompressed, 16 kHz PCM16 WAV file."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("input WAV must be mono")
        if wav_file.getsampwidth() != PCM16_BYTES_PER_SAMPLE:
            raise ValueError("input WAV must be 16-bit PCM")
        if wav_file.getframerate() != PCM16_SAMPLE_RATE:
            raise ValueError("input WAV must be 16 kHz")
        if wav_file.getcomptype() != "NONE":
            raise ValueError("input WAV must be uncompressed PCM")
        return wav_file.readframes(wav_file.getnframes())


async def wait_for(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    label: str,
) -> None:
    """Wait for a collector predicate without coupling to a scenario runner."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


@dataclass
class RealtimeEventCollector:
    """Collect server events and decode response audio by response identity."""

    events: list[dict[str, object]] = field(default_factory=list)
    event_received_at_s: list[float] = field(default_factory=list)
    response_audio: dict[str, list[bytes]] = field(default_factory=dict)
    response_ids: list[str] = field(default_factory=list)
    output_sample_rate_hz: int = 24_000

    @staticmethod
    def response_id(event: dict[str, object]) -> str | None:
        response_id = event.get("response_id")
        if isinstance(response_id, str):
            return response_id
        response = event.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str):
                return response_id
        return None

    def add(self, event: dict[str, object], *, received_at_s: float | None = None) -> None:
        received_at = time.monotonic() if received_at_s is None else float(received_at_s)
        stored_event = dict(event)
        stored_event.setdefault("_client_received_at_s", received_at)
        self.events.append(stored_event)
        self.event_received_at_s.append(received_at)
        response_id = self.response_id(stored_event)
        event_type = stored_event.get("type")
        if event_type == "response.created" and response_id and response_id not in self.response_ids:
            self.response_ids.append(response_id)
        if event_type == "response.output_audio.delta":
            delta = stored_event.get("delta") or stored_event.get("audio")
            if isinstance(delta, str) and response_id:
                try:
                    self.response_audio.setdefault(response_id, []).append(base64.b64decode(delta))
                except ValueError:
                    pass
            sample_rate_hz = stored_event.get("sample_rate_hz")
            if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                self.output_sample_rate_hz = sample_rate_hz

    def count(self, event_type: str) -> int:
        return sum(event.get("type") == event_type for event in self.events)

    def audio_bytes(self, response_id: str | None = None) -> bytes:
        if response_id is not None:
            return b"".join(self.response_audio.get(response_id, ()))
        return b"".join(
            chunk for response_id in self.response_ids for chunk in self.response_audio.get(response_id, ())
        )

    def response_text(self, response_id: str) -> str:
        """Join all text/transcript deltas for one response identity."""
        return "".join(
            str(event.get("delta") or "")
            for event in self.events
            if self.response_id(event) == response_id
            and event.get("type")
            in {
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
                "response.text.delta",
            }
        )

    def response_is_done(self, response_id: str) -> bool:
        return any(
            event.get("type") == "response.done" and self.response_id(event) == response_id for event in self.events
        )

    def errors(self) -> list[dict[str, object]]:
        return [event for event in self.events if event.get("type") == "error"]

    def first_received_at(
        self,
        *event_types: str,
        after_s: float = 0.0,
    ) -> float | None:
        for event, received_at_s in zip(self.events, self.event_received_at_s, strict=True):
            if received_at_s >= after_s and event.get("type") in event_types:
                return received_at_s
        return None

    def last_received_at(self, event_type: str) -> float | None:
        for event, received_at_s in zip(
            reversed(self.events),
            reversed(self.event_received_at_s),
            strict=True,
        ):
            if event.get("type") == event_type:
                return received_at_s
        return None


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

    async def send(self, event: dict[str, object]) -> None:
        await self._ws.send(json.dumps(event))

    def raise_if_reader_stopped(self) -> None:
        """Fail a caller waiting on events after the WebSocket reader exits."""

        task = self._reader_task
        if task is None or not task.done():
            return
        if task.cancelled():
            raise ConnectionError("Realtime WebSocket reader was cancelled")
        error = task.exception()
        if error is not None:
            raise ConnectionError("Realtime WebSocket reader failed") from error
        raise ConnectionError("Realtime WebSocket closed before the requested event arrived")

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
    ) -> None:
        session_extra_body = dict(extra_body or {})
        session_extra_body.pop("native_duplex", None)
        session_extra_body.update(auto_response=auto_response, force_listen_count=0)
        session: dict[str, object] = {
            "model": model,
            "modalities": ["audio", "text"],
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
        await self.send({"type": "session.update", "session": session})

        def session_created() -> bool:
            if self.events.count("session.created") > 0:
                return True
            self.raise_if_reader_stopped()
            return False

        await wait_for(
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
    ) -> int:
        """Stream PCM16 units, optionally attaching camera frames to each unit.

        A unit's frame slot takes a single JPEG/PNG (raw bytes or base64) or a
        sequence of them, which the Realtime wire caps at two per append.
        Returns the number of appends that carried at least one frame.
        """
        audio_end_ms = 0
        frames_sent = 0
        for chunk, frame in units:
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
                event["video_frames"] = encoded_frames
                frames_sent += 1
            await self.send(event)
            if realtime:
                await asyncio.sleep(duration_ms / 1000)
        return frames_sent

    async def commit(self) -> None:
        await self.send({"type": "input_audio_buffer.commit", "final": True})

    async def acknowledge_playback(self) -> None:
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

        await wait_for(
            session_closed,
            timeout_s=timeout_s,
            label="session.closed",
        )
