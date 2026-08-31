# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reusable Realtime WebSocket, PCM, and event helpers for MiniCPM-o demos."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from vllm_omni.metrics.definitions import compute_audio_rtf

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


def _rounded_ms(value: float) -> float:
    return round(float(value), 3)


def _finite_number(value: object, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and (not nonnegative or parsed >= 0) else None


def _interval_summary(values: list[float]) -> dict[str, float | int]:
    clean = sorted(_rounded_ms(value) for value in values if math.isfinite(value) and value >= 0)
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(clean)) - 1)
        return clean[min(index, len(clean) - 1)]

    return {
        "count": len(clean),
        "mean": _rounded_ms(sum(clean) / len(clean)),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": clean[-1],
    }


def summarize_session_request_metrics(
    request_metrics: list[dict[str, object]],
    *,
    session_id: str | None,
) -> dict[str, object]:
    """Average client-observed metrics across turns that emitted audio."""

    def mean(metric: str, *, digits: int = 3) -> float | None:
        values = [value for request in request_metrics if (value := _finite_number(request.get(metric))) is not None]
        return round(sum(values) / len(values), digits) if values else None

    return {
        "session_id": session_id,
        "audio_turn_count": len(request_metrics),
        "mean_ttft_ms": mean("ttft_ms"),
        "mean_ttfp_ms": mean("ttfp_ms"),
        "mean_rtf": mean("rtf", digits=6),
    }


def _event_stage_metrics(event: dict[str, object]) -> dict[str, object] | None:
    candidates: list[object] = [event.get("vllm_omni")]
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend((metadata, metadata.get("vllm_omni")))
    response = event.get("response")
    if isinstance(response, dict):
        response_metadata = response.get("metadata")
        if isinstance(response_metadata, dict):
            candidates.extend((response_metadata, response_metadata.get("vllm_omni")))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        stage_metrics = candidate.get("stage_metrics")
        if isinstance(stage_metrics, dict):
            return stage_metrics
    return None


def build_realtime_url(
    url: str,
    model: str | None,
    *,
    autostart: bool | None = None,
    native_duplex: bool | None = True,
    session_id: str | None = None,
) -> str:
    """Add the explicit native-duplex query parameters to a Realtime URL."""
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"}:
        parts = parts._replace(scheme="ws" if parts.scheme == "http" else "wss")
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError(f"Unsupported Realtime URL: {url!r}")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["duplex"] = "1"
    if model:
        query["model"] = model
    if native_duplex is not None:
        query["minicpmo45_native_duplex"] = "1" if native_duplex else "0"
    if autostart is not None:
        query["autostart"] = "1" if autostart else "0"
    if session_id:
        query["session_id"] = session_id
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def reference_audio_data_url(path: str | None) -> str | None:
    """Encode a local reference WAV for a Realtime session update."""
    if path is None:
        return None
    audio = Path(path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {audio}")
    return "data:audio/wav;base64," + base64.b64encode(audio.read_bytes()).decode("ascii")


def chunk_period_ms(events: list[dict[str, object]], *, default: int = 1000) -> int:
    """Read the negotiated native-duplex model-unit duration."""
    for event in reversed(events):
        session = event.get("session")
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        value = capabilities.get("chunk_period_ms") if isinstance(capabilities, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def has_residual_model_unit(pcm16: bytes, *, chunk_period_ms: int) -> bool:
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_period_ms // 1000
    return bool(unit_bytes and len(pcm16) % unit_bytes)


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


def write_pcm16_wav(path: Path, pcm16: bytes, *, sample_rate_hz: int) -> None:
    """Write mono PCM16 bytes as a WAV artifact."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16)


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
        if event_type == "response.audio.delta":
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
                "response.audio_transcript.delta",
                "response.output_text.delta",
                "response.text.delta",
            }
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

    def timing_summary(
        self,
        *,
        after_s: float,
        input_committed_at_s: float | None = None,
        response_id: str | None = None,
        measurement_origin: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Summarize engine token metrics and client-observed audio cadence."""
        stage0_metrics: dict[str, object] | None = None
        response_created_at_s: float | None = None
        first_text_received_at_s: float | None = None
        audio_received_at_s: list[float] = []
        cumulative_audio_ms: list[float] = []
        for event, received_at_s in zip(self.events, self.event_received_at_s, strict=True):
            if received_at_s < after_s:
                continue
            event_response_id = self.response_id(event)
            if response_id is not None and event_response_id != response_id:
                continue
            if event.get("type") == "response.created" and response_created_at_s is None:
                response_created_at_s = received_at_s
            if (
                event.get("type")
                in {
                    "response.audio_transcript.delta",
                    "response.output_text.delta",
                    "response.text.delta",
                }
                and isinstance(event.get("delta"), str)
                and bool(event["delta"])
                and first_text_received_at_s is None
            ):
                first_text_received_at_s = received_at_s

            stage_metrics = _event_stage_metrics(event)
            stage0 = stage_metrics.get("0") if isinstance(stage_metrics, dict) else None
            if isinstance(stage0, dict):
                stage0_metrics = stage0

            if event.get("type") != "response.audio.delta" or (
                response_id is not None and event_response_id != response_id
            ):
                continue
            delta = event.get("delta") or event.get("audio")
            if not isinstance(delta, str) or not delta:
                continue
            audio_received_at_s.append(received_at_s)
            metadata = event.get("metadata")
            duration_ms = metadata.get("audio_duration_ms") if isinstance(metadata, dict) else None
            if isinstance(duration_ms, int | float) and math.isfinite(float(duration_ms)):
                cumulative_audio_ms.append(max(0.0, float(duration_ms)))

        result: dict[str, object] = {}
        if stage0_metrics is not None:
            raw_itls = stage0_metrics.get("vllm_itls_ms")
            itls = (
                [parsed for value in raw_itls if (parsed := _finite_number(value, nonnegative=True)) is not None]
                if isinstance(raw_itls, list)
                else []
            )
            result["stage0_tokens"] = {
                "source": "engine_stage_metrics",
                "output_token_count": int(_finite_number(stage0_metrics.get("num_tokens_out"), nonnegative=True) or 0),
                "ttft_ms": _finite_number(stage0_metrics.get("vllm_ttft_ms"), nonnegative=True) or 0.0,
                "tpot_ms": _finite_number(stage0_metrics.get("vllm_tpot_ms"), nonnegative=True),
                "itls_ms": itls,
                "inter_token_interval_ms": _interval_summary(itls),
            }

        if audio_received_at_s:
            intervals_ms = [
                (current - previous) * 1000.0 for previous, current in zip(audio_received_at_s, audio_received_at_s[1:])
            ]
            chunk_durations_ms: list[float] = []
            previous_duration_ms = 0.0
            for duration_ms in cumulative_audio_ms:
                chunk_durations_ms.append(
                    duration_ms - previous_duration_ms if duration_ms >= previous_duration_ms else duration_ms
                )
                previous_duration_ms = duration_ms
            interval_summary = _interval_summary(intervals_ms)
            result["audio_output"] = {
                "source": "client_monotonic_receive",
                "chunk_count": len(audio_received_at_s),
                "response_created_to_first_audio_ms": (
                    _rounded_ms((audio_received_at_s[0] - response_created_at_s) * 1000.0)
                    if response_created_at_s is not None
                    else None
                ),
                "commit_to_first_audio_ms": (
                    _rounded_ms((audio_received_at_s[0] - input_committed_at_s) * 1000.0)
                    if input_committed_at_s is not None
                    else None
                ),
                "inter_chunk_interval_ms": interval_summary,
                "chunk_duration_ms": _interval_summary(chunk_durations_ms),
                "max_chunk_gap_ms": interval_summary["max"],
            }
            request_started_at_s = input_committed_at_s if input_committed_at_s is not None else response_created_at_s
            if request_started_at_s is not None:
                audio_duration_ms = (
                    max(cumulative_audio_ms)
                    if cumulative_audio_ms
                    else len(self.audio_bytes(response_id))
                    * 1000.0
                    / (self.output_sample_rate_hz * PCM16_BYTES_PER_SAMPLE)
                )
                audio_generation_ms = max(
                    0.0,
                    (audio_received_at_s[-1] - request_started_at_s) * 1000.0,
                )
                result["request_metrics"] = {
                    "source": "client_monotonic_receive",
                    "measurement_origin": measurement_origin
                    or {
                        "ttft": "input_audio_buffer.commit client send to first non-empty text delta",
                        "ttfp": "input_audio_buffer.commit client send to first audio packet",
                        "rtf": "commit-to-last-audio receive time divided by emitted audio duration",
                    },
                    "ttft_ms": (
                        _rounded_ms((first_text_received_at_s - request_started_at_s) * 1000.0)
                        if first_text_received_at_s is not None
                        else None
                    ),
                    "ttfp_ms": _rounded_ms((audio_received_at_s[0] - request_started_at_s) * 1000.0),
                    "rtf": round(
                        compute_audio_rtf(
                            audio_generation_ms / 1000.0,
                            audio_duration_ms / 1000.0,
                        ),
                        6,
                    )
                    if audio_duration_ms > 0
                    else None,
                    "audio_generation_ms": _rounded_ms(audio_generation_ms),
                    "audio_duration_ms": _rounded_ms(audio_duration_ms),
                }
        return result


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
        native_duplex: bool = True,
        auto_response: bool = True,
        temperature: float | None = None,
        extra_body: dict[str, object] | None = None,
        turn_detection: dict[str, object] | None = None,
        session_id: str | None = None,
        idle_timeout_s: float | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        session_extra_body = dict(extra_body or {})
        if native_duplex:
            session_extra_body.update(
                {
                    "auto_response": auto_response,
                    "minicpmo45_native_duplex": True,
                    "force_listen_count": 0,
                }
            )
        else:
            session_extra_body["minicpmo45_native_duplex"] = False
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
        if session_id:
            session["session_id"] = session_id
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
            if not pcm16:
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
