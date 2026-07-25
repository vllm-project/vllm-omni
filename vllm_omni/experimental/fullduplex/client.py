"""Reusable Realtime WebSocket, PCM, and event helpers for MiniCPM-o demos."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
import wave
from collections.abc import Callable
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


def _rounded_ms(value: float) -> float:
    return round(float(value), 3)


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
    model: str,
    *,
    autostart: bool | None = None,
    session_id: str | None = None,
) -> str:
    """Add the explicit native-duplex query parameters to a Realtime URL."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("duplex", "1")
    query.setdefault("model", model)
    query.setdefault("minicpmo45_native_duplex", "1")
    if autostart is not None:
        query.setdefault("autostart", "1" if autostart else "0")
    if session_id:
        query.setdefault("session_id", session_id)
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
    ) -> dict[str, object]:
        """Summarize engine token metrics and client-observed audio cadence."""
        stage0_metrics: dict[str, object] | None = None
        response_created_at_s: float | None = None
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
                [float(value) for value in raw_itls if isinstance(value, int | float)]
                if isinstance(raw_itls, list)
                else []
            )
            result["stage0_tokens"] = {
                "source": "engine_stage_metrics",
                "output_token_count": int(stage0_metrics.get("num_tokens_out") or 0),
                "ttft_ms": float(stage0_metrics.get("vllm_ttft_ms") or 0.0),
                "tpot_ms": float(stage0_metrics.get("vllm_tpot_ms") or 0.0),
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
        return result


class RealtimeDuplexClient:
    """Small async client used by the user demo and reusable smoke probes."""

    def __init__(self, url: str, *, max_size: int = 64 * 1024 * 1024) -> None:
        self.url = url
        self.max_size = max_size
        self.events = RealtimeEventCollector()
        self._ws: Any = None
        self._reader_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> RealtimeDuplexClient:
        self._ws = await websockets.connect(self.url, max_size=self.max_size)
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
                    self.events.add(event)
        except ConnectionClosed:
            return

    async def send(self, event: dict[str, object]) -> None:
        await self._ws.send(json.dumps(event))

    async def configure(
        self,
        model: str,
        *,
        output_audio_format: str = "pcm16",
        ref_audio: str | None = None,
        session_id: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        session: dict[str, object] = {
            "model": model,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": output_audio_format,
            "turn_detection": None,
            "overlap_policy": "listen_only",
            "playback_commit_policy": "ack_only",
            "extra_body": {
                "auto_response": True,
                "minicpmo45_native_duplex": True,
                "force_listen_count": 0,
            },
        }
        if ref_audio is not None:
            session["ref_audio"] = ref_audio
        if session_id:
            session["session_id"] = session_id
        await self.send({"type": "session.update", "session": session})
        await wait_for(
            lambda: self.events.count("session.created") > 0,
            timeout_s=timeout_s,
            label="session.created",
        )

    async def stream_pcm16(
        self,
        pcm16: bytes,
        *,
        chunk_ms: int = 200,
        realtime: bool = True,
    ) -> None:
        chunk_bytes = max(
            PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000,
            PCM16_BYTES_PER_SAMPLE,
        )
        audio_end_ms = 0
        for offset in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[offset : offset + chunk_bytes]
            duration_ms = len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
            audio_end_ms += duration_ms
            await self.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "input_audio_format": "pcm16",
                    "sample_rate_hz": PCM16_SAMPLE_RATE,
                    "duration_ms": duration_ms,
                    "audio_end_ms": audio_end_ms,
                }
            )
            if realtime:
                await asyncio.sleep(duration_ms / 1000)

    async def commit(self) -> None:
        await self.send({"type": "input_audio_buffer.commit", "final": True})

    async def acknowledge_playback(self) -> None:
        for response_id in self.events.response_ids:
            pcm16 = self.events.audio_bytes(response_id)
            if not pcm16:
                continue
            played_ms = len(pcm16) * 1000 // (self.events.output_sample_rate_hz * PCM16_BYTES_PER_SAMPLE)
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
        await self.send({"type": "session.close"})
        await wait_for(
            lambda: self.events.count("session.closed") > 0,
            timeout_s=timeout_s,
            label="session.closed",
        )
