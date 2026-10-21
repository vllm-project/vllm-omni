# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WebSocket client for the ``/v1/live/sessions`` E2E tests.

Speaks the OpenAI Live protocol directly: 16 kHz PCM16 in both directions,
input paced in real time in 100 ms chunks. Every server event is kept with its
arrival time (seconds since the client started) so tests can check what
arrived in a given window.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import websockets

RATE = 16_000
CHUNK = RATE // 10
SPEECH_WAV = Path(__file__).resolve().parents[3] / "assets" / "minicpmo_4_5" / "response_required_16k.wav"
# Output counts as speech when enough 80 ms blocks reach this RMS (float scale),
# as in personaplex_realtime_duplex.py. PersonaPlex streams audio for every
# frame, silence included, so the mere presence of audio proves nothing.
VOICED_BLOCK_RMS = 1e-3
VOICED_BLOCK = RATE * 80 // 1000
MIN_VOICED_BLOCKS = 5
_TRIMS_RE = re.compile(r"^vllm_omni:live_context_trims(?:_total)?\{[^}]*\}\s+(\S+)$", re.M)


def live_url(omni_server) -> str:
    return f"ws://{omni_server.host}:{omni_server.port}/v1/live/sessions"


def metrics_url(omni_server) -> str:
    return f"http://{omni_server.host}:{omni_server.port}/metrics"


def load_speech(path: Path = SPEECH_WAV) -> np.ndarray:
    with wave.open(str(path)) as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (RATE, 1, 2), path
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def start_event(model: str, voice: str, event_id: str) -> dict[str, Any]:
    return {
        "type": "session.start",
        "event_id": event_id,
        "session": {
            "model": model,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": voice}},
        },
    }


@dataclass
class Utterance:
    start: float
    end: float
    window_end: float


@dataclass
class LiveSession:
    """One Live session; use as ``async with`` to open it."""

    url: str
    model: str
    voice: str
    clock: float = field(default_factory=time.monotonic)
    name: str = "session"
    events: list[dict[str, Any]] = field(default_factory=list)
    audio: list[tuple[float, np.ndarray]] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    close_code: int | None = None
    closed_by_server: bool = False

    def now(self) -> float:
        return time.monotonic() - self.clock

    async def __aenter__(self) -> LiveSession:
        self._ws = await websockets.connect(self.url, max_size=None)
        self._started = asyncio.Event()
        self._closed = asyncio.Event()
        self._closing = False
        self._receiver = asyncio.create_task(self._receive())
        await self._ws.send(json.dumps(start_event(self.model, self.voice, f"start-{self.name}")))
        await asyncio.wait_for(self._started.wait(), timeout=120)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if not self._closed.is_set() and not self.closed_by_server:
            await self.close()
        await self._ws.close()
        await self._receiver

    async def _receive(self) -> None:
        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                t = self.now()
                kind = event.get("type")
                if kind == "session.output_audio.delta":
                    self.audio.append((t, np.frombuffer(base64.b64decode(event["delta"]), dtype="<i2")))
                    continue
                self.events.append({"t": t, **event})
                if kind == "session.started":
                    self._started.set()
                elif kind == "session.closed":
                    self._closed.set()
        except websockets.ConnectionClosed:
            pass
        self.close_code = self._ws.close_code
        if not self._closing:
            self.closed_by_server = True
        self._started.set()
        self._closed.set()

    async def _stream(self, pcm: np.ndarray) -> None:
        tick = time.monotonic()
        for offset in range(0, pcm.size, CHUNK):
            if self.closed_by_server:
                return
            chunk = pcm[offset : offset + CHUNK]
            if chunk.size < CHUNK:
                chunk = np.pad(chunk, (0, CHUNK - chunk.size))
            payload = {"type": "session.input_audio.append", "audio": base64.b64encode(chunk.tobytes()).decode()}
            await self._ws.send(json.dumps(payload))
            tick += CHUNK / RATE
            await asyncio.sleep(max(0.0, tick - time.monotonic()))

    async def silence(self, seconds: float) -> None:
        await self._stream(np.zeros(int(RATE * seconds), dtype="<i2"))

    async def say(self, pcm: np.ndarray, *, gap_s: float) -> Utterance:
        """Stream ``pcm`` then ``gap_s`` of silence; the reply window spans both."""
        start = self.now()
        await self._stream(pcm)
        end = self.now()
        await self.silence(gap_s)
        utterance = Utterance(start, end, self.now())
        self.utterances.append(utterance)
        return utterance

    async def close(self) -> None:
        self._closing = True
        await self._ws.send(json.dumps({"type": "session.close"}))
        await asyncio.wait_for(self._closed.wait(), timeout=60)

    def errors(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("type") == "error"]

    def closed_reason(self) -> str | None:
        closed = [event for event in self.events if event.get("type") == "session.closed"]
        return closed[-1].get("reason") if closed else None

    def voiced_blocks(self, t0: float, t1: float) -> int:
        chunks = [pcm for t, pcm in self.audio if t0 <= t < t1]
        if not chunks:
            return 0
        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        blocks = pcm[: pcm.size - pcm.size % VOICED_BLOCK].reshape(-1, VOICED_BLOCK)
        return int(np.count_nonzero(np.sqrt(np.mean(np.square(blocks), axis=1)) >= VOICED_BLOCK_RMS))

    def answered(self, utterance: Utterance) -> bool:
        return self.voiced_blocks(utterance.start, utterance.window_end) >= MIN_VOICED_BLOCKS

    def transcript(self, t0: float, t1: float) -> str:
        return "".join(
            event["delta"]
            for event in self.events
            if event.get("type") == "session.output_transcript.delta" and t0 <= event["t"] < t1
        )


async def start_over_limit(url: str, model: str, voice: str, event_id: str) -> tuple[list[dict[str, Any]], int | None]:
    """Send ``session.start`` and collect what arrives until the server closes."""
    events: list[dict[str, Any]] = []

    async def collect(ws) -> None:
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    events.append(json.loads(raw))
        except websockets.ConnectionClosed:
            pass

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(start_event(model, voice, event_id)))
        await asyncio.wait_for(collect(ws), timeout=60)
        return events, ws.close_code


def context_trims(url: str) -> float:
    """``vllm_omni:live_context_trims`` summed over its series (0 before the first trim)."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode()
    assert "# TYPE vllm_omni:live_context_trims" in text, "live context trim metric is not exported"
    return sum(float(value) for value in _TRIMS_RE.findall(text))


async def watch_context_trims(url: str, session: LiveSession, stop: asyncio.Event) -> list[tuple[float, float]]:
    """Poll the trim counter every second; return ``(session time, value)`` at each change."""
    changes: list[tuple[float, float]] = []
    last = None
    while not stop.is_set():
        value = await asyncio.to_thread(context_trims, url)
        if value != last:
            changes.append((session.now(), value))
            last = value
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except TimeoutError:
            pass
    return changes
