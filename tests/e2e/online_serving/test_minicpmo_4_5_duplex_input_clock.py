# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU coverage for input-clocked MiniCPM-o 4.5 duplex sessions (``extra_body.clock: "input"``).

One server, sessions in order:

* a plain session (no input clock) streams the speech asset: it warms the
  pipeline up (the first session after boot can come back without audio) and
  must never receive an ``input_audio_buffer.processed`` acknowledgement;
* two input-clocked sessions stream the same speech asset plus trailing
  silence in 200 ms appends, each sent only after the previous append's
  acknowledgement. Each checks the acknowledgement contract, the first also
  that the model stands still while no input arrives, and both together that
  the event sequence keyed by input position is identical (the deploy's
  Thinker decoding is greedy and its Talker seeded).

``VLLM_TEST_MINICPMO_4_5_MODEL`` (a Hub id or a local checkpoint directory) and
``VLLM_TEST_MINICPMO_4_5_DUPLEX_DEPLOY_CONFIG`` override the model and deploy
config, e.g. a two-GPU deploy on smaller cards.
"""

from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path
from typing import Any

import pybase64 as base64
import pytest
import websockets
from websockets.asyncio.client import ClientConnection

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    DEPLOY_CONFIG,
    MODEL,
    REF_AUDIO_RELATIVE_PATH,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import _ref_audio_data_url
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = pytest.mark.omni

_MODEL = os.environ.get("VLLM_TEST_MINICPMO_4_5_MODEL", MODEL)
_DEPLOY_CONFIG = os.environ.get("VLLM_TEST_MINICPMO_4_5_DUPLEX_DEPLOY_CONFIG", DEPLOY_CONFIG)
SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=["--trust-remote-code"],
        ),
        id="three-stage-single-gpu",
    )
]

APPEND_MS = 200
TRAILING_SILENCE_S = 10.0
ACK_TIMEOUT_S = 120.0
QUIET_S = 5.0
#: ``units[].decision`` of a unit settled without its model output.
_SETTLED_DECISIONS = {"dropped", "cancelled", "aborted", "timed_out"}
_OUTPUT_TYPES = {
    "response.created",
    "response.output_audio.delta",
    "response.output_audio_transcript.delta",
    "response.done",
}


def _ref_audio() -> Path:
    local = Path(_MODEL)
    return local / REF_AUDIO_RELATIVE_PATH if local.is_dir() else resolve_ref_audio()


def _appends() -> list[bytes]:
    """The speech asset and trailing silence as 200 ms 16 kHz PCM16 chunks."""
    with wave.open(str(validated_input_wav()), "rb") as wav_file:
        pcm = wav_file.readframes(wav_file.getnframes())
    step = 16_000 * APPEND_MS // 1000 * 2
    pcm += bytes(int(TRAILING_SILENCE_S * 16_000) * 2)
    pcm += bytes((-len(pcm)) % step)
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def _session_update(ref_audio: Path, extra_body: dict[str, object]) -> dict[str, object]:
    return {
        "type": "session.update",
        "session": {
            "model": _MODEL,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "audio": {"input": {"sample_rate_hz": 16_000}, "output": {"sample_rate_hz": 24_000}},
            "turn_detection": None,
            "ref_audio": _ref_audio_data_url(str(ref_audio)),
            "extra_body": {"auto_response": True, **extra_body},
        },
    }


def _append_event(pcm: bytes) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
            "format": "pcm16",
            "sample_rate_hz": 16_000,
        }
    )


class _Recorder:
    """Every server event of one session, in arrival order, with an ack waiter."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.events: list[dict[str, Any]] = []
        self.acks = 0
        self.error: dict[str, Any] | None = None
        self._changed = asyncio.Condition()

    async def read(self) -> None:
        async for raw in self.ws:
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            async with self._changed:
                if event.get("type") == "error":
                    # Surfaced by ``wait_for`` (this task is never awaited).
                    self.error = event
                    self._changed.notify_all()
                    return
                self.events.append(event)
                if event.get("type") == "input_audio_buffer.processed":
                    self.acks += 1
                self._changed.notify_all()

    async def wait_for(self, predicate, timeout_s: float) -> None:
        async with self._changed:
            await asyncio.wait_for(self._changed.wait_for(lambda: self.error is not None or predicate()), timeout_s)
        if self.error is not None:
            raise AssertionError(f"server error: {self.error}")


async def _plain_session(url: str, ref_audio: Path, chunks: list[bytes]) -> list[dict[str, Any]]:
    """Warm-up and control: stream the asset without the input clock and wait for the reply."""
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        recorder = _Recorder(ws)
        reader = asyncio.create_task(recorder.read())
        await ws.send(json.dumps(_session_update(ref_audio, {})))
        for pcm in chunks:
            await ws.send(_append_event(pcm))
            await asyncio.sleep(APPEND_MS / 1000)
        try:
            await recorder.wait_for(lambda: any(e["type"] == "response.done" for e in recorder.events), 60.0)
        except (TimeoutError, asyncio.TimeoutError):
            pass  # A cold pipeline may not answer; this session only warms it up.
        await ws.send(json.dumps({"type": "session.close"}))
        reader.cancel()
        return recorder.events


async def _clocked_session(url: str, ref_audio: Path, chunks: list[bytes], *, quiet_s: float) -> dict[str, Any]:
    """Step the model one append at a time; optionally stay silent afterwards and watch nothing happen."""
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        recorder = _Recorder(ws)
        reader = asyncio.create_task(recorder.read())
        await ws.send(json.dumps(_session_update(ref_audio, {"clock": "input"})))
        for sent, pcm in enumerate(chunks, start=1):
            await ws.send(_append_event(pcm))
            await recorder.wait_for(lambda sent=sent: recorder.acks >= sent, ACK_TIMEOUT_S)
        settled = len(recorder.events)
        if quiet_s:
            await asyncio.sleep(quiet_s)
        quiet_events = recorder.events[settled:]
        await ws.send(json.dumps({"type": "session.close"}))
        reader.cancel()
        return {"events": recorder.events[:settled], "quiet_events": quiet_events}


def _signature(events: list[dict[str, Any]]) -> list[tuple]:
    """What the model did, keyed by the input position (``audio_end_ms``) of the acknowledgement it preceded."""
    signature: list[tuple] = []
    pending: list[tuple] = []
    for event in events:
        kind = event.get("type")
        if kind == "input_audio_buffer.processed":
            signature.extend((event["audio_end_ms"], *item) for item in pending)
            signature.extend((event["audio_end_ms"], "unit", u["end_ms"], u["decision"]) for u in event["units"])
            pending = []
        elif kind == "response.output_audio.delta":
            pending.append(("audio_bytes", len(base64.b64decode(event.get("delta", "")))))
        elif kind == "response.output_audio_transcript.delta":
            pending.append(("text", event.get("delta")))
        elif kind in ("response.created", "response.done"):
            pending.append((kind,))
    assert not pending, f"outputs after the last acknowledgement: {pending}"
    return signature


def _check_acknowledgements(events: list[dict[str, Any]], *, appends: int) -> None:
    created = next(e for e in events if e["type"] == "session.created")
    chunk_period_ms = created["session"]["capabilities"]["chunk_period_ms"]
    acks = [e for e in events if e["type"] == "input_audio_buffer.processed"]

    assert len(acks) == appends, "exactly one acknowledgement per append"
    assert [a["input_index"] for a in acks] == list(range(1, appends + 1))
    assert {a["trigger"] for a in acks} == {"input_audio_buffer.append"}
    assert not [a for a in acks if "decision" in a], "an append was refused at admission"
    assert [a["audio_end_ms"] for a in acks] == [APPEND_MS * i for i in range(1, appends + 1)]
    unit_ends = [a["unit_end_ms"] for a in acks]
    assert unit_ends == sorted(unit_ends)
    assert all(end % chunk_period_ms == 0 for end in unit_ends)
    # Every unit was completed by the model: none settled without its output.
    settled = [u for a in acks for u in a["units"] if u["decision"] in _SETTLED_DECISIONS or "reason" in u]
    assert not settled, f"units settled without their model output: {settled}"

    # Audio is emitted for a speaking unit before the acknowledgement that covers it. Its last few
    # codec frames may follow with the next unit (vocoder lookahead, as in real time): see the
    # MiniCPM-o entry under "Input-clocked sessions" in docs/serving/realtime_duplex_api.md.
    since_ack: list[str] = []
    for event in events:
        if event["type"] == "input_audio_buffer.processed":
            if any(u["decision"] == "speak" for u in event["units"]):
                assert "response.output_audio.delta" in since_ack, f"speaking unit without its audio: {event}"
            since_ack = []
        elif event["type"] in _OUTPUT_TYPES:
            since_ack.append(event["type"])

    transcript = "".join(e.get("delta") or "" for e in events if e["type"] == "response.output_audio_transcript.delta")
    assert any(e["type"] == "response.output_audio.delta" for e in events), "the model never spoke"
    assert transcript.strip(), "the model spoke without a transcript"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_input_clocked_sessions_are_deterministic_and_stand_still_without_input(omni_server) -> None:
    url = realtime_url(omni_server)
    ref_audio = _ref_audio()
    chunks = _appends()

    plain = asyncio.run(_plain_session(url, ref_audio, chunks))
    assert not [e for e in plain if e["type"] == "input_audio_buffer.processed"], "a plain session got acks"

    first = asyncio.run(_clocked_session(url, ref_audio, chunks, quiet_s=QUIET_S))
    second = asyncio.run(_clocked_session(url, ref_audio, chunks, quiet_s=0))

    for run in (first, second):
        _check_acknowledgements(run["events"], appends=len(chunks))
    assert first["quiet_events"] == [], "the model advanced while no input arrived"
    assert _signature(first["events"]) == _signature(second["events"])
