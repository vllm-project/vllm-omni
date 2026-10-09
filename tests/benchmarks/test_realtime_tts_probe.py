# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The Seed-TTS Realtime probe queues for a duplex session slot instead of failing."""

from __future__ import annotations

import pytest

from vllm_omni.benchmarks.patch import patch as bench_patch
from vllm_omni.clients import duplex as duplex_client

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _AdmissionGate:
    """Stands in for ``DuplexClient``: refuses admission ``refusals`` times, then admits."""

    instances: list[_AdmissionGate] = []
    refusals = 0
    refusal_code = "resource_exhausted"

    def __init__(self, url: str, **kwargs: object) -> None:
        del url, kwargs
        self.entered = False
        type(self).instances.append(self)

    async def __aenter__(self) -> _AdmissionGate:
        cls = type(self)
        if cls.refusals > 0:
            cls.refusals -= 1
            raise duplex_client.DuplexProtocolError("duplex_session_capacity_exhausted: limit=4", code=cls.refusal_code)
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    async def events(self):  # the collector's consume loop ends at once
        return
        yield  # pragma: no cover


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> type[_AdmissionGate]:
    _AdmissionGate.instances = []
    _AdmissionGate.refusals = 0
    _AdmissionGate.refusal_code = "resource_exhausted"
    monkeypatch.setattr(duplex_client, "DuplexClient", _AdmissionGate)
    return _AdmissionGate


async def _configure(probe: bench_patch._RealtimeTTSProbe) -> None:
    try:
        await probe.configure("test-model", auto_response=True)
    finally:
        await probe.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_configure_waits_for_a_free_session_slot(gate) -> None:
    """Two refusals, then a slot: the request is served, not counted as failed."""
    gate.refusals = 2
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    await _configure(probe)
    assert len(gate.instances) == 3
    assert gate.instances[-1].entered is True
    assert probe._client is gate.instances[-1]


@pytest.mark.asyncio
async def test_configure_raises_other_rejections_at_once(gate) -> None:
    gate.refusals = 1
    gate.refusal_code = "unsupported_turn_detection"
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    with pytest.raises(duplex_client.DuplexProtocolError):
        await _configure(probe)
    assert len(gate.instances) == 1


@pytest.mark.asyncio
async def test_configure_gives_up_when_the_slot_wait_budget_is_spent(gate, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_patch._RealtimeTTSProbe, "_SESSION_SLOT_WAIT_S", 0.0)
    gate.refusals = 5
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    with pytest.raises(duplex_client.DuplexProtocolError) as excinfo:
        await _configure(probe)
    assert excinfo.value.code == "resource_exhausted"
    assert len(gate.instances) == 1


class _SilenceSink:
    """Stands in for a configured ``DuplexClient``: records every appended chunk."""

    def __init__(self) -> None:
        self.config = duplex_client.SessionConfig()
        self.chunks: list[bytes] = []

    async def append_audio(self, pcm: bytes, *, is_speech: bool | None = None, video_frames=None) -> None:
        del video_frames
        assert is_speech is False
        self.chunks.append(pcm)


@pytest.mark.asyncio
async def test_stream_silence_stops_at_the_turn_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence is fed until the turn settles, not for the whole cap."""
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    sink = _SilenceSink()
    probe._client = sink

    async def no_wait(_: float) -> None:
        return

    monkeypatch.setattr(bench_patch.asyncio, "sleep", no_wait)
    streamed = await probe.stream_silence(seconds=12.0, chunk_ms=200, until=lambda: len(sink.chunks) >= 3)
    assert len(sink.chunks) == 3
    assert streamed == pytest.approx(0.6)
    assert all(len(chunk) == sink.config.input_audio.byte_count(200) for chunk in sink.chunks)


@pytest.mark.asyncio
async def test_stream_silence_runs_to_the_cap_when_the_turn_never_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    sink = _SilenceSink()
    probe._client = sink

    async def no_wait(_: float) -> None:
        return

    monkeypatch.setattr(bench_patch.asyncio, "sleep", no_wait)
    streamed = await probe.stream_silence(seconds=1.0, chunk_ms=200, until=lambda: False)
    assert len(sink.chunks) == 5
    assert streamed == pytest.approx(1.0)


class _TurnEvents:
    def __init__(self, response_ids: list[str], audio: dict[str, bytes], text: dict[str, str] | None = None) -> None:
        self.response_ids = response_ids
        self._audio = audio
        self._text = text or {}

    def audio_bytes(self, response_id: str) -> bytes:
        return self._audio.get(response_id, b"")

    def response_text(self, response_id: str) -> str:
        return self._text.get(response_id, "")


def test_turn_response_is_the_first_one_with_audio_even_if_the_model_spoke_again(caplog) -> None:
    events = _TurnEvents(["r0", "r1", "r2"], {"r1": b"\x01\x00", "r2": b"\x02\x00"})
    with caplog.at_level("WARNING", logger=bench_patch.logger.name):
        assert bench_patch._seed_tts_turn_response_id(events, 0, 0) == "r1"
    assert "spoke again" in caplog.text


def test_turn_response_ignores_responses_from_before_the_turn() -> None:
    events = _TurnEvents(["r0", "r1"], {"r0": b"\x01\x00", "r1": b"\x02\x00"})
    assert bench_patch._seed_tts_turn_response_id(events, 1, 0) == "r1"


def test_turn_without_audio_is_an_error() -> None:
    events = _TurnEvents(["r0"], {})
    with pytest.raises(RuntimeError, match="no audio response"):
        bench_patch._seed_tts_turn_response_id(events, 0, 0)


def test_stall_report_says_the_model_never_answered() -> None:
    events = _TurnEvents(["r0"], {"r0": b"\x01\x00"})
    report = bench_patch._seed_tts_turn_stall_report(events, 1, 3, 30.0)
    assert "turn 3 never started a response" in report
    assert "30.0s of silence" in report


def test_stall_report_lists_a_response_that_never_finished() -> None:
    events = _TurnEvents(["r1"], {"r1": b"\x01\x00\x02\x00"}, {"r1": "hello"})
    report = bench_patch._seed_tts_turn_stall_report(events, 0, 0, 4.2)
    assert "started 1 response(s) after 4.2s of silence" in report
    assert "r1 (4 audio bytes, text 'hello')" in report
