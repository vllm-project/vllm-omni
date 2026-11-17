# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval: drain padding must not disturb the clip it drains after.

The step loop is driven by the frames the client appends, and the client stops
appending the moment the clip ends, so a reply still open at EOF freezes
mid-utterance. Silence is appended instead while the session drains. That
padding has to start after the commit — an extra unit in the pre-commit buffer
would change the residual unit the commit is shaped to flush — and it has to
stop when the drain ends.
"""

import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import PreparedSample
from vllm_omni.experimental.fullduplex.client import RealtimeDuplexClient, RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class _RecordingClient:
    """Records the commit and every appended unit, in order.

    Mirrors the real ``RealtimeDuplexClient`` surface: every coroutine defined
    here exists on the real client too (see
    ``test_the_fake_client_mirrors_the_real_client_surface``), so these tests
    cannot pass against a method the real client lacks.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.events = RealtimeEventCollector()
        self.calls: list[tuple[str, int | None]] = []
        self.fail_appends = bool(kwargs.get("fail_appends"))
        self.stall_commit = bool(kwargs.get("stall_commit"))
        self.stall_stream = bool(kwargs.get("stall_stream"))
        self.commit_gate = asyncio.Event()
        self.events.add({"type": "session.created", "session": {"capabilities": {}}})
        self.events.add({"type": "response.created", "response": {"id": "r1"}})
        self.events.add({"type": "response.done", "response": {"id": "r1", "status": "completed"}})

    async def __aenter__(self) -> "_RecordingClient":
        return self

    async def __aexit__(self, *_exc) -> bool:
        return False

    async def configure(self, *args, **kwargs) -> None:
        return None

    async def stream_av_units(self, *args, **kwargs) -> int:
        if self.stall_stream:
            await asyncio.sleep(3600.0)
        return 0

    async def commit(self) -> None:
        # A real commit is a socket write: it yields before it lands, so the
        # padding would get its first unit in if it started too early.
        if self.stall_commit:
            await self.commit_gate.wait()
        await asyncio.sleep(0.01)
        self.calls.append(("commit", None))

    async def append_silence(self, duration_ms: int) -> None:
        if self.fail_appends:
            raise ConnectionResetError("socket closed")
        self.calls.append(("append_silence", duration_ms))

    async def acknowledge_playback(self) -> None:
        return None

    async def close_session(self, *, timeout_s: float = 20.0) -> None:
        return None

    def raise_if_reader_stopped(self) -> None:
        return None


def _sample() -> DuplexSample:
    return DuplexSample(
        id="s1",
        split="dev",
        family="pr",
        task_type=None,
        video=None,
        question_audio=None,
        video_duration=1.0,
    )


def _prepared(duration: float = 1.0) -> PreparedSample:
    return PreparedSample(pcm=b"", frames=(), duration=duration, ref_audio=None, ref_audio_sha256=None)


async def _instant_wait(check, *, timeout_s: float, label: str) -> bool:
    await asyncio.sleep(0.05)  # long enough for the padding to append a unit
    return True


async def test_padding_starts_only_after_the_commit() -> None:
    client = _RecordingClient()
    stop = asyncio.Event()

    task, commit_timeout = await runner._commit_clip_then_pad(client, 20, stop)
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert commit_timeout is None
    assert client.calls[0] == ("commit", None)
    assert client.calls[1:] and all(call == ("append_silence", 20) for call in client.calls[1:])


def test_the_fake_client_mirrors_the_real_client_surface() -> None:
    """The fake must only define coroutines the real client also defines.

    A fake ``append_audio`` once let these tests pass while the real
    ``RealtimeDuplexClient`` had no such method: the swallowed
    ``AttributeError`` ended the padding with zero units sent.
    """
    real_coroutines = {name for name, attr in vars(RealtimeDuplexClient).items() if asyncio.iscoroutinefunction(attr)}
    fake_coroutines = {name for name, attr in vars(_RecordingClient).items() if asyncio.iscoroutinefunction(attr)}

    assert fake_coroutines <= real_coroutines


async def test_padding_works_against_the_real_client() -> None:
    """The padding must send real append events, not just fake-client calls.

    Regression for the swallowed ``AttributeError``: only ``send`` is
    stubbed, everything else — the event the padding builds included — is
    the real client's code.
    """
    client = RealtimeDuplexClient("ws://localhost:8000/v1/realtime")
    sent: list[dict] = []

    async def record_send(event: dict) -> None:
        sent.append(event)

    client.send = record_send
    stop = asyncio.Event()
    task = asyncio.create_task(runner._pad_silence_until_stopped(client, 20, stop))
    try:
        for _ in range(1000):
            if sent:
                break
            await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert sent, "the padding must send at least one unit on the real client"
    first = sent[0]
    assert first["type"] == "input_audio_buffer.append"
    assert first["input_audio_format"] == "pcm16"
    assert first["sample_rate_hz"] == runner.PCM16_SAMPLE_RATE
    assert first["duration_ms"] == 20
    assert first["audio_end_ms"] == 20
    assert base64.b64decode(first["audio"]) == bytes(640)  # 20 ms of zeroed PCM16
    assert [event["audio_end_ms"] for event in sent] == sorted(event["audio_end_ms"] for event in sent)


async def test_sample_pads_only_after_its_commit(monkeypatch, tmp_path) -> None:
    created: list[_RecordingClient] = []

    def client_factory(*args, **kwargs):
        client = _RecordingClient(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_factory)
    monkeypatch.setattr(runner, "wait_for", _instant_wait)

    await runner.generate_sample(
        _sample(),
        url="ws://localhost:8000/v1/realtime",
        model="test-model",
        ref_audio=None,
        output_root=tmp_path,
        prepared=_prepared(),
    )

    calls = created[0].calls
    assert calls[0] == ("commit", None)
    # The padding period comes from the session's negotiated chunk period,
    # not from the test: only its shape is pinned here.
    assert calls[1:] and all(
        name == "append_silence" and period is not None and period > 0 for name, period in calls[1:]
    )


async def test_sample_replay_accepts_a_session_without_native_commit(tmp_path) -> None:
    calls: list[str] = []
    pcm = bytes(32_000)
    output_audio = bytes(4_800)

    class ReplaySession:
        frame_transport = "append"
        supports_session_stats = False

        def __init__(self) -> None:
            self.events = RealtimeEventCollector()

        async def configure(
            self,
            model: str,
            *,
            ref_audio: str | None,
            instructions: str | None,
            extra_body: dict[str, object] | None,
            turn_detection: str,
        ) -> None:
            assert model == "test-model"
            calls.append("configure")

        def prepare_input(self, audio: bytes, frames: tuple[tuple[float, bytes], ...]) -> bytes:
            assert audio == pcm
            return audio

        async def stream_input(
            self,
            audio: bytes,
            frames: tuple[tuple[float, bytes], ...],
            *,
            sample_id: str,
            unit_ms: int,
            realtime: bool,
        ) -> None:
            assert audio == pcm
            assert frames == ()
            assert sample_id == "s1"
            assert realtime
            calls.append("stream")

        async def finish_input(self, *, timeout_s: float) -> runner.ReplayDrainResult:
            calls.append("finish")
            self.events.add({"type": "response.created", "response": {"id": "r1"}})
            self.events.add(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "r1",
                    "delta": base64.b64encode(output_audio).decode("ascii"),
                }
            )
            self.events.add({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
            return runner.ReplayDrainResult(response_done=True)

        async def acknowledge_playback(self) -> None:
            pass

        async def get_session_stats(self, *, timeout_s: float, event_id: str) -> dict[str, object]:
            raise AssertionError("The runner requested stats from a session without that capability")

        async def close(self, *, timeout_s: float) -> None:
            calls.append("close")

    @asynccontextmanager
    async def open_session(
        url: str, model: str, headers: dict[str, str] | None
    ) -> AsyncIterator[runner.DuplexReplaySession]:
        assert (url, model, headers) == ("ws://localhost:8000/v1/realtime", "test-model", {"X-Test": "replay"})
        yield ReplaySession()

    result = await runner.generate_sample(
        _sample(),
        url="ws://localhost:8000/v1/realtime",
        model="test-model",
        ref_audio=None,
        output_root=tmp_path,
        prepared=PreparedSample(pcm=pcm, frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None),
        additional_headers={"X-Test": "replay"},
        session_factory=open_session,
    )

    assert calls == ["configure", "stream", "finish", "close"]
    assert result.error == ""
    assert result.metadata["response_done"] is True
    assert result.metadata["input_committed"] is False
    assert result.audio_bytes == len(output_audio)
    assert result.request_metrics[0]["audio_continuity"]["audio_chunk_count"] == 1


async def test_padding_stops_when_the_drain_finishes() -> None:
    client = _RecordingClient()
    stop = asyncio.Event()

    task, _commit_timeout = await runner._commit_clip_then_pad(client, 20, stop)
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    sent = len(client.calls)
    await asyncio.sleep(0.1)
    assert len(client.calls) == sent


async def test_padding_sends_nothing_when_already_stopped() -> None:
    client = _RecordingClient()
    stop = asyncio.Event()
    stop.set()

    task, _commit_timeout = await runner._commit_clip_then_pad(client, 20, stop)
    await asyncio.wait_for(task, timeout=1.0)

    assert client.calls == [("commit", None)]


async def test_padding_ends_quietly_when_the_socket_is_closed() -> None:
    client = _RecordingClient(fail_appends=True)
    stop = asyncio.Event()

    task, _commit_timeout = await runner._commit_clip_then_pad(client, 20, stop)
    # A closed socket ends the padding instead of failing the sample.
    await asyncio.wait_for(task, timeout=1.0)

    assert client.calls == [("commit", None)]


async def test_stalled_commit_is_bounded_and_padding_still_starts(monkeypatch) -> None:
    """A commit write that never lands costs a recorded timeout, not the sample."""
    monkeypatch.setattr(runner, "_FINAL_COMMIT_TIMEOUT_S", 0.05)
    client = _RecordingClient(stall_commit=True)
    stop = asyncio.Event()

    task, commit_timeout = await asyncio.wait_for(runner._commit_clip_then_pad(client, 20, stop), timeout=1.0)
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)
    # Let the shielded commit write unwind so it leaves no pending task behind.
    client.commit_gate.set()
    await asyncio.sleep(0)

    assert commit_timeout is not None
    assert ("commit", None) not in client.calls  # the stalled write never landed
    assert client.calls and all(call == ("append_silence", 20) for call in client.calls)


async def test_stalled_input_is_bounded_and_the_sample_still_lands(monkeypatch, tmp_path) -> None:
    """A stalled input write is recorded; the sample still returns its artifacts."""
    monkeypatch.setattr(runner, "_INPUT_SLACK_S", 0.05)
    created: list[_RecordingClient] = []

    def client_factory(*args, **kwargs):
        client = _RecordingClient(*args, **{**kwargs, "stall_stream": True})
        created.append(client)
        return client

    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_factory)
    monkeypatch.setattr(runner, "wait_for", _instant_wait)

    result = await asyncio.wait_for(
        runner.generate_sample(
            _sample(),
            url="ws://localhost:8000/v1/realtime",
            model="test-model",
            ref_audio=None,
            output_root=tmp_path,
            prepared=_prepared(duration=0.0),
        ),
        timeout=5.0,
    )

    assert result.metadata["input_timeout"] is not None
    assert result.metadata["input_committed"] is True  # the commit itself was fine
    assert result.metadata["response_done"] is True
    assert result.metadata["input_timeout"] in result.error
    assert result.output.exists()  # the sample still lands its artifacts
