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

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import PreparedSample
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class _RecordingClient:
    """Records the commit and every appended unit, in order."""

    def __init__(self, *args, **kwargs) -> None:
        self.events = RealtimeEventCollector()
        self.calls: list[tuple[str, bool | None]] = []
        self.fail_appends = bool(kwargs.get("fail_appends"))
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
        return 0

    async def commit(self) -> None:
        # A real commit is a socket write: it yields before it lands, so the
        # padding would get its first unit in if it started too early.
        await asyncio.sleep(0.01)
        self.calls.append(("commit", None))

    async def append_audio(self, audio: bytes, *, is_speech: bool = True) -> None:
        if self.fail_appends:
            raise ConnectionResetError("socket closed")
        self.calls.append(("append", is_speech))

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


def _prepared() -> PreparedSample:
    return PreparedSample(pcm=b"", frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None)


async def _instant_wait(check, *, timeout_s: float, label: str) -> bool:
    await asyncio.sleep(0.05)  # long enough for the padding to append a unit
    return True


async def test_padding_starts_only_after_the_commit() -> None:
    client = _RecordingClient()
    stop = asyncio.Event()

    task = await runner._commit_clip_then_pad(client, 20, stop)
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert client.calls[0] == ("commit", None)
    assert client.calls[1:] and all(call == ("append", False) for call in client.calls[1:])


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
    assert calls[1:] and all(call == ("append", False) for call in calls[1:])


async def test_padding_stops_when_the_drain_finishes() -> None:
    client = _RecordingClient()
    stop = asyncio.Event()

    task = await runner._commit_clip_then_pad(client, 20, stop)
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

    task = await runner._commit_clip_then_pad(client, 20, stop)
    await asyncio.wait_for(task, timeout=1.0)

    assert client.calls == [("commit", None)]


async def test_padding_ends_quietly_when_the_socket_is_closed() -> None:
    client = _RecordingClient(fail_appends=True)
    stop = asyncio.Event()

    task = await runner._commit_clip_then_pad(client, 20, stop)
    # A closed socket ends the padding instead of failing the sample.
    await asyncio.wait_for(task, timeout=1.0)

    assert client.calls == [("commit", None)]
