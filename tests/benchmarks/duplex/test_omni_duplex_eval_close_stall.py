# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval: a closing write that never returns must not cost the sample.

``close_session`` awaits an unbounded ``send({"type": "session.close"})`` before
its own timeout starts, so the same socket stall that blocked the playback ack
can park the sample after it already paid for the session: no result, no
artifacts, and an outer ``asyncio.wait_for()`` that waits forever for the
cancellation cleanup. These tests pin the bounded close and its forced transport
teardown.
"""

import asyncio
import time

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import PreparedSample
from vllm_omni.benchmarks.duplex_session_inputs import TURN_DETECTION_SERVER_VAD
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_CLOSE_TIMEOUT_S = 0.05
_GRACE_S = 0.05
#: Generous next to the tiny deadlines above, tight next to an unbounded wait.
_BUDGET_S = 5.0


class _FakeTransport:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class _FakeWebSocket:
    """The one attribute the teardown needs from a real WebSocket: a transport."""

    def __init__(self) -> None:
        self.transport = _FakeTransport()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _StalledCloseClient:
    """A duplex client whose closing write never comes back."""

    def __init__(self, *args, **kwargs) -> None:
        self.events = RealtimeEventCollector()
        self._ws = _FakeWebSocket()
        self.ignore_cancellation = bool(kwargs.get("ignore_cancellation"))
        self.close_calls = 0

    async def close_session(self, *, timeout_s: float = 20.0) -> None:
        self.close_calls += 1
        never = asyncio.Event().wait()
        if self.ignore_cancellation:
            # A stuck write does not unwind on cancellation: the coroutine it
            # waits on keeps running while the task awaiting it is cancelled.
            await asyncio.shield(never)
        else:
            await never


class _StalledSessionClient(_StalledCloseClient):
    """A whole session that produces a response and then stalls on the close."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.configured = False
        self.acks = 0
        self.events.add({"type": "session.created", "session": {"capabilities": {}}})
        self.events.add({"type": "response.created", "response": {"id": "r1"}})
        self.events.add({"type": "response.done", "response": {"id": "r1", "status": "completed"}})

    async def __aenter__(self) -> "_StalledSessionClient":
        return self

    async def __aexit__(self, *_exc) -> bool:
        # The real client awaits the close handshake here; the point of the
        # teardown is that this never blocks on a socket that is already down.
        await self._ws.close()
        return False

    async def configure(self, *args, **kwargs) -> None:
        self.configured = True

    async def stream_av_units(self, *args, **kwargs) -> int:
        return 0

    async def acknowledge_playback(self) -> None:
        self.acks += 1

    def raise_if_reader_stopped(self) -> None:
        return None


class _SuppressingAckSessionClient(_StalledSessionClient):
    """The playback ack swallows its first cancellation and never unwinds."""

    async def acknowledge_playback(self) -> None:
        self.acks += 1
        never = asyncio.Event().wait()
        try:
            await never
        except asyncio.CancelledError:
            # A stuck write does not unwind on cancellation: the coroutine it
            # waits on keeps running while the task awaiting it is cancelled.
            await asyncio.shield(asyncio.Event().wait())


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


def _install_deadlines(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_FINAL_CLOSE_TIMEOUT_S", _CLOSE_TIMEOUT_S)
    monkeypatch.setattr(runner, "_FORCE_CLOSE_GRACE_S", _GRACE_S)
    monkeypatch.setattr(runner, "_FINAL_ACK_TIMEOUT_S", _CLOSE_TIMEOUT_S)
    monkeypatch.setattr(runner, "_SERVER_TURN_SETTLE_S", 0.0)


@pytest.mark.parametrize("ignore_cancellation", [False, True])
async def test_close_returns_within_the_deadline_and_tears_the_transport_down(monkeypatch, ignore_cancellation) -> None:
    """A close send that never returns is abandoned, not waited on."""
    _install_deadlines(monkeypatch)
    client = _StalledCloseClient(ignore_cancellation=ignore_cancellation)
    started = time.monotonic()

    with pytest.raises(TimeoutError) as raised:
        await runner._close_duplex_session(client, timeout_s=_CLOSE_TIMEOUT_S)

    elapsed = time.monotonic() - started
    assert client.close_calls == 1
    # Bounded by the deadline plus the one grace period for the teardown, not by
    # the socket.
    assert elapsed < _BUDGET_S
    assert "closing the duplex session" in str(raised.value)
    # The forced teardown: the connection is aborted rather than closed politely.
    assert client._ws.transport.aborted is True
    assert client._ws.close_calls == 1


async def test_sample_still_produces_a_result_when_the_close_send_stalls(monkeypatch, tmp_path) -> None:
    """A stalled close costs the sample its handshake, not its artifacts."""
    _install_deadlines(monkeypatch)
    created: list[_StalledSessionClient] = []

    def client_factory(*args, **kwargs):
        client = _StalledSessionClient(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_factory)
    prepared = PreparedSample(pcm=b"", frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None)
    started = time.monotonic()

    result = await runner.generate_sample(
        _sample(),
        url="ws://localhost:8000/v1/realtime",
        model="test-model",
        ref_audio=None,
        output_root=tmp_path,
        turn_detection=TURN_DETECTION_SERVER_VAD,
        prepared=prepared,
    )

    elapsed = time.monotonic() - started
    client = created[0]
    assert elapsed < _BUDGET_S
    # The session ran to the end of its response and then hit the stalled close.
    assert client.configured is True
    assert client.close_calls == 1
    assert result.metadata["response_done"] is True
    assert result.metadata["close_timeout"]
    # The transport was torn down, so the client's exit path did not wait on it.
    assert client._ws.transport.aborted is True
    # And the sample produced its artifacts instead of hanging before writing them.
    assert result.output.exists()
    assert result.output.with_name("s1.meta.json").exists()


async def test_clean_close_leaves_the_transport_alone(monkeypatch) -> None:
    """A close that lands is not torn down: the session ends on the handshake."""

    class _CleanClient(_StalledCloseClient):
        async def close_session(self, *, timeout_s: float = 20.0) -> None:
            self.close_calls += 1

    _install_deadlines(monkeypatch)
    client = _CleanClient()

    await runner._close_duplex_session(client, timeout_s=_CLOSE_TIMEOUT_S)

    assert client.close_calls == 1
    assert client._ws.transport.aborted is False
    assert client._ws.close_calls == 0


async def test_sample_survives_an_ack_that_swallows_cancellation(monkeypatch, tmp_path) -> None:
    """A periodic ack that ignores cancellation costs time, not the sample."""
    # Keep the default server-turn settle (2s > the 250ms ack period) so the
    # periodic ack is parked inside acknowledge_playback before cleanup.
    monkeypatch.setattr(runner, "_FINAL_CLOSE_TIMEOUT_S", _CLOSE_TIMEOUT_S)
    monkeypatch.setattr(runner, "_FORCE_CLOSE_GRACE_S", _GRACE_S)
    monkeypatch.setattr(runner, "_FINAL_ACK_TIMEOUT_S", _CLOSE_TIMEOUT_S)
    created: list[_SuppressingAckSessionClient] = []

    def client_factory(*args, **kwargs):
        client = _SuppressingAckSessionClient(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_factory)
    prepared = PreparedSample(pcm=b"", frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None)
    started = time.monotonic()

    result = await asyncio.wait_for(
        runner.generate_sample(
            _sample(),
            url="ws://localhost:8000/v1/realtime",
            model="test-model",
            ref_audio=None,
            output_root=tmp_path,
            turn_detection=TURN_DETECTION_SERVER_VAD,
            prepared=prepared,
        ),
        timeout=_BUDGET_S,
    )

    elapsed = time.monotonic() - started
    client = created[0]
    assert client.acks >= 1
    # Bounded by deadline + grace, not by the socket the ack parked on.
    assert elapsed < _BUDGET_S
    # The bounded final ack timed out on the same stuck write (asyncio's own
    # TimeoutError carries no message, so recordability is all there is to see).
    assert result.metadata["ack_timeout"] is not None
    assert result.metadata["response_done"] is True
    assert result.output.exists()
    assert result.output.with_name("s1.meta.json").exists()
    # The mocked write answers each cancellation but the last by waiting again,
    # while a closing loop cancels every remaining task once: hand the tasks
    # still parked on it one more cancellation, so the suite does not then wait
    # for an unwind that only a further cancellation would produce.
    for _unwind in range(2):
        parked = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if not parked:
            break
        for task in parked:
            task.cancel()
        await asyncio.wait(parked, timeout=_GRACE_S)
