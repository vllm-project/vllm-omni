# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import time
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.control_plane import DuplexControlPlane
from vllm_omni.engine.duplex.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexFence,
    OpenDuplexSessionMessage,
    SignalDuplexTurnMessage,
)
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_cancel_retry_does_not_preempt_new_epoch_append(monkeypatch):
    old = DuplexFence("cancel-retry")
    current = DuplexFence(old.session_id, epoch=1)
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    port = _TypedStagePort()
    plane = DuplexControlPlane(extension=None, stage_port=port, result_sink=sink)
    session = plane.sessions.open_session(current)
    session.bind_stage_request(0, "live-request", fence=current)
    started, release = asyncio.Event(), asyncio.Event()
    cancelled = False

    async def append(message):
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        await plane.put_result(
            message.control_id,
            fence=current,
            operation="append",
            session_id=current.session_id,
            stage_results=[],
        )

    monkeypatch.setattr(plane, "handle_append", append)
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="live",
            session_id=current.session_id,
            fence=current,
            mode="append_tokens",
            payload={},
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    plane.dispatch(
        SignalDuplexTurnMessage(
            control_id="retry",
            session_id=old.session_id,
            fence=old,
            next_fence=current,
            event="input.cancel",
        )
    )
    try:
        # dispatch must not even schedule cancellation of the newer append.
        assert not plane._control_task_preemption_reasons
        release.set()
        await asyncio.wait_for(plane.drain(), 1)
        replies = [sink.get_nowait(), sink.get_nowait()]
        assert all(reply.ok for reply in replies)
        assert not cancelled
        assert session.fence == current
        assert session.resource_request_ids() == ["live-request"]
        assert not port.cleanup_calls
    finally:
        release.set()
        await plane.shutdown()


class _TypedStagePort:
    stage_count = 0

    def __init__(self):
        self.cleanup_calls = []

    def sampling_defaults(self):
        return ()

    async def cleanup(self, request_ids, *, abort=False):
        self.cleanup_calls.append((request_ids, abort))


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["close", "cancel"])
async def test_valid_terminal_preempts_pending_append(monkeypatch, terminal):
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=_TypedStagePort(), result_sink=sink)
    fence = DuplexFence("valid-preemption")
    plane.sessions.open_session(fence)
    started = asyncio.Event()

    async def append(message):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(plane, "handle_append", append)
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="append", session_id=fence.session_id, fence=fence, mode="append_tokens", payload={}
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    if terminal == "close":
        message = CloseDuplexSessionMessage(control_id="terminal", session_id=fence.session_id, fence=fence)
    else:
        message = SignalDuplexTurnMessage(
            control_id="terminal",
            session_id=fence.session_id,
            fence=fence,
            next_fence=DuplexFence(fence.session_id, epoch=1),
            event="input.cancel",
        )
    plane.dispatch(message)
    try:
        await asyncio.wait_for(plane.drain(), 1)
        results = {reply.control_id: reply for reply in [sink.get_nowait(), sink.get_nowait()]}
        assert not results["append"].ok
        assert results["terminal"].ok
    finally:
        await plane.shutdown()


@pytest.mark.asyncio
async def test_rejected_duplicate_open_is_not_accepted():
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=_TypedStagePort(), result_sink=sink)
    fence = DuplexFence("duplicate-open")
    current = plane.sessions.open_session(fence)
    await plane.handle_open(
        OpenDuplexSessionMessage(
            control_id="duplicate",
            session_id=fence.session_id,
            fence=fence,
            capabilities={},
            session_config={},
            runtime_config={},
        )
    )
    reply = sink.get_nowait()
    assert not reply.ok
    assert reply.error.acceptance == "not_accepted"
    assert plane.sessions.get(fence.session_id) is current
    await plane.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal", "invalid"),
    [
        (terminal, invalid)
        for terminal in ("close", "cancel")
        for invalid in ("incarnation", "epoch", "turn", "response")
    ]
    + [("cancel", invalid) for invalid in ("next_incarnation", "next_epoch", "missing_next")],
)
async def test_invalid_preemption_has_no_append_side_effect(monkeypatch, terminal, invalid):
    fence = DuplexFence("fenced-preemption", incarnation=2, epoch=3, turn_id=4, response_seq=5)
    kwargs = dict(incarnation=2, epoch=3, turn_id=4, response_seq=5)
    field = {"incarnation": "incarnation", "epoch": "epoch", "turn": "turn_id", "response": "response_seq"}.get(invalid)
    if field:
        kwargs[field] -= 1
    stale = DuplexFence(fence.session_id, **kwargs)
    next_fence = DuplexFence(fence.session_id, incarnation=2, epoch=4)
    if invalid == "next_incarnation":
        next_fence = DuplexFence(fence.session_id, incarnation=1, epoch=4)
    elif invalid == "next_epoch":
        next_fence = DuplexFence(fence.session_id, incarnation=2, epoch=3)
    elif invalid == "missing_next":
        next_fence = None
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    port = _TypedStagePort()
    plane = DuplexControlPlane(extension=None, stage_port=port, result_sink=sink)
    session = plane.sessions.open_session(fence)
    started, release = asyncio.Event(), asyncio.Event()
    cancelled = False

    async def append(message):
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        await plane.put_result(
            message.control_id, fence=fence, operation="append", session_id=fence.session_id, stage_results=[]
        )

    monkeypatch.setattr(plane, "handle_append", append)
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="live", session_id=fence.session_id, fence=fence, mode="append_tokens", payload={}
        )
    )
    await started.wait()
    if terminal == "close":
        plane.dispatch(CloseDuplexSessionMessage(control_id="invalid", session_id=fence.session_id, fence=stale))
    else:
        plane.dispatch(
            SignalDuplexTurnMessage(
                control_id="invalid",
                session_id=fence.session_id,
                fence=stale,
                next_fence=next_fence,
                event="input.cancel",
            )
        )
    try:
        # A rejected command must reply even while the live append is blocked.
        reply = await asyncio.wait_for(sink.get(), 1)
        assert reply.control_id == "invalid" and not reply.ok
        assert not cancelled
        assert session.fence == fence and plane.sessions.get(fence.session_id) is session
        assert not port.cleanup_calls
        release.set()
        await asyncio.wait_for(plane.drain(), 1)
        assert (await sink.get()).control_id == "live"
    finally:
        release.set()
        await plane.shutdown()


@pytest.mark.asyncio
async def test_deadline_normalizes_asyncio_timeout(monkeypatch):
    # Python 3.10 has a distinct asyncio.TimeoutError; emulate its identity
    # on newer interpreters too, without mocking the append implementation.
    class AsyncTimeoutError(Exception):
        pass

    async def wait_for(awaitable, *, timeout):
        awaitable.close()
        raise AsyncTimeoutError

    async def pending():
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "TimeoutError", AsyncTimeoutError)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    with pytest.raises(TimeoutError, match="append deadline"):
        await StagePool._await_with_deadline(pending(), time.monotonic() + 10, "append deadline")


@pytest.mark.asyncio
async def test_asyncio_timeout_preserves_append_guard_and_output(monkeypatch, mocker):
    class AsyncTimeoutError(Exception):
        pass

    client = SimpleNamespace(
        stage_type="llm",
        append_streaming_prompt_unit_async=mocker.AsyncMock(side_effect=AsyncTimeoutError),
    )
    pool = StagePool(0, [client])
    monkeypatch.setattr(pool, "get_bound_replica_id", lambda request_id: 0)
    monkeypatch.setattr(pool, "_wait_for_streaming_prompt_append_ready", mocker.AsyncMock())
    monkeypatch.setattr(pool, "_wait_for_streaming_prompt_output_retirement", mocker.AsyncMock())
    processor = mocker.Mock()
    pool._output_processor = processor
    monkeypatch.setattr(asyncio, "TimeoutError", AsyncTimeoutError)
    with pytest.raises(AsyncTimeoutError):
        await pool.submit_streaming_prompt_update(
            "request", SimpleNamespace(prompt_token_ids=[1]), operation_id="op", operation_fingerprint=b"signature"
        )
    assert pool._native_append_operations["request"] == ("op", b"signature", (1,), "uncertain")
    processor.add_request.assert_called_once()
    processor.remove_request.assert_not_called()
