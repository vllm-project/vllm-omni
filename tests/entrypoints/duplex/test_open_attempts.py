# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import msgspec
import pytest

from vllm_omni.engine.duplex.control_client import DuplexControlRequestError
from vllm_omni.engine.duplex.messages import DuplexControlError, DuplexFence
from vllm_omni.entrypoints.duplex.open_attempt import RuntimeOpenAttempts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_rejected_open_does_not_close_existing_session(mocker):
    record = DuplexControlError(code="invalid_argument", message="duplicate", acceptance="not_accepted")
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(record), type=DuplexControlError)
    failure = DuplexControlRequestError(
        {"operation": "open", "error": {"code": decoded.code, "acceptance": decoded.acceptance}}
    )
    close = mocker.AsyncMock(return_value=True)
    attempts = RuntimeOpenAttempts()
    with pytest.raises(DuplexControlRequestError):
        await attempts.execute(DuplexFence("same"), mocker.AsyncMock(side_effect=failure), close)
    close.assert_not_awaited()
    assert attempts.pending_count == 0


@pytest.mark.asyncio
async def test_cancelled_open_waits_for_dispatch_before_cleanup(mocker):
    entered, release = asyncio.Event(), asyncio.Event()

    async def dispatch():
        entered.set()
        await release.wait()
        return {}

    close = mocker.AsyncMock(return_value=True)
    attempts = RuntimeOpenAttempts()
    task = asyncio.create_task(attempts.execute(DuplexFence("pending"), dispatch, close))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    close.assert_not_awaited()
    release.set()
    assert await attempts.retry_pending("pending")
    close.assert_awaited_once()
    assert attempts.pending_count == 0


@pytest.mark.asyncio
async def test_failed_cleanup_retries_without_reconnect(mocker):
    retried = asyncio.Event()
    calls = 0

    async def close():
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        retried.set()
        return True

    attempts = RuntimeOpenAttempts(retry_initial_s=0.01, retry_max_s=0.01)
    with pytest.raises(TimeoutError):
        await attempts.execute(DuplexFence("retry"), mocker.AsyncMock(side_effect=TimeoutError), close)
    await asyncio.wait_for(retried.wait(), 1)
    assert attempts.pending_count == 0


@pytest.mark.asyncio
async def test_pending_open_capacity_rejects_before_dispatch(mocker):
    entered, release = asyncio.Event(), asyncio.Event()

    async def dispatch():
        entered.set()
        await release.wait()

    attempts = RuntimeOpenAttempts(max_pending=1)
    first = asyncio.create_task(attempts.execute(DuplexFence("first"), dispatch, mocker.AsyncMock()))
    await entered.wait()
    rejected = mocker.AsyncMock()
    try:
        with pytest.raises(DuplexControlRequestError) as exc:
            await attempts.execute(DuplexFence("second"), rejected, mocker.AsyncMock())
        assert exc.value.code == "resource_exhausted"
        assert exc.value.acceptance == "not_accepted"
        rejected.assert_not_awaited()
    finally:
        release.set()
        await first
