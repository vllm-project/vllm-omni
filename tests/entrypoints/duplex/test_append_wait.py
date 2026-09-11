# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest

from vllm_omni.entrypoints.duplex.session_runner import _await_append_result

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [True, False])
async def test_append_wait_preserves_result(result: bool) -> None:
    async def append() -> bool:
        return result

    task = asyncio.create_task(append())
    assert await _await_append_result(task) is result
    assert await _await_append_result(task) is result


@pytest.mark.asyncio
async def test_cancelled_append_does_not_cancel_waiter() -> None:
    entered = asyncio.Event()

    async def append() -> bool:
        entered.set()
        await asyncio.Event().wait()
        return True

    task = asyncio.create_task(append())
    waiter = asyncio.create_task(_await_append_result(task))
    await entered.wait()
    task.cancel()
    assert await asyncio.wait_for(waiter, 1) is False
    assert await _await_append_result(task) is False


@pytest.mark.asyncio
async def test_cancelled_waiter_propagates_cancellation_to_append() -> None:
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def append() -> bool:
        entered.set()
        try:
            await asyncio.Event().wait()
            return True
        finally:
            exited.set()

    task = asyncio.create_task(append())
    waiter = asyncio.create_task(_await_append_result(task))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(exited.wait(), 1)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_append_wait_preserves_failure() -> None:
    async def append() -> bool:
        raise ValueError("append failed")

    task = asyncio.create_task(append())
    with pytest.raises(ValueError, match="append failed"):
        await _await_append_result(task)
