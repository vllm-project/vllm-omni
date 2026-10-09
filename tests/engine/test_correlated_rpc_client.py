# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_omni.engine.messages import (
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    ErrorMessage,
)
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(rpc_id: str) -> CollectiveRPCRequestMessage:
    return CollectiveRPCRequestMessage(
        rpc_id=rpc_id,
        method="health",
        timeout=1,
        args=(),
        kwargs={},
        stage_ids=None,
    )


def test_correlated_rpc_client_unregisters_timeout_before_late_result() -> None:
    request_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    client = CorrelatedRpcClient(request_queue, result_queue)

    try:
        with pytest.raises(TimeoutError, match="first timed out"):
            client.execute(
                ("collective", "first"),
                _request("first"),
                timeout=0.01,
                timeout_message="first timed out",
            )
        assert request_queue.get_nowait().rpc_id == "first"

        result_queue.put(
            CollectiveRPCResultMessage(
                rpc_id="first",
                method="health",
                stage_ids=[0],
                results=["late"],
            )
        )
        result_queue.put(
            CollectiveRPCResultMessage(
                rpc_id="second",
                method="health",
                stage_ids=[0],
                results=["current"],
            )
        )
        current = client.execute(
            ("collective", "second"),
            _request("second"),
            timeout=1,
            timeout_message="second timed out",
        )

        assert isinstance(current, CollectiveRPCResultMessage)
        assert current.results == ["current"]
    finally:
        client.close()


def test_correlated_rpc_client_rejects_after_fatal_without_enqueuing() -> None:
    request_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    client = CorrelatedRpcClient(request_queue, result_queue)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                client.execute,
                ("collective", "pending"),
                _request("pending"),
                timeout=1,
                timeout_message="unexpected timeout",
            )
            assert request_queue.get(timeout=1).rpc_id == "pending"
            result_queue.put(ErrorMessage(error="orchestrator failed", fatal=True))
            with pytest.raises(RuntimeError, match="orchestrator failed"):
                pending.result(timeout=1)

        with pytest.raises(RuntimeError, match="orchestrator failed"):
            client.execute(
                ("collective", "after-fatal"),
                _request("after-fatal"),
                timeout=1,
                timeout_message="unexpected timeout",
            )
        assert request_queue.empty()
    finally:
        client.close()


@pytest.fixture
def rpc_transport():
    requests: queue.Queue = queue.Queue(maxsize=2)
    results: queue.Queue = queue.Queue()
    client = CorrelatedRpcClient(requests, results)
    try:
        yield client, requests, results
    finally:
        client.close()
        assert not client._router._thread.is_alive()
        assert not client._router._pending


async def _admit(client, requests, rpc_id, *, timeout=1):
    task = asyncio.create_task(
        client.execute_async(("collective", rpc_id), _request(rpc_id), timeout=timeout, timeout_message="reply expired")
    )
    await asyncio.sleep(0)
    assert requests.get_nowait().rpc_id == rpc_id
    return task


def _result(rpc_id):
    return CollectiveRPCResultMessage(rpc_id=rpc_id, method="health", stage_ids=[0], results=[rpc_id])


@pytest.mark.asyncio
async def test_async_and_sync_waiters_share_out_of_order_routing(rpc_transport):
    client, requests, results = rpc_transport
    with ThreadPoolExecutor(max_workers=1) as executor:
        sync = executor.submit(
            client.execute, ("collective", "sync"), _request("sync"), timeout=2, timeout_message="unexpected timeout"
        )
        # Use the event loop, not a blocking get, while waiting for admission.
        for _ in range(1000):
            if not requests.empty():
                break
            await asyncio.sleep(0.001)
        assert requests.get_nowait().rpc_id == "sync"
        first = await _admit(client, requests, "first")
        second = await _admit(client, requests, "second")
        try:
            results.put_nowait(_result("second"))
            results.put_nowait(_result("sync"))
            results.put_nowait(_result("first"))
            assert (await first).results == ["first"]
            assert (await second).results == ["second"]
            assert (await asyncio.wrap_future(sync)).results == ["sync"]
            assert not client._router._pending
        finally:
            first.cancel()
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_full_queue_unregisters_without_waiting(rpc_transport):
    client, requests, _ = rpc_transport
    requests.put_nowait(_request("one"))
    requests.put_nowait(_request("two"))
    with pytest.raises(queue.Full):
        await client.execute_async(("collective", "full"), _request("full"), timeout=1, timeout_message="expired")
    assert not client._router._pending
    assert [requests.get_nowait().rpc_id for _ in range(2)] == ["one", "two"]


@pytest.mark.asyncio
async def test_async_timeout_retires_waiter_before_late_reply(rpc_transport):
    client, requests, results = rpc_transport
    expired = await _admit(client, requests, "expired", timeout=0.01)
    with pytest.raises(TimeoutError, match="reply expired"):
        await expired
    assert not client._router._pending
    current = await _admit(client, requests, "current")
    results.put_nowait(_result("expired"))
    results.put_nowait(_result("current"))
    assert (await current).results == ["current"]
    assert not client._router._pending


@pytest.mark.asyncio
async def test_async_cancel_retires_waiter_without_retracting_admitted_request(rpc_transport):
    client, requests, results = rpc_transport
    cancelled = await _admit(client, requests, "cancelled", timeout=None)
    waiter = client._router._pending[("collective", "cancelled")]
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert waiter.cancelled() and not client._router._pending
    active = await _admit(client, requests, "active")
    results.put_nowait(_result("cancelled"))
    results.put_nowait(_result("active"))
    assert (await active).results == ["active"]


@pytest.mark.asyncio
async def test_async_duplicate_registration_preserves_original_waiter(rpc_transport):
    client, requests, results = rpc_transport
    active = await _admit(client, requests, "same")
    try:
        with pytest.raises(RuntimeError, match="duplicate pending"):
            await client.execute_async(("collective", "same"), _request("same"), timeout=1, timeout_message="expired")
        assert requests.empty() and len(client._router._pending) == 1
        results.put_nowait(_result("same"))
        assert (await active).results == ["same"]
    finally:
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["close", "fatal"])
async def test_async_terminal_errors_unblock_all_waiters_and_reject_admission(rpc_transport, terminal):
    client, requests, results = rpc_transport
    first = await _admit(client, requests, "first", timeout=None)
    second = await _admit(client, requests, "second", timeout=None)
    try:
        if terminal == "close":
            client.close()
        else:
            results.put_nowait(ErrorMessage(error="orchestrator failed", fatal=True))
        for task in (first, second):
            with pytest.raises(RuntimeError, match="closed|orchestrator failed"):
                await asyncio.wait_for(task, 1)
        assert not client._router._pending
        with pytest.raises(RuntimeError, match="closed|orchestrator failed"):
            await client.execute_async(("collective", "after"), _request("after"), timeout=1, timeout_message="expired")
        assert requests.empty()
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_async_cancel_racing_with_popped_reply_keeps_shared_router_alive(rpc_transport, mocker, terminal):
    client, requests, results = rpc_transport
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    deliver = client._router._deliver

    def held_delivery(waiter, message):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(3):
            raise AssertionError("test did not release its own router thread")
        deliver(waiter, message)

    mocker.patch.object(client._router, "_deliver", side_effect=held_delivery)
    pending = await _admit(client, requests, "cancelled", timeout=None)
    waiter = client._router._pending[("collective", "cancelled")]
    legacy = client._router.register(("collective", "legacy")) if terminal else None
    try:
        results.put_nowait(ErrorMessage(error="terminal", fatal=True) if terminal else _result("cancelled"))
        await asyncio.wait_for(entered.wait(), 1)
        # Fatal fanout clears every waiter before delivery, whereas normal
        # routing pops only the cancelled correlation.
        assert not client._router._pending
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert waiter.cancelled()
        release.set()
        if terminal:
            with pytest.raises(RuntimeError, match="terminal"):
                await client.execute_async(
                    ("collective", "after"), _request("after"), timeout=1, timeout_message="expired"
                )
            # A fatal broadcast must continue through legacy waiters even if
            # its first future was cancelled while delivery was in flight.
            assert legacy is not None
            assert legacy.get(timeout=1).error == "terminal"
            assert client._router._thread.is_alive()
        else:
            active = await _admit(client, requests, "active")
            results.put_nowait(_result("active"))
            assert (await active).results == ["active"]
            assert client._router._thread.is_alive()
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_timeout_budget_includes_registration_and_admission(rpc_transport, mocker):
    client, requests, _ = rpc_transport
    clock = mocker.patch("vllm_omni.engine.rpc_result_router.time", autospec=True)
    clock.monotonic.side_effect = [100.0, 100.2, 100.7]
    wait = mocker.patch("vllm_omni.engine.rpc_result_router.asyncio.wait_for", new=mocker.AsyncMock())
    wait.return_value = _result("budget")
    await client.execute_async(("collective", "budget"), _request("budget"), timeout=1, timeout_message="expired")
    assert wait.call_args.kwargs["timeout"] == pytest.approx(0.3)
    assert requests.get_nowait().rpc_id == "budget"
    assert not client._router._pending


@pytest.mark.asyncio
@pytest.mark.parametrize("clock_values, admitted", [([100.0, 102.0], False), ([100.0, 100.1, 102.0], True)])
async def test_async_expired_budget_never_starts_reply_wait(rpc_transport, mocker, clock_values, admitted):
    client, requests, _ = rpc_transport
    clock = mocker.patch("vllm_omni.engine.rpc_result_router.time", autospec=True)
    clock.monotonic.side_effect = clock_values
    with pytest.raises(TimeoutError, match="expired"):
        await client.execute_async(("collective", "expired"), _request("expired"), timeout=1, timeout_message="expired")
    assert requests.empty() is not admitted
    assert not client._router._pending
