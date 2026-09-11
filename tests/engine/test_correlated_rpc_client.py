# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import queue
import threading
import time
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
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                client.execute,
                ("collective", "second"),
                _request("second"),
                timeout=1,
                timeout_message="second timed out",
            )
            # A legitimate reply follows registration/submission. Sending it
            # first races the router's intentional unknown-reply rejection.
            assert request_queue.get(timeout=1).rpc_id == "second"
            result_queue.put(
                CollectiveRPCResultMessage(
                    rpc_id="second",
                    method="health",
                    stage_ids=[0],
                    results=["current"],
                )
            )
            current = pending.result(timeout=1)

        assert isinstance(current, CollectiveRPCResultMessage)
        assert current.results == ["current"]
    finally:
        client.close()


@pytest.mark.parametrize("stop", ["timeout", "fatal", "close"])
def test_saturated_submission_unblocks_on_deadline_or_router_shutdown(stop):
    entered = threading.Event()

    class ObservedQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            if isinstance(item, CollectiveRPCRequestMessage):
                entered.set()
            return super().put(item, *args, **kwargs)

    request_queue = ObservedQueue(maxsize=1)
    request_queue.put("occupied")
    result_queue: queue.Queue[object] = queue.Queue()
    client = CorrelatedRpcClient(request_queue, result_queue)
    executor = ThreadPoolExecutor(max_workers=1)
    pending = executor.submit(
        client.execute,
        ("collective", "blocked"),
        _request("blocked"),
        timeout=0.05 if stop == "timeout" else None,
        timeout_message="submission deadline",
        block_on_submit=True,
    )
    try:
        assert entered.wait(timeout=1)
        if stop == "fatal":
            result_queue.put(ErrorMessage(error="engine died", fatal=True))
        elif stop == "close":
            client.close()
        # Wait first so a Future wait timeout cannot masquerade as the RPC
        # TimeoutError (both use the same builtin exception on Python 3.12).
        deadline = time.monotonic() + 1
        while not pending.done() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert pending.done(), "queue submission ignored deadline or router shutdown"
        if stop == "timeout":
            with pytest.raises(TimeoutError, match="submission deadline"):
                pending.result()
        else:
            with pytest.raises(RuntimeError, match="engine died|router closed"):
                pending.result()
        assert request_queue.get_nowait() == "occupied"
        assert request_queue.empty()
    finally:
        # The old unbounded implementation must not leave the test worker
        # stuck, even when the assertion above fails during red verification.
        client.close()
        try:
            request_queue.get_nowait()
        except queue.Empty:
            pass
        executor.shutdown(wait=True)


def test_submit_and_reply_share_one_timeout_budget(monkeypatch):
    import vllm_omni.engine.rpc_result_router as module

    now = [10.0]
    budgets = []
    monkeypatch.setattr(module, "time", type("Clock", (), {"monotonic": staticmethod(lambda: now[0])}), raising=False)

    class Waiter:
        def get_nowait(self):
            raise queue.Empty

        def get(self, *, timeout):
            budgets.append(timeout)
            raise queue.Empty

    class RequestQueue:
        def put(self, message, **kwargs):
            now[0] += 0.4

    client = CorrelatedRpcClient(RequestQueue(), queue.Queue())
    monkeypatch.setattr(client._router, "register", lambda key: Waiter())
    try:
        with pytest.raises(TimeoutError, match="overall deadline"):
            client.execute(
                ("collective", "budget"),
                _request("budget"),
                timeout=1,
                timeout_message="overall deadline",
                block_on_submit=True,
            )
        assert budgets == pytest.approx([0.6])
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
