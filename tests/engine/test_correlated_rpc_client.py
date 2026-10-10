# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import queue
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import janus
import pytest

from vllm_omni.engine import rpc_result_router
from vllm_omni.engine.messages import (
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    ErrorMessage,
)
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=["stdlib", "janus"])
def bounded_request_queue(request: pytest.FixtureRequest) -> Iterator:
    if request.param == "stdlib":
        yield queue.Queue(maxsize=1)
    else:
        transport: janus.Queue = janus.Queue(maxsize=1)
        try:
            yield transport.sync_q
        finally:
            transport.close()


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


@pytest.mark.parametrize("timeout", [0.0, 0.01])
def test_blocking_submission_times_out_when_queue_is_full(bounded_request_queue, monkeypatch, timeout: float) -> None:
    queued = _request("already-queued")
    bounded_request_queue.put_nowait(queued)
    client = CorrelatedRpcClient(bounded_request_queue, queue.Queue())
    original_put = bounded_request_queue.put

    def bounded_put(item, block=True, timeout=None):
        # Fail on an unbounded call instead of hanging the regression suite.
        assert timeout is not None, "blocking submission must receive the RPC timeout"
        return original_put(item, block=block, timeout=timeout)

    monkeypatch.setattr(bounded_request_queue, "put", bounded_put)
    key = ("collective", "timed-out")
    try:
        with pytest.raises(TimeoutError, match="submission timed out") as error:
            client.execute(
                key,
                _request("timed-out"),
                timeout=timeout,
                timeout_message="submission timed out",
                block_on_submit=True,
            )
        assert isinstance(error.value.__cause__, queue.Full)
        assert bounded_request_queue.get_nowait() is queued
        assert bounded_request_queue.empty()
        assert key not in client._router._pending
    finally:
        client.close()


def test_nonblocking_submission_preserves_queue_full(bounded_request_queue) -> None:
    bounded_request_queue.put_nowait(_request("already-queued"))
    client = CorrelatedRpcClient(bounded_request_queue, queue.Queue())
    key = ("collective", "not-submitted")
    try:
        with pytest.raises(queue.Full):
            client.execute(key, _request("not-submitted"), timeout=1, timeout_message="unexpected timeout")
        assert key not in client._router._pending
    finally:
        client.close()


@pytest.mark.parametrize("block_on_submit", [False, True])
def test_negative_timeout_is_rejected_before_submission(bounded_request_queue, block_on_submit: bool) -> None:
    client = CorrelatedRpcClient(bounded_request_queue, queue.Queue())
    key = ("collective", "invalid-timeout")
    try:
        with pytest.raises(ValueError, match="non-negative"):
            client.execute(
                key,
                _request("invalid-timeout"),
                timeout=-1,
                timeout_message="unexpected timeout",
                block_on_submit=block_on_submit,
            )
        assert bounded_request_queue.empty()
        assert key not in client._router._pending
    finally:
        client.close()


@pytest.mark.parametrize("finished_at, remaining", [(107.0, 3.0), (110.0, 0.0), (112.0, 0.0)])
def test_submission_and_result_wait_share_one_deadline(bounded_request_queue, mocker, finished_at, remaining) -> None:
    client = CorrelatedRpcClient(bounded_request_queue, queue.Queue())
    waiter: queue.Queue = queue.Queue()
    expected = CollectiveRPCResultMessage(rpc_id="budget", method="health", stage_ids=[0], results=["ok"])
    waiter.put_nowait(expected)
    mocker.patch.object(client._router, "register", return_value=waiter)
    mocker.patch.object(rpc_result_router, "monotonic", side_effect=[100.0, 102.0, finished_at], create=True)
    submit = mocker.spy(bounded_request_queue, "put")
    receive = mocker.spy(waiter, "get")
    message = _request("budget")
    try:
        assert (
            client.execute(
                ("collective", "budget"),
                message,
                timeout=10,
                timeout_message="budget exhausted",
                block_on_submit=True,
            )
            is expected
        )
        submit.assert_called_once_with(message, timeout=8.0)
        receive.assert_called_once_with(timeout=remaining)
    finally:
        client.close()


def test_blocking_submission_without_deadline_still_receives_ack(bounded_request_queue) -> None:
    result_queue: queue.Queue = queue.Queue()
    client = CorrelatedRpcClient(bounded_request_queue, result_queue)
    expected = CollectiveRPCResultMessage(rpc_id="unbounded", method="health", stage_ids=[0], results=["ok"])
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                client.execute,
                ("collective", "unbounded"),
                _request("unbounded"),
                timeout=None,
                timeout_message="unexpected timeout",
                block_on_submit=True,
            )
            try:
                assert bounded_request_queue.get(timeout=1).rpc_id == "unbounded"
                result_queue.put_nowait(expected)
                assert pending.result(timeout=1) is expected
            finally:
                client.close()
    finally:
        client.close()


def test_submission_transport_error_is_preserved(bounded_request_queue, monkeypatch) -> None:
    client = CorrelatedRpcClient(bounded_request_queue, queue.Queue())
    key = ("collective", "transport-error")

    def failed_put(*args, **kwargs):
        raise RuntimeError("transport closed")

    monkeypatch.setattr(bounded_request_queue, "put", failed_put)
    try:
        with pytest.raises(RuntimeError, match="transport closed"):
            client.execute(
                key,
                _request("transport-error"),
                timeout=1,
                timeout_message="unexpected timeout",
                block_on_submit=True,
            )
        assert key not in client._router._pending
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
