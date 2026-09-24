# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
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
