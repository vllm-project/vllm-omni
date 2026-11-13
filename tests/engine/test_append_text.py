# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""append_text from the async engine to the stage scheduler: the route each layer takes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import AppendTextMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_async_omni_sends_appends_for_the_internal_ids_of_an_external_request():
    async def run() -> None:
        omni = AsyncOmni.__new__(AsyncOmni)
        omni.request_states = {
            "take-1-abc": SimpleNamespace(request_id="take-1-abc", external_request_id="take-1"),
            "take-2-def": SimpleNamespace(request_id="take-2-def", external_request_id="take-2"),
        }
        omni.engine = SimpleNamespace(append_text_async=AsyncMock())
        await omni.append_text("take-1", [5, 6], final=True)
        omni.engine.append_text_async.assert_awaited_once_with("take-1-abc", [5, 6], True)
        with pytest.raises(ValueError, match="No running request"):
            await omni.append_text("gone", [1])

    asyncio.run(run())


def test_an_append_waits_for_its_request_add_to_reach_the_engine_queue():
    """generate() registers a request before its ADD task runs; an append must not overtake that ADD."""

    async def run() -> None:
        order: list[str] = []
        omni = AsyncOmni.__new__(AsyncOmni)
        omni.request_states = {"take-1-abc": SimpleNamespace(request_id="take-1-abc", external_request_id="take-1")}

        async def submit() -> None:
            await asyncio.sleep(0.01)
            order.append("add")

        async def append_text_async(*_args) -> None:
            order.append("append")

        omni._pending_submits = {"take-1-abc": asyncio.create_task(submit())}
        omni.engine = SimpleNamespace(append_text_async=append_text_async)
        await omni.append_text("take-1", [5])
        assert order == ["add", "append"]

        # A submit revoked before admission (pause, sleep, abort) refuses the append.
        revoked = asyncio.create_task(asyncio.sleep(10))
        omni._pending_submits = {"take-1-abc": revoked}
        asyncio.get_running_loop().call_soon(revoked.cancel)
        with pytest.raises(ValueError, match="not admitted"):
            await omni.append_text("take-1", [6])
        assert order == ["add", "append"]

    asyncio.run(run())


def test_a_stage_that_cannot_take_the_append_ends_the_request_with_a_client_error():
    async def run() -> None:
        async def refuse(*_args):
            raise ValueError("Stage 0 runs no request take-1-abc that accepts streamed text.")

        orch = Orchestrator.__new__(Orchestrator)
        orch.request_states = {"take-1-abc": SimpleNamespace(entry_stage_id=0)}
        orch.stage_pools = [SimpleNamespace(append_text=refuse)]
        orch._append_text_tasks = set()
        orch._fail_request_client_error = AsyncMock()
        assert await orch._dispatch_message(AppendTextMessage(request_id="take-1-abc", token_ids=[1]))
        await asyncio.gather(*list(orch._append_text_tasks))
        orch._fail_request_client_error.assert_awaited_once_with(
            "take-1-abc", 0, "Stage 0 runs no request take-1-abc that accepts streamed text."
        )

    asyncio.run(run())


def test_the_engine_queues_the_append_behind_the_request_add():
    async def run() -> None:
        engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
        engine._put_request_queue = AsyncMock()
        await engine.append_text_async("take-1-abc", [7], final=False)
        engine._put_request_queue.assert_awaited_once_with(
            AppendTextMessage(request_id="take-1-abc", token_ids=[7], final=False)
        )

    asyncio.run(run())


def test_the_orchestrator_sends_appends_in_order_to_the_entry_stage_and_drops_inactive_ones():
    async def run() -> None:
        calls: list[tuple[str, list[int], bool]] = []

        async def append_text(request_id, token_ids, final):
            calls.append((request_id, token_ids, final))
            await asyncio.sleep(0)
            return None

        orch = Orchestrator.__new__(Orchestrator)
        orch.request_states = {"take-1-abc": SimpleNamespace(entry_stage_id=1)}
        orch.stage_pools = [
            SimpleNamespace(append_text=Mock(side_effect=AssertionError)),
            SimpleNamespace(append_text=append_text),
        ]
        orch._append_text_tasks = set()
        for ids, final in (([1], False), ([2, 3], False), ([], True)):
            assert await orch._dispatch_message(AppendTextMessage(request_id="take-1-abc", token_ids=ids, final=final))
        assert await orch._dispatch_message(AppendTextMessage(request_id="ended", token_ids=[9]))
        await asyncio.gather(*list(orch._append_text_tasks))
        assert calls == [("take-1-abc", [1], False), ("take-1-abc", [2, 3], False), ("take-1-abc", [], True)]
        assert not orch._append_text_tasks

    asyncio.run(run())


def _pool(**client_methods) -> tuple[StagePool, SimpleNamespace]:
    client = SimpleNamespace(stage_type="llm", **client_methods)
    pool = StagePool(0, [client])  # type: ignore[arg-type]
    pool.bind_request_to_replica("take-1-abc", 0)
    pool._output_processor = SimpleNamespace(external_req_ids={"take-1-abc": ["take-1-abc-0"]})
    return pool, client


def test_the_stage_pool_sends_to_the_bound_replica_with_the_engine_request_id():
    async def run() -> None:
        pool, client = _pool(append_text_async=AsyncMock(return_value=None))
        assert await pool.append_text("take-1-abc", [4], True) is None
        client.append_text_async.assert_awaited_once_with("take-1-abc-0", [4], True)

        client.append_text_async = AsyncMock(return_value="Request take-1-abc-0 is not running on this stage.")
        assert "not running" in await pool.append_text("take-1-abc", [5], False)
        with pytest.raises(ValueError, match="runs no request"):
            await pool.append_text("unbound", [5], False)

    asyncio.run(run())


def test_the_stage_client_sends_a_utility_call_to_the_engine_that_holds_the_request():
    async def run() -> None:
        client = StageEngineCoreClient.__new__(StageEngineCoreClient)
        client.core_engine = b"engine-0"
        client._call_utility_async = AsyncMock(return_value=None)
        await client.append_text_async("r", [1, 2], False)
        client._call_utility_async.assert_awaited_once_with("omni_append_text", "r", [1, 2], False, engine=b"engine-0")

        client.reqs_in_flight = {"r": b"engine-1"}
        await client.append_text_async("r", [3], True)
        client._call_utility_async.assert_awaited_with("omni_append_text", "r", [3], True, engine=b"engine-1")

    asyncio.run(run())


def test_the_engine_core_applies_an_append_and_returns_a_refusal_instead_of_raising():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace(append_text=Mock())
    assert engine.omni_append_text("r", [1, 2], True) is None
    engine.scheduler.append_text.assert_called_once_with("r", [1, 2], True)

    engine.scheduler.append_text.side_effect = ValueError("Request r is not running on this stage.")
    assert engine.omni_append_text("r", [3], False) == "Request r is not running on this stage."

    engine.scheduler = SimpleNamespace()
    assert "does not accept" in engine.omni_append_text("r", [3], False)
