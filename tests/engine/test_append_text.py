# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""append_text from the async engine to the stage scheduler: the route each layer takes."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from unittest.mock import AsyncMock, create_autospec

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.output_processor import OutputProcessor

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import AppendTextMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _client_states(**external_by_internal: str) -> dict[str, ClientRequestState]:
    return {
        internal: ClientRequestState(request_id=internal, external_request_id=external)
        for internal, external in external_by_internal.items()
    }


def _orchestrator(*pools: StagePool, entry_stage_id: int = 0) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.request_states = {
        "take-1-abc": OrchestratorRequestState(request_id="take-1-abc", entry_stage_id=entry_stage_id)
    }
    orch.stage_pools = list(pools)
    orch._append_text_tasks = set()
    orch._fail_request_client_error = AsyncMock()
    return orch


def test_async_omni_sends_appends_for_the_internal_ids_of_an_external_request():
    async def run() -> None:
        omni = AsyncOmni.__new__(AsyncOmni)
        omni.request_states = _client_states(**{"take-1-abc": "take-1", "take-2-def": "take-2"})
        omni.engine = create_autospec(AsyncOmniEngine, instance=True)
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
        omni.request_states = _client_states(**{"take-1-abc": "take-1"})

        async def submit() -> None:
            await asyncio.sleep(0.01)
            order.append("add")

        async def append_text_async(*_args) -> None:
            order.append("append")

        omni._pending_submits = {"take-1-abc": asyncio.create_task(submit())}
        omni.engine = create_autospec(AsyncOmniEngine, instance=True)
        omni.engine.append_text_async.side_effect = append_text_async
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
        pool = create_autospec(StagePool, instance=True)
        pool.append_text.side_effect = ValueError("Stage 0 runs no request take-1-abc that accepts streamed text.")
        orch = _orchestrator(pool)
        assert await orch._dispatch_message(AppendTextMessage(request_id="take-1-abc", token_ids=[1]))
        await asyncio.gather(*list(orch._append_text_tasks))
        orch._fail_request_client_error.assert_awaited_once_with(
            "take-1-abc", 0, "Stage 0 runs no request take-1-abc that accepts streamed text."
        )

    asyncio.run(run())


def test_a_lost_append_ends_the_request_instead_of_leaving_it_waiting_for_text():
    """Without this, the request would hold its seat until the text deadline, or forever with it off."""

    async def run() -> None:
        pool = create_autospec(StagePool, instance=True)
        pool.append_text.side_effect = ConnectionError("engine core gone")
        orch = _orchestrator(pool)
        assert await orch._dispatch_message(AppendTextMessage(request_id="take-1-abc", token_ids=[1]))
        await asyncio.gather(*list(orch._append_text_tasks))
        orch._fail_request_client_error.assert_awaited_once()
        args, kwargs = orch._fail_request_client_error.await_args
        assert args[:2] == ("take-1-abc", 0) and "engine core gone" in args[2]
        assert kwargs["status_code"] == HTTPStatus.INTERNAL_SERVER_ERROR.value

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

        other, entry = create_autospec(StagePool, instance=True), create_autospec(StagePool, instance=True)
        other.append_text.side_effect = AssertionError
        entry.append_text.side_effect = append_text
        orch = _orchestrator(other, entry, entry_stage_id=1)
        for ids, final in (([1], False), ([2, 3], False), ([], True)):
            assert await orch._dispatch_message(AppendTextMessage(request_id="take-1-abc", token_ids=ids, final=final))
        assert await orch._dispatch_message(AppendTextMessage(request_id="ended", token_ids=[9]))
        await asyncio.gather(*list(orch._append_text_tasks))
        assert calls == [("take-1-abc", [1], False), ("take-1-abc", [2, 3], False), ("take-1-abc", [], True)]
        assert not orch._append_text_tasks

    asyncio.run(run())


def _pool() -> tuple[StagePool, StageEngineCoreClient]:
    client = create_autospec(StageEngineCoreClient, instance=True)
    client.stage_type = "llm"
    output_processor = create_autospec(OutputProcessor, instance=True)
    output_processor.external_req_ids = {"take-1-abc": ["take-1-abc-0"]}
    pool = StagePool(0, [client], output_processor=output_processor)
    pool.bind_request_to_replica("take-1-abc", 0)
    return pool, client


def test_the_stage_pool_sends_to_the_bound_replica_with_the_engine_request_id():
    async def run() -> None:
        pool, client = _pool()
        client.append_text_async.return_value = None
        assert await pool.append_text("take-1-abc", [4], True) is None
        client.append_text_async.assert_awaited_once_with("take-1-abc-0", [4], True)

        client.append_text_async.return_value = "Request take-1-abc-0 is not running on this stage."
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
    engine.scheduler = create_autospec(OmniARScheduler, instance=True)
    assert engine.omni_append_text("r", [1, 2], True) is None
    engine.scheduler.append_text.assert_called_once_with("r", [1, 2], True)

    engine.scheduler.append_text.side_effect = ValueError("Request r is not running on this stage.")
    assert engine.omni_append_text("r", [3], False) == "Request r is not running on this stage."

    # A stage whose scheduler has no append_text (vLLM's own scheduler).
    engine.scheduler = create_autospec(Scheduler, instance=True)
    assert "does not accept" in engine.omni_append_text("r", [3], False)
