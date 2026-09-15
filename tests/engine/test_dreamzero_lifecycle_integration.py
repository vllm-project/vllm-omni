# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Orchestrator boundary for coordinated session lifecycle.

Drives the real `Orchestrator` output/settlement methods and the real
`DiffusionStageLifecycleCoordinator` against a fake output queue and fake stage
workers. The point is the wiring between them: a coordinator-only test cannot
show that a terminal success waits for settlement, that a duplicate terminal is
suppressed, or that shutdown converts an unsettled success into an error.
"""

from __future__ import annotations

import asyncio

import pytest

from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator, _DeferredTerminal
from vllm_omni.experimental.ar_diffusion.stage_lifecycle import (
    DiffusionStageLifecycleCoordinator,
    DiffusionStageLifecycleTopology,
    SessionControls,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ENCODE, DENOISE, DECODE = 0, 1, 2


class FakeWorker:
    """A state-owning stage worker with the real lifecycle reply contract."""

    def __init__(self, stage_id: int, *, state_owning: bool = True) -> None:
        self.stage_id = stage_id
        self.state_owning = state_owning
        self.sessions: set[str] = set()
        self.fail_close: set[str] = set()
        self.block_close = asyncio.Event()
        self.block_close.set()

    async def rpc(self, method: str, args: tuple):
        if method == "register_ar_diffusion_generation":
            return True
        if method == "close_ar_diffusion_session":
            await self.block_close.wait()
            (session_id,) = args
            if session_id in self.fail_close:
                return {"supported": False, "error": f"stage {self.stage_id} cannot clean up {session_id}"}
            self.sessions.discard(session_id)
            return True
        if method == "get_ar_diffusion_release_events":
            return []
        if method == "ack_ar_diffusion_release_events":
            return len(args[0])
        raise AssertionError(f"unexpected lifecycle RPC {method}")


class FakeOutputQueue:
    """Stands in for the janus async queue the orchestrator publishes to."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    async def put(self, msg: object) -> None:
        self.messages.append(msg)

    def successes(self) -> list[OutputMessage]:
        return [m for m in self.messages if isinstance(m, OutputMessage) and m.finished]

    def errors(self) -> list[ErrorMessage]:
        return [m for m in self.messages if isinstance(m, ErrorMessage)]


def _orchestrator() -> tuple[Orchestrator, FakeOutputQueue, dict[int, FakeWorker]]:
    """A real Orchestrator with only the lifecycle collaborators wired up."""
    workers = {
        ENCODE: FakeWorker(ENCODE),
        DENOISE: FakeWorker(DENOISE),
        DECODE: FakeWorker(DECODE, state_owning=False),
    }
    topology = DiffusionStageLifecycleTopology(
        stage_ids=(ENCODE, DENOISE, DECODE), state_owning_stage_ids=(ENCODE, DENOISE)
    )

    async def rpc(method, stage_id, args):
        return [[await workers[stage_id].rpc(method, args)]]

    orchestrator = Orchestrator.__new__(Orchestrator)
    queue = FakeOutputQueue()
    orchestrator.output_async_queue = queue
    orchestrator.request_states = {}
    orchestrator._deferred_terminals = {}
    orchestrator._session_lifecycle_tasks = set()
    orchestrator._session_lifecycle = DiffusionStageLifecycleCoordinator(topology, rpc)
    return orchestrator, queue, workers


def _terminal(request_id: str, *, stage_id: int = DECODE) -> OutputMessage:
    return OutputMessage(
        request_id=request_id,
        stage_id=stage_id,
        replica_id=0,
        engine_outputs=object(),
        metrics=None,
        finished=True,
    )


async def _admit(orchestrator: Orchestrator, request_id: str, session_id: str, **kwargs) -> None:
    await orchestrator._session_lifecycle.admit(request_id, SessionControls(session_id, **kwargs))


def test_a_terminal_success_waits_for_settlement():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        workers[DENOISE].sessions.add("A")

        await orchestrator._emit_output(_terminal("r0"))
        # Held back: nothing has confirmed the cleanup this finish implies.
        assert queue.successes() == []
        assert "r0" in orchestrator._deferred_terminals

        await orchestrator._complete_session_lifecycle(["r0"], success=True)

    asyncio.run(scenario())

    assert len(queue.successes()) == 1
    assert queue.errors() == []
    assert orchestrator._deferred_terminals == {}


def test_a_failed_settlement_replaces_the_success_with_one_lifecycle_error():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True, close_session=True)
        workers[ENCODE].sessions.add("A")
        workers[ENCODE].fail_close.add("A")

        await orchestrator._emit_output(_terminal("r0"))
        await orchestrator._complete_session_lifecycle(["r0"], success=True)

    asyncio.run(scenario())

    assert queue.successes() == []
    errors = queue.errors()
    assert len(errors) == 1
    assert errors[0].error_type == "session_lifecycle_error"
    assert errors[0].request_id == "r0"
    assert errors[0].stage_id == DECODE
    assert orchestrator._session_lifecycle.blocked_reason is not None


def test_a_duplicate_terminal_cannot_bypass_a_pending_outcome():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")

        await orchestrator._emit_output(_terminal("r0"))
        # A second terminal while the first is pending must not reach the client.
        await orchestrator._emit_output(_terminal("r0", stage_id=DENOISE))
        assert queue.messages == []
        assert len(orchestrator._deferred_terminals) == 1

        await orchestrator._complete_session_lifecycle(["r0"], success=True)

    asyncio.run(scenario())

    # Exactly one terminal outcome, and it is the first one.
    assert len(queue.successes()) == 1
    assert queue.successes()[0].stage_id == DECODE


def test_shutdown_reports_a_lifecycle_error_for_an_unsettled_success():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        await orchestrator._emit_output(_terminal("r0"))
        # Shutdown arrives while the outcome is still pending.
        await orchestrator._finalize_deferred_terminals()

    asyncio.run(scenario())

    assert queue.successes() == []
    errors = queue.errors()
    assert len(errors) == 1
    assert errors[0].error_type == "session_lifecycle_error"
    assert "shut down before session lifecycle cleanup was confirmed" in errors[0].error
    assert orchestrator._deferred_terminals == {}


def test_shutdown_publishes_an_outcome_whose_settlement_was_confirmed():
    orchestrator, queue, _workers = _orchestrator()
    # Settlement succeeded and only its publication was interrupted.
    orchestrator._deferred_terminals["r0"] = _DeferredTerminal(message=_terminal("r0"), settled=True)

    asyncio.run(orchestrator._finalize_deferred_terminals())

    assert len(queue.successes()) == 1
    assert queue.errors() == []


def test_shutdown_finalization_is_idempotent():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        await orchestrator._emit_output(_terminal("r0"))
        await orchestrator._finalize_deferred_terminals()
        await orchestrator._finalize_deferred_terminals()
        # Ordinary cleanup afterwards must not publish a second outcome either.
        await orchestrator._complete_session_lifecycle(["r0"], success=True)

    asyncio.run(scenario())

    assert len(queue.errors()) == 1
    assert queue.successes() == []


def test_settling_twice_publishes_one_outcome_and_releases_the_gate_once():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        await orchestrator._emit_output(_terminal("r0"))
        await orchestrator._complete_session_lifecycle(["r0"], success=True)
        # Re-entering cleanup for the same request is a no-op.
        await orchestrator._complete_session_lifecycle(["r0"], success=True)
        # The gate was released exactly once, so the next request is admitted.
        await _admit(orchestrator, "r1", "B", reset=True)
        await orchestrator._session_lifecycle.complete("r1", success=True)

    asyncio.run(scenario())

    assert len(queue.successes()) == 1
    assert orchestrator._session_lifecycle.is_active("B")


def test_an_already_failed_request_does_not_get_a_second_terminal():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        workers[ENCODE].fail_close.add("A")
        # Inference failed: its error was published by the error path, so nothing
        # is deferred. A cleanup failure here must not add a second terminal.
        await orchestrator._complete_session_lifecycle(["r0"], success=False)

    asyncio.run(scenario())

    assert queue.messages == []
    assert orchestrator._session_lifecycle.blocked_reason is not None


def test_admission_stays_gated_until_the_outcome_is_published():
    orchestrator, queue, workers = _orchestrator()
    order: list[str] = []

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        await orchestrator._emit_output(_terminal("r0"))

        # Park the cleanup RPC so settlement cannot finish.
        workers[ENCODE].block_close.clear()
        workers[ENCODE].fail_close.clear()

        async def next_request():
            await _admit(orchestrator, "r1", "B", reset=True)
            order.append("r1-admitted")
            await orchestrator._session_lifecycle.complete("r1", success=True)

        task = asyncio.create_task(next_request())
        for _ in range(5):
            await asyncio.sleep(0)
        order.append("outcome-pending")
        assert queue.successes() == []

        workers[ENCODE].block_close.set()
        await orchestrator._complete_session_lifecycle(["r0"], success=True)
        order.append("outcome-published")
        await task

    asyncio.run(scenario())

    assert order == ["outcome-pending", "outcome-published", "r1-admitted"]
    assert len(queue.successes()) == 1


def test_a_non_coordinated_request_is_published_immediately():
    orchestrator, queue, _workers = _orchestrator()

    # Never admitted, so the coordinator does not own it.
    asyncio.run(orchestrator._emit_output(_terminal("unrelated")))

    assert len(queue.successes()) == 1
    assert orchestrator._deferred_terminals == {}


def test_a_non_terminal_output_is_never_deferred():
    orchestrator, queue, workers = _orchestrator()

    async def scenario():
        await _admit(orchestrator, "r0", "A", reset=True)
        workers[ENCODE].sessions.add("A")
        chunk = _terminal("r0")
        chunk.finished = False
        await orchestrator._emit_output(chunk)

    asyncio.run(scenario())

    assert len(queue.messages) == 1
    assert orchestrator._deferred_terminals == {}
