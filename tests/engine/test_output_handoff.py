# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The orchestrator → serving-loop output handoff (``LoopHandoffQueue``)."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.output_handoff import LoopHandoffQueue

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ProducerLoop:
    """An event loop on its own thread, standing in for the orchestrator's."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="producer", daemon=True)
        self._thread.start()

    def run(self, fn: Callable[[], None]) -> None:
        """Run *fn* as one callback (one loop turn) on the producer loop and wait for it."""

        async def _call() -> None:
            fn()

        asyncio.run_coroutine_threadsafe(_call(), self.loop).result(timeout=5.0)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5.0)
        self.loop.close()


@pytest.fixture
def producer():
    loop = _ProducerLoop()
    yield loop
    loop.close()


async def _settle() -> None:
    # Let wakes the producer scheduled reach this loop.
    for _ in range(5):
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_a_put_outside_any_loop_wakes_at_once() -> None:
    queue: LoopHandoffQueue[str] = LoopHandoffQueue()
    woken = asyncio.Event()
    queue.set_consumer(asyncio.get_running_loop(), woken.set)

    # The orchestrator's failure path puts from its thread after its loop died.
    thread = threading.Thread(target=lambda: queue.sync_q.put_nowait("error"))
    thread.start()
    thread.join(timeout=5.0)

    await asyncio.wait_for(woken.wait(), timeout=5.0)
    assert queue.drain_nowait(64) == ["error"]
    queue.close()


@pytest.mark.asyncio
async def test_puts_racing_a_drain_are_never_lost(producer: _ProducerLoop) -> None:
    queue: LoopHandoffQueue[int] = LoopHandoffQueue()
    woken = asyncio.Event()
    queue.set_consumer(asyncio.get_running_loop(), woken.set)
    total = 2000
    received: list[int] = []

    async def produce() -> None:
        # Twenty producer turns while the consumer drains 64 at a time.
        for start in range(0, total, 100):

            def put_chunk(start: int = start) -> None:
                for index in range(start, min(start + 100, total)):
                    queue.async_q.put_nowait(index)

            await asyncio.to_thread(producer.run, put_chunk)

    producer_task = asyncio.create_task(produce())
    while len(received) < total:
        await asyncio.wait_for(woken.wait(), timeout=5.0)
        woken.clear()
        received.extend(queue.drain_nowait(64))
        if len(received) < total and queue.sync_q.qsize():
            woken.set()
    await producer_task

    assert received == list(range(total))
    queue.close()


def _handoff_engine(alive: bool = True) -> AsyncOmniEngine:
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = LoopHandoffQueue()
    engine.orchestrator_thread = SimpleNamespace(is_alive=lambda: alive)
    engine._event_driven_orch_default = True
    return engine


def _serving_loop(engine: AsyncOmniEngine, route: Callable[[object], bool]):
    from vllm_omni.entrypoints.async_omni_base import AsyncOmniBase

    omni = object.__new__(AsyncOmniBase)
    omni.engine = engine
    omni.final_output_task = None
    omni._route_engine_message = route
    return omni


async def _stop(omni) -> None:
    omni.final_output_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await omni.final_output_task


@pytest.mark.asyncio
async def test_final_output_loop_reads_the_handoff_without_the_drain_thread(
    monkeypatch: pytest.MonkeyPatch, producer: _ProducerLoop
) -> None:
    monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    engine = _handoff_engine()

    async def no_drain_thread(timeout: float = 1.0):
        raise AssertionError("the handoff must replace the drain thread")

    engine.get_output_blocking_async = no_drain_thread
    # Queued before the serving loop attached: still read, and first.
    engine.output_queue.sync_q.put_nowait("early")
    routed: list[object] = []
    done = asyncio.Event()

    def route(msg: object) -> bool:
        routed.append(msg)
        if len(routed) == 1 + 150:
            done.set()
        return True

    omni = _serving_loop(engine, route)
    try:
        omni._final_output_handler()
        await _settle()

        def put_turn() -> None:
            # More than one batch in one producer turn.
            for index in range(150):
                engine.output_queue.async_q.put_nowait(index)

        producer.run(put_turn)
        await asyncio.wait_for(done.wait(), timeout=5.0)
        assert routed == ["early", *range(150)]
    finally:
        await _stop(omni)
        engine.output_queue.close()


@pytest.mark.asyncio
async def test_final_output_loop_handoff_reports_a_dead_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.entrypoints.async_omni_base as async_omni_base

    monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    monkeypatch.setattr(async_omni_base, "_FINAL_OUTPUT_BLOCKING_WAIT_S", 0.05)
    engine = _handoff_engine(alive=False)
    dead: list[str] = []
    omni = _serving_loop(engine, lambda msg: True)
    omni.request_states = {}
    omni._on_engine_dead = dead.append
    try:
        omni._final_output_handler()
        await asyncio.wait_for(omni.final_output_task, timeout=5.0)
    finally:
        engine.output_queue.close()
    assert dead and "Orchestrator died" in dead[0]
