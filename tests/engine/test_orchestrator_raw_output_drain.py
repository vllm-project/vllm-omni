"""Unit tests for bounded non-blocking raw-output draining in the Orchestrator.

Covers the head-of-line blocking fix from vllm-project/vllm-omni#4561: a single
orchestration round must drain all already-ready raw outputs from an LLM replica
(up to a fairness bound) instead of consuming only one per round.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.orchestrator import Orchestrator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_eco(request_id: str = "r", n_outputs: int = 1) -> SimpleNamespace:
    """An EngineCoreOutputs-like object with ``n_outputs`` engine outputs."""
    outputs = [
        SimpleNamespace(
            request_id=request_id,
            is_segment_finished=False,
            new_prompt_len_snapshot=None,
        )
        for _ in range(n_outputs)
    ]
    return SimpleNamespace(outputs=outputs, scheduler_stats=None)


class _FakePool:
    """Minimal StagePool stand-in scripting a sequence of non-blocking polls."""

    def __init__(self, scripted: list) -> None:
        # Each entry is returned in order; None means "queue empty".
        self._scripted = list(scripted)
        self.processed: list = []

    def poll_llm_raw_output_nowait(self, replica_id: int):
        if not self._scripted:
            return None
        return self._scripted.pop(0)

    async def process_llm_raw_outputs(self, replica_id: int, raw_outputs, iteration_stats=None) -> list:
        self.processed.append(raw_outputs)
        return []


def _make_orchestrator() -> Orchestrator:
    import asyncio

    orch = object.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    orch.request_states = {}
    orch._stat_logger = None

    async def _noop_kv(stage_id, raw_outputs):
        return None

    async def _noop_handle(stage_id, replica_id, outputs):
        return None

    orch._handle_kv_ready_raw_outputs = _noop_kv
    orch._handle_processed_outputs = _noop_handle
    return orch


async def test_drain_processes_all_ready_outputs_in_one_round():
    """Three ready raw outputs then empty: one round drains all three, not one."""
    orch = _make_orchestrator()
    pool = _FakePool([_make_eco("a"), _make_eco("b"), _make_eco("c")])

    did_work = await orch._drain_llm_replica_raw_outputs(0, 0, pool)

    assert did_work is True
    assert len(pool.processed) == 3
    # Queue is drained -> a subsequent round reports no work.
    assert await orch._drain_llm_replica_raw_outputs(0, 0, pool) is False


async def test_drain_is_bounded_per_round():
    """Drain stops at the fairness bound even if more outputs are ready."""
    orch = _make_orchestrator()
    over = Orchestrator._MAX_RAW_OUTPUTS_PER_DRAIN + 5
    pool = _FakePool([_make_eco(f"r{i}") for i in range(over)])

    await orch._drain_llm_replica_raw_outputs(0, 0, pool)

    assert len(pool.processed) == Orchestrator._MAX_RAW_OUTPUTS_PER_DRAIN
    # The remainder is left for the next round (not dropped).
    assert len(pool._scripted) == 5


async def test_drain_skips_empty_outputs_but_keeps_draining():
    """An output-less item must not stop the drain: items behind it still run."""
    orch = _make_orchestrator()
    empty = SimpleNamespace(outputs=[], scheduler_stats=None)
    pool = _FakePool([empty, _make_eco("after-empty")])

    did_work = await orch._drain_llm_replica_raw_outputs(0, 0, pool)

    assert did_work is True
    # Only the non-empty item is processed, but it WAS reached (drain continued).
    assert len(pool.processed) == 1
    assert pool.processed[0].outputs[0].request_id == "after-empty"


async def test_drain_returns_false_on_empty_queue():
    """No ready outputs -> no work, no processing."""
    orch = _make_orchestrator()
    pool = _FakePool([])

    assert await orch._drain_llm_replica_raw_outputs(0, 0, pool) is False
    assert pool.processed == []
