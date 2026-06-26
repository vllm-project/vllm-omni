"""Regression tests for ``StagePool.process_llm_raw_outputs`` iteration stats.

Guards the fix where ``process_llm_raw_outputs`` previously did
``iteration_stats = IterationStats()`` unconditionally, clobbering the
caller-supplied object. The orchestrator builds its own ``IterationStats``,
passes it in, then logs metrics off that SAME object via the stat logger --
so the unconditional reassignment caused per-(stage, replica) iteration
metrics to be recorded empty. The fix is
``iteration_stats = iteration_stats or IterationStats()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingOutputProcessor:
    """Minimal output processor that records the ``iteration_stats`` it received.

    Mirrors the ``FakeOutputProcessor`` shape in ``test_orchestrator.py`` but
    captures the third positional arg passed to ``process_outputs`` so the test
    can assert it is the caller's object (by identity) rather than a throwaway
    local allocated inside ``process_llm_raw_outputs``.
    """

    def __init__(self) -> None:
        self.received_iteration_stats: Any = None
        self.process_outputs_calls = 0

    def process_outputs(self, _outputs, _timestamp, iteration_stats):
        self.process_outputs_calls += 1
        self.received_iteration_stats = iteration_stats
        # Empty request_outputs keeps record_output_timestamps a no-op.
        return SimpleNamespace(request_outputs=[], reqs_to_abort=[])

    def update_scheduler_stats(self, _scheduler_stats) -> None:
        return None


class _FakeLLMClient:
    """Non-None client stub; only needs to exist as a live replica slot.

    ``process_llm_raw_outputs`` casts the slot to a StagePoolLLMClient but, with
    no reqs_to_abort and a None scheduler_stats below, never calls into it.
    """


def _raw_outputs() -> SimpleNamespace:
    """Stand-in for EngineCoreOutputs with the attributes the method reads."""
    return SimpleNamespace(outputs=["tok"], timestamp=123.0, scheduler_stats=None)


def _build_pool(processor: _RecordingOutputProcessor) -> StagePool:
    return StagePool(
        stage_id=0,
        clients=[_FakeLLMClient()],
        output_processor=processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )


async def test_caller_iteration_stats_not_clobbered() -> None:
    """A caller-supplied IterationStats must reach process_outputs unchanged."""
    processor = _RecordingOutputProcessor()
    pool = _build_pool(processor)
    caller_stats = IterationStats()

    await pool.process_llm_raw_outputs(0, _raw_outputs(), iteration_stats=caller_stats)

    assert processor.process_outputs_calls == 1
    # Identity check: the exact object the caller passed, not a fresh one.
    assert processor.received_iteration_stats is caller_stats


async def test_none_iteration_stats_gets_fresh_object() -> None:
    """When the caller passes None, the `or IterationStats()` path allocates one."""
    processor = _RecordingOutputProcessor()
    pool = _build_pool(processor)

    await pool.process_llm_raw_outputs(0, _raw_outputs(), iteration_stats=None)

    assert processor.process_outputs_calls == 1
    assert processor.received_iteration_stats is not None
    assert isinstance(processor.received_iteration_stats, IterationStats)
