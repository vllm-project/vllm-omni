# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Terminal audio must not retire a request before upstream terminal metrics."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def build():
    o = Orchestrator.__new__(Orchestrator)
    o.stage_pools = [
        SimpleNamespace(
            final_output=False, build_stage_metrics=lambda *a, **k: SimpleNamespace(stage_id=0, num_tokens_out=42)
        ),
        SimpleNamespace(
            final_output=True, build_stage_metrics=lambda *a, **k: SimpleNamespace(stage_id=1, num_tokens_out=0)
        ),
    ]
    state = OrchestratorRequestState(
        request_id="r",
        final_stage_id=1,
        final_output_stage_ids={1},
        stage_submit_ts={0: 0, 1: 0},
        sampling_params_list=[None, None],
    )
    o.request_states = {"r": state}
    o.output_async_queue = asyncio.Queue()
    o.async_chunk = True
    o._pd_pair = None
    o._cfg_tracker = MagicMock()
    o._cfg_tracker.is_companion.return_value = False
    o._cfg_tracker.cleanup_parent.return_value = []
    o._is_duplex_session_request = lambda s: False
    o._duplex_output_decision = lambda *a: None
    o._stage_receives_async_chunks = lambda s: True
    cleaned = []

    async def cleanup(ids):
        cleaned.extend(ids)
        for r in ids:
            o.request_states.pop(r, None)

    o._cleanup_request_ids = cleanup
    return o, state, cleaned


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
async def test_terminal_metrics_precede_final_audio_for_either_arrival_order(order):
    o, state, cleaned = build()
    for stage in order:
        await o._handle_processed_outputs(stage, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
        if stage == 1 and not state.finished_stage_ids.issuperset({0, 1}):
            assert o.output_async_queue.empty()
            assert "r" in o.request_states
            assert cleaned == []
    metrics = o.output_async_queue.get_nowait()
    final = o.output_async_queue.get_nowait()
    assert type(metrics).__name__ == "StageMetricsMessage"
    assert metrics.stage_id == 0 and metrics.metrics.num_tokens_out == 42
    assert type(final).__name__ == "OutputMessage" and final.finished
    assert o.output_async_queue.empty()
    assert cleaned == ["r"]
    assert state.pending_final_output is None


@pytest.mark.asyncio
async def test_intermediate_audio_is_not_delayed():
    o, state, cleaned = build()
    await o._handle_processed_outputs(1, 0, [SimpleNamespace(request_id="r", finished=False, error=None)])
    message = o.output_async_queue.get_nowait()
    assert not message.finished
    assert state.pending_final_output is None
    assert cleaned == []


@pytest.mark.asyncio
async def test_non_chunk_route_retains_existing_completion_behavior():
    o, state, cleaned = build()
    o.async_chunk = False
    await o._handle_processed_outputs(1, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
    assert o.output_async_queue.get_nowait().finished
    assert cleaned == ["r"]


@pytest.mark.asyncio
async def test_raw_final_fallback_does_not_overtake_pending_real_output():
    o, state, cleaned = build()
    await o._handle_processed_outputs(1, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
    await o._finish_raw_terminal_requests(1, 0, {"r"})
    assert o.output_async_queue.empty()
    assert cleaned == []
    assert state.pending_final_output is not None
    await o._handle_processed_outputs(0, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
    assert o.output_async_queue.get_nowait().stage_id == 0
    assert o.output_async_queue.get_nowait().finished
    assert cleaned == ["r"]


@pytest.mark.asyncio
async def test_raw_upstream_terminal_releases_pending_output_without_deadlock():
    o, state, cleaned = build()
    await o._handle_processed_outputs(1, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
    terminal = SimpleNamespace(finish_reason="stop", is_segment_finished=False)
    assert await o._apply_raw_terminal_stage_finish(0, terminal, state)
    await o._finish_raw_terminal_requests(0, 0, {"r"})
    assert o.output_async_queue.get_nowait().finished
    assert cleaned == ["r"]


@pytest.mark.asyncio
async def test_cancelled_pending_output_is_not_published_by_late_upstream():
    o, state, cleaned = build()
    await o._handle_processed_outputs(1, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
    assert state.pending_final_output is not None
    await o._cleanup_request_ids(["r"])
    await o._handle_processed_outputs(0, 0, [SimpleNamespace(request_id="r", finished=True, error=None)])
    assert o.output_async_queue.empty()
    assert cleaned == ["r"]
