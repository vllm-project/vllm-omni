# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for error propagation paths within the Orchestrator.

Covers:
- EngineDeadError from an LLM stage poll → fatal error broadcast + shutdown
- Diffusion stage error output (OmniRequestOutput.from_error) → routed correctly
"""

from __future__ import annotations

import asyncio
import queue
import time
from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

from .test_orchestrator import (
    FakeStageClient,
    OrchestratorFixture,
    _build_harness,
    _enqueue_add_request,
    _wait_for,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sampling_params(max_tokens: int = 4):
    from vllm.sampling_params import SamplingParams

    return SamplingParams(max_tokens=max_tokens)


async def _get_any_output_message(fixture: OrchestratorFixture, *, timeout: float = 2.0) -> dict:
    """Like _get_output_message but returns any message type (including errors)."""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            return fixture.output_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)


@pytest.fixture
def orchestrator_factory():
    fixtures: list[OrchestratorFixture] = []

    def _factory(*args, **kwargs) -> OrchestratorFixture:
        fixture = _build_harness(*args, **kwargs)
        fixtures.append(fixture)
        return fixture

    yield _factory

    for fixture in fixtures:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait({"type": "shutdown"})
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


# ───────── EngineDeadError from LLM stage poll ─────────


class FakeDeadLLMStageClient(FakeStageClient):
    """LLM stage client that raises EngineDeadError on get_output_async."""

    async def get_output_async(self):
        raise EngineDeadError("Stage-0 engine core is dead")


@pytest.mark.asyncio
async def test_engine_dead_error_broadcasts_fatal_and_shuts_down(orchestrator_factory) -> None:
    """When a stage raises EngineDeadError during poll, the orchestrator must:
    1. Enqueue a fatal error message for each affected request
    2. Shut itself down (thread exits)
    """
    stage0 = FakeDeadLLMStageClient(stage_type="llm", final_output=True)
    orchestrator_fixture = orchestrator_factory([stage0])
    request = SimpleNamespace(request_id="req-dead", prompt_token_ids=[1, 2])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-dead",
            prompt=request,
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )

        # Collect the fatal error message.
        msg = await _get_any_output_message(orchestrator_fixture)

        assert msg["type"] == "error"
        assert msg["fatal"] is True
        assert msg["request_id"] == "req-dead"
        assert "Stage-0 engine core is dead" in msg["error"]

        # The orchestrator thread should exit after the fatal error.
        orchestrator_fixture.thread.join(timeout=5)
        assert not orchestrator_fixture.thread.is_alive()

        # Request state should be cleaned up.
        assert "req-dead" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait({"type": "shutdown"})
            orchestrator_fixture.thread.join(timeout=5)


# ───────── Diffusion stage error output routing ─────────


@pytest.mark.asyncio
async def test_diffusion_error_output_routed_as_finished(orchestrator_factory) -> None:
    """When a diffusion stage returns an OmniRequestOutput with a non-None
    error, the orchestrator must route it as a finished output message and
    clean up the request state.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-err",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push an error output from the diffusion stage.
        stage0.push_diffusion_output(OmniRequestOutput.from_error("req-err", "gpu fault"))

        msg = await _get_any_output_message(orchestrator_fixture)

        assert msg["type"] == "output"
        assert msg["request_id"] == "req-err"
        assert msg["finished"] is True
        assert msg["engine_outputs"].error == "gpu fault"

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-err" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait({"type": "shutdown"})
        orchestrator_fixture.thread.join(timeout=5)
