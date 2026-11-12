# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A failing inter-stage bridge (``custom_process_input_func``) must fail the one
request it belongs to, not the orchestrator.

Regression tests for the AR->diffusion forward path: a per-request exception raised
while building the next-stage prompt (for example a model's stage input processor
rejecting a user-supplied ``negative_prompt``) must surface as a request-scoped
``ErrorMessage`` while the orchestrator keeps serving other requests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.messages import ShutdownRequestMessage
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    OrchestratorFixture,
    _build_harness,
    _build_request_output,
    _build_stage_pools,
    _engine_core_outputs,
    _enqueue_add_request,
    _wait_for,
)
from .test_orchestrator_error_handling import _sampling_params, _wait_for_error_message

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


async def _drive_bridge_failure(orchestrator_factory, exc: BaseException):
    """Two-stage pipeline (LLM -> diffusion) whose diffusion bridge raises ``exc``."""
    stage0 = FakeStageClient(stage_type="llm", final_output=False, next_inputs=[{"prompt_token_ids": [7, 8]}])
    stage1 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")

    def failing_bridge(*_args, **_kwargs):
        raise exc

    stage1.custom_process_input_func = failing_bridge
    proc0 = FakeOutputProcessor(request_outputs=[_build_request_output("req-x", token_ids=[3], finished=True)])
    proc1 = FakeOutputProcessor()
    stage_pools = _build_stage_pools([[stage0], [stage1]], output_processors=[proc0, proc1])
    fixture = orchestrator_factory([], stage_pools=stage_pools)
    try:
        await _enqueue_add_request(
            fixture,
            request_id="req-x",
            prompt=SimpleNamespace(request_id="req-x", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params(), OmniDiffusionSamplingParams()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        # Completing stage 0 triggers the AR->diffusion bridge, which raises.
        stage0.push_engine_core_outputs(_engine_core_outputs("s0-raw", 1.0))

        error_msg = await _wait_for_error_message(fixture, request_id="req-x")
        return fixture, stage1, error_msg
    except BaseException:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
        raise


@pytest.mark.asyncio
async def test_bridge_client_error_fails_request_not_server(orchestrator_factory) -> None:
    fixture, stage1, error_msg = await _drive_bridge_failure(
        orchestrator_factory, OmniClientError("Ming-Image does not accept negative_prompt")
    )
    try:
        assert error_msg.fatal is False, "a 4xx client error must not be reported as fatal"
        assert error_msg.status_code == 400
        assert "negative_prompt" in error_msg.error
        assert error_msg.stage_id == 1
        assert fixture.thread.is_alive(), "orchestrator thread died on a per-request bridge error"
        assert "req-x" not in fixture.orchestrator.request_states
        assert stage1.add_request_calls == [], "the diffusion stage must not receive the failed request"
    finally:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
