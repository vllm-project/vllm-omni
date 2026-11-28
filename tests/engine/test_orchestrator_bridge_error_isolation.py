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

import asyncio
import queue
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.messages import EngineQueueMessage, ErrorMessage, ShutdownRequestMessage
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.ming_image import (
    _CAPTURE_LAYERS,
    _IMAGE_END_TOKEN_ID,
    _IMAGE_PATCH_TOKEN_ID,
    _NUM_QUERY_TOKENS,
    thinker2image,
)

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


# The Ming-Image thinker appends ``<image><imagePatch>*256</image>`` after the
# text prompt; 157158 is the vendor's ``<image>`` opener id.
_IMAGE_OPEN_TOKEN_ID = 157158
_THINKER_HIDDEN_SIZE = 8


def _ming_query_suffix(text_prompt_ids: list[int]) -> list[int]:
    """Prompt token ids as the Ming-Image thinker emits them for image requests."""
    return [
        *text_prompt_ids,
        _IMAGE_OPEN_TOKEN_ID,
        *([_IMAGE_PATCH_TOKEN_ID] * _NUM_QUERY_TOKENS),
        _IMAGE_END_TOKEN_ID,
    ]


def _thinker_hidden_states(prompt_length: int) -> dict[str, torch.Tensor]:
    """Capture-layer fixture shaped like a real Ming-Image thinker output.

    Every tensor is ``(prompt_length, hidden)``: the per-layer contract
    ``thinker2image`` validates before slicing the query and direct blocks.
    """
    multimodal = {
        "final_hidden_states": torch.arange(prompt_length * _THINKER_HIDDEN_SIZE)
        .float()
        .view(prompt_length, _THINKER_HIDDEN_SIZE)
    }
    for layer in _CAPTURE_LAYERS:
        multimodal[f"hidden_states_{layer}"] = torch.full((prompt_length, _THINKER_HIDDEN_SIZE), float(layer))
    return multimodal


def _ming_thinker_output(
    request_id: str,
    *,
    prompt_token_ids: list[int],
    multimodal_output: dict[str, torch.Tensor] | None = None,
):
    """A finished thinker RequestOutput in the shape ``thinker2image`` consumes."""
    output = _build_request_output(
        request_id,
        token_ids=[3],
        prompt_token_ids=list(prompt_token_ids),
        finished=True,
    )
    # AR stages attach hidden states to the completion output's
    # ``multimodal_output``; the bridge reads them back by the same attribute.
    if multimodal_output is not None:
        output.outputs[0].multimodal_output = multimodal_output
    return output


async def _drain_output_until_quiet(
    fixture: OrchestratorFixture, *, quiet: float = 0.3, timeout: float = 2.0
) -> list[EngineQueueMessage]:
    """Drain the output queue until it stays quiet, returning every message seen."""
    drained: list[EngineQueueMessage] = []
    deadline = time.monotonic() + timeout
    quiet_after = time.monotonic() + quiet
    while time.monotonic() < deadline:
        try:
            drained.append(fixture.output_sync_q.get_nowait())
            quiet_after = time.monotonic() + quiet
        except queue.Empty:
            if time.monotonic() >= quiet_after:
                break
            await asyncio.sleep(0.01)
    return drained


@pytest.mark.asyncio
async def test_ming_image_bridge_rejects_malformed_prompt_and_keeps_serving(orchestrator_factory) -> None:
    """The real Ming-Image bridge must contain malformed prompts per request (#11198).

    A thinker output whose prompt tokens cannot hold the image-query suffix is
    client input the bridge rejects while the orchestrator builds the diffusion
    stage's request. That rejection must surface as exactly one request-scoped
    non-fatal 400 for the offending request, while the same orchestrator still
    forwards a follow-up valid Ming-Image request into the diffusion stage.
    """
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    # The production bridge, not a stub: the malformed prompt must be rejected
    # by the same ``thinker2image`` code the serving path dispatches through.
    stage1.custom_process_input_func = thinker2image
    proc0 = FakeOutputProcessor()
    stage_pools = _build_stage_pools([[stage0], [stage1]], output_processors=[proc0, FakeOutputProcessor()])
    fixture = orchestrator_factory([], stage_pools=stage_pools)
    try:
        # --- Bad request: the thinker output carries a text-only prompt, far
        # --- shorter than the 258-token image-query suffix the bridge requires.
        await _enqueue_add_request(
            fixture,
            request_id="req-bad",
            prompt=SimpleNamespace(request_id="req-bad", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[_sampling_params(), OmniDiffusionSamplingParams()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        proc0.request_outputs = [_ming_thinker_output("req-bad", prompt_token_ids=[7, 8, 9])]
        stage0.push_engine_core_outputs(_engine_core_outputs("req-bad", 1.0))

        error_msg = await _wait_for_error_message(fixture, request_id="req-bad")

        assert error_msg.fatal is False, "a malformed Ming-Image prompt must not be reported as fatal"
        assert error_msg.status_code == 400
        assert "shorter than its image-query suffix" in error_msg.error
        assert error_msg.stage_id == 1
        assert fixture.thread.is_alive(), "orchestrator thread died on a malformed Ming-Image prompt"
        assert "req-bad" not in fixture.orchestrator.request_states
        assert stage1.add_request_calls == [], "the diffusion stage must not receive the failed request"

        # --- Valid request on the same orchestrator: the query suffix and the
        # --- capture-layer hidden states let it cross the bridge downstream.
        text_prompt_ids = [7, 8, 9]
        good_prompt_ids = _ming_query_suffix(text_prompt_ids)
        await _enqueue_add_request(
            fixture,
            request_id="req-good",
            prompt=SimpleNamespace(request_id="req-good", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[_sampling_params(), OmniDiffusionSamplingParams()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 2)
        proc0.request_outputs = [
            _ming_thinker_output(
                "req-good",
                prompt_token_ids=good_prompt_ids,
                multimodal_output=_thinker_hidden_states(len(good_prompt_ids)),
            )
        ]
        stage0.push_engine_core_outputs(_engine_core_outputs("req-good", 2.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        request_id, diffusion_request, _params = stage1.add_request_calls[0]
        assert request_id == "req-good"
        extra = diffusion_request["extra"]
        assert extra["query_hidden_states"].shape == (_NUM_QUERY_TOKENS, _THINKER_HIDDEN_SIZE)
        assert extra["direct_hidden_states"].shape == (
            len(text_prompt_ids),
            len(_CAPTURE_LAYERS) * _THINKER_HIDDEN_SIZE,
        )
        assert extra["num_layers"] == 1
        # The direct block concatenates the capture layers over the text prefix,
        # so each third of it is filled with that layer's constant.
        for i, layer in enumerate(_CAPTURE_LAYERS):
            block = extra["direct_hidden_states"][:, i * _THINKER_HIDDEN_SIZE : (i + 1) * _THINKER_HIDDEN_SIZE]
            assert set(block.unique().tolist()) == {float(layer)}
        assert fixture.thread.is_alive(), "orchestrator thread died while serving the follow-up request"

        # --- Exactly one request-scoped 400 overall: the failure above must not
        # --- duplicate, escalate to a fatal error, or touch the valid request.
        drained = await _drain_output_until_quiet(fixture)
        stray_errors = [msg for msg in drained if isinstance(msg, ErrorMessage)]
        assert stray_errors == [], f"unexpected extra error messages after the request-scoped 400: {stray_errors}"
    finally:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
