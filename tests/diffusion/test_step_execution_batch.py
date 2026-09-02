# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the step-execution -> request-batch forward bridge.

Covers ``vllm_omni.diffusion.worker.step_execution_batch``, which lets any
pipeline implementing ``SupportsStepExecution`` gain a request-batch
``forward()`` without a model-specific merge/scatter implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.worker.step_execution_batch import (
    StepExecutionRequestBatchMixin,
    run_step_execution_to_completion,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# The bridge under test only reads a handful of attributes off its request and
# sampling-params arguments (``request_id``, ``sampling_params``, ``prompt``,
# ``kv_sender_info``, ``prepared_layout``, and ``sampling_params.seed`` /
# ``.generator`` / ``.generator_device``) - it never checks their concrete
# types. Real ``OmniDiffusionRequest`` / ``OmniDiffusionSamplingParams`` /
# ``DiffusionRequestBatch`` pull in vllm-omni's full diffusion stack, which is
# pinned to a bleeding-edge vllm dev build unavailable here, so duck-type the
# same attribute contract instead of standing up that whole dependency chain.
@dataclass
class _FakeSamplingParams:
    seed: int | None = None
    generator: Any = None
    generator_device: str | None = None


@dataclass
class _FakeRequest:
    prompt: str
    sampling_params: _FakeSamplingParams
    request_id: str
    kv_sender_info: dict | None = None
    prepared_layout: Any | None = None


@dataclass
class _FakeRequestBatch:
    requests: list[_FakeRequest] = field(default_factory=list)


@dataclass
class _FakeDiffusionOutput:
    output: Any
    stage_durations: dict | None = None


class _FakeStepPipeline(StepExecutionRequestBatchMixin):
    """Minimal step-execution pipeline stub.

    Each request's initial latent is its seed (so per-request identity is
    checkable end to end) and each request denoises for ``seed % 3 + 1``
    steps, so a batch mixes requests that finish at different times -
    exactly the case the bridge's ``pending`` list must handle correctly.
    """

    device = torch.device("cpu")
    supports_step_execution = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0
        self.max_states_per_denoise_call = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        seed = state.sampling.seed or 0
        num_steps = seed % 3 + 1
        state.timesteps = [torch.tensor(float(num_steps - i)) for i in range(num_steps)]
        state.latents = torch.tensor([[float(seed)]])
        return state

    def denoise_step(self, input_batch, *, states, **kwargs):
        del kwargs
        self.denoise_calls += 1
        self.max_states_per_denoise_call = max(self.max_states_per_denoise_call, len(states))
        # "Denoise" by adding 1.0 to each row's latent.
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del kwargs
        self.scheduler_calls += 1
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        return _FakeDiffusionOutput(output=state.latents.clone())


class _ChunkedStepPipeline(_FakeStepPipeline):
    def prepare_encode(self, state, **kwargs):
        state = super().prepare_encode(state, **kwargs)
        state.chunk_num_steps = 1
        return state


def _make_request(request_id: str, seed: int) -> _FakeRequest:
    return _FakeRequest(
        prompt=f"prompt-{request_id}",
        sampling_params=_FakeSamplingParams(seed=seed),
        request_id=request_id,
    )


def test_run_step_execution_to_completion_orders_outputs_like_input():
    pipeline = _FakeStepPipeline()
    requests = [_make_request("c", seed=2), _make_request("a", seed=0), _make_request("b", seed=1)]

    outputs = run_step_execution_to_completion(pipeline, requests)

    assert [out.output.item() for out in outputs] == [2.0 + 3, 0.0 + 1, 1.0 + 2]
    assert pipeline.prepare_calls == 3
    assert pipeline.decode_calls == 3


def test_run_step_execution_to_completion_stops_finished_requests_early():
    # seeds 0 and 3 each denoise for 1 step; seed 1 denoises for 2 steps.
    pipeline = _FakeStepPipeline()
    requests = [_make_request("short", seed=0), _make_request("long", seed=1)]

    outputs = run_step_execution_to_completion(pipeline, requests)

    assert outputs[0].output.item() == 1.0  # 0 + one denoise step
    assert outputs[1].output.item() == 3.0  # 1 + two denoise steps
    # Two denoise waves total: both requests together, then just "long".
    assert pipeline.denoise_calls == 2
    assert pipeline.max_states_per_denoise_call == 2


def test_run_step_execution_to_completion_empty_batch():
    assert run_step_execution_to_completion(_FakeStepPipeline(), []) == []


def test_run_step_execution_to_completion_rejects_chunked_pipelines():
    pipeline = _ChunkedStepPipeline()
    requests = [_make_request("a", seed=0)]

    with pytest.raises(NotImplementedError):
        run_step_execution_to_completion(pipeline, requests)


def test_step_execution_request_batch_mixin_forward_matches_helper():
    pipeline = _FakeStepPipeline()
    requests = [_make_request("a", seed=0), _make_request("b", seed=2)]

    assert pipeline.supports_request_batch is True
    outputs = pipeline.forward(_FakeRequestBatch(requests=requests))

    assert [out.output.item() for out in outputs] == [1.0, 5.0]


def test_run_step_execution_to_completion_raises_on_row_mismatch():
    class _BadRowCountPipeline(_FakeStepPipeline):
        def denoise_step(self, input_batch, *, states, **kwargs):
            del kwargs
            self.denoise_calls += 1
            # Drop a row so the returned noise_pred under-counts requests.
            return (
                torch.ones_like(input_batch.latents)[:-1] if len(states) > 1 else torch.ones_like(input_batch.latents)
            )

    pipeline = _BadRowCountPipeline()
    requests = [_make_request("a", seed=0), _make_request("b", seed=0)]

    with pytest.raises(ValueError, match="noise_pred rows"):
        run_step_execution_to_completion(pipeline, requests)
