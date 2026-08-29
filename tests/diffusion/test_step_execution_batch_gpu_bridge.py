# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU variant of test_step_execution_batch.py: same fake step-execution
pipeline and assertions, but every tensor lives on CUDA, so the bridge's use
of the real ``InputBatch``/``scatter_latents`` gather-scatter machinery is
exercised with actual device transfers instead of CPU tensors that happen to
work regardless of the declared device.
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

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


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


class _FakeCudaStepPipeline(StepExecutionRequestBatchMixin):
    """Same contract as the CPU fake pipeline, but every tensor is placed on
    ``self.device`` explicitly, so a bridge bug that silently mixes host and
    device tensors (e.g. ``torch.ones_like`` losing the device, a ``.cpu()``
    dropped in scatter) would surface as a RuntimeError here instead of
    passing by accident on a CPU-only run."""

    supports_step_execution = True

    def __init__(self, device: torch.device):
        self.device = device
        self.max_states_per_denoise_call = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        seed = state.sampling.seed or 0
        num_steps = seed % 3 + 1
        state.timesteps = [torch.tensor(float(num_steps - i), device=self.device) for i in range(num_steps)]
        state.latents = torch.tensor([[float(seed)]], device=self.device)
        return state

    def denoise_step(self, input_batch, *, states, **kwargs):
        del kwargs
        assert input_batch.latents.is_cuda
        self.max_states_per_denoise_call = max(self.max_states_per_denoise_call, len(states))
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del kwargs
        assert noise_pred.is_cuda
        assert state.latents.is_cuda
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        assert state.latents.is_cuda
        return _FakeDiffusionOutput(output=state.latents.clone())


def _make_request(request_id: str, seed: int) -> _FakeRequest:
    return _FakeRequest(
        prompt=f"prompt-{request_id}",
        sampling_params=_FakeSamplingParams(seed=seed),
        request_id=request_id,
    )


def test_bridge_runs_real_input_batch_gather_scatter_on_cuda():
    device = _require_cuda()
    pipeline = _FakeCudaStepPipeline(device)
    # Different step counts, like the CPU test: exercises the "pending" list
    # dropping an early-finished request mid-batch, with real CUDA tensors
    # flowing through InputBatch.make_batch / scatter_latents.
    requests = [_make_request("short", seed=0), _make_request("long", seed=1)]

    outputs = run_step_execution_to_completion(pipeline, requests)

    assert outputs[0].output.item() == 1.0
    assert outputs[1].output.item() == 3.0
    assert outputs[0].output.is_cuda and outputs[1].output.is_cuda
    assert pipeline.max_states_per_denoise_call == 2


def test_step_execution_request_batch_mixin_forward_on_cuda():
    device = _require_cuda()
    pipeline = _FakeCudaStepPipeline(device)
    requests = [_make_request("a", seed=0), _make_request("b", seed=2)]

    outputs = pipeline.forward(_FakeRequestBatch(requests=requests))

    assert [out.output.item() for out in outputs] == [1.0, 5.0]
    assert all(out.output.is_cuda for out in outputs)
