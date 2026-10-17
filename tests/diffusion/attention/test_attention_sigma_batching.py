# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.attention.test_attention_schedule_context import (
    _make_runner,
    _names,
    _new_requests,
    _request,
    _runner_config,
)
from tests.diffusion.attention.test_attention_schedule_context import (
    attention_env as _attention_env,
)
from tests.diffusion.attention.test_attention_schedule_context import (
    runner_platform as _runner_platform,
)
from tests.diffusion.test_diffusion_step_pipeline import _StepPipeline
from vllm_omni.diffusion.attention.schedule import resolve_batch_attention_sigma_schedule
from vllm_omni.diffusion.data import AttentionScheduleConfig, DiffusionOutput
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin, get_forward_context
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

attention_env = _attention_env
runner_platform = _runner_platform


def _service():
    return AttentionScheduleConfig(
        profiles={"low": {"default": "TORCH_SDPA"}, "high": {"default": "FLASH_ATTN"}},
        sigma=[
            {"low": 0.0, "high": 0.3, "profile": "low"},
            {"low": 0.3, "high": 1.0, "profile": "high"},
        ],
    )


class _SigmaStepPipeline(DenoiseProgressMixin, _StepPipeline):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.calls = []
        self.predictions = []

    def prepare_encode(self, state, **kwargs):
        state = super().prepare_encode(state, **kwargs)
        state.timesteps = [torch.tensor(float(step)) for step in range(3)]
        # Both requests are at step 1, but their model-private trajectories
        # (different step counts/flow shifts in real models) select different windows.
        sigma = 0.2 if state.request_id == "req-low" else 0.8
        state.scheduler = SimpleNamespace(sigmas=[1.0, sigma, 0.1, 0.0])
        state.step_index = 1
        return state

    def denoise_step(self, input_batch, *, states=None, **kwargs):
        assert len(states) == 1, "mixed noise profiles must not share a forward"
        (state,) = states
        self.record_denoise_step(state.step_index, scheduler=state.scheduler, total_steps=state.total_steps)
        ctx = get_forward_context()
        self.calls.append(
            (state.request_id, ctx.denoise_step_idx, ctx.denoise_sigma, self.layer.effective_attention()[0])
        )
        return torch.full_like(input_batch.latents, ctx.denoise_sigma)

    def step_scheduler(self, state, noise_pred, **kwargs):
        self.predictions.append((state.request_id, noise_pred.clone()))
        return super().step_scheduler(state, noise_pred, **kwargs)


def test_runner_splits_equal_step_different_noise_and_preserves_row_order(attention_env, runner_platform):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _SigmaStepPipeline(layer)
    runner = _make_runner(pipeline, config)
    output = runner.execute_stepwise(_new_requests(_request("req-high"), _request("req-low")))
    assert [item[0:3] for item in pipeline.calls] == [("req-high", 1, 0.8), ("req-low", 1, 0.2)]
    assert [item[-1] for item in pipeline.calls] == [
        layer._schedule_candidates["high"].impl,
        layer._schedule_candidates["low"].impl,
    ]
    assert [request_id for request_id, _pred in pipeline.predictions] == ["req-high", "req-low"]
    torch.testing.assert_close(pipeline.predictions[0][1], torch.tensor([0.8]))
    torch.testing.assert_close(pipeline.predictions[1][1], torch.tensor([0.2]))
    assert output is not None


class _SigmaRequestPipeline(DenoiseProgressMixin):
    supports_request_batch = True

    def __init__(self, layer):
        self.layer = layer
        self.selected = []

    def forward(self, batch):
        self.selected.append(("encode", self.layer.effective_attention()[0]))
        for i, sigma in enumerate([1.0, 0.3, 0.2]):
            self.record_denoise_step(i, normalized_sigma=sigma, total_steps=3)
            self.selected.append((i, self.layer.effective_attention()[0]))
        self.record_denoise_step(None)
        self.selected.append(("decode", self.layer.effective_attention()[0]))
        return [DiffusionOutput(output={"request_id": req.request_id}) for req in batch.requests]


def test_request_mode_binds_sigma_only_inside_denoising(attention_env, runner_platform):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _SigmaRequestPipeline(layer)
    runner = _make_runner(pipeline, config)
    output = DiffusionModelRunner.execute_model(runner, _request("req-a", steps=3))
    assert output.error is None
    assert _names(layer, pipeline.selected) == [
        ("encode", "base"),
        (0, "high"),
        (1, "high"),
        (2, "low"),
        ("decode", "base"),
    ]


def test_request_mode_rejects_a_mixed_sigma_batch():
    first, second = _request("a"), _request("b")
    second.sampling_params.attention_sigma_schedule = []
    with pytest.raises(ValueError, match="identical"):
        resolve_batch_attention_sigma_schedule([first, second], _runner_config(_service()))


@pytest.mark.parametrize("mode", ["request", "step"])
def test_scheduler_keys_include_immutable_sigma_windows(mode):
    first, second = _request("a"), _request("b")
    first.sampling_params.attention_sigma_schedule = [{"low": 0.0, "high": 0.3, "profile": "low"}]
    second.sampling_params.attention_sigma_schedule = [{"low": 0.3, "high": 1.0, "profile": "high"}]
    build = build_request_batch_sampling_params_key if mode == "request" else StepScheduler()._build_sampling_params_key
    first_key, second_key = build(first), build(second)
    assert first_key != second_key
    assert isinstance(first_key.attention_sigma_schedule, tuple)
    first.sampling_params.attention_sigma_schedule[0]["high"] = 0.1
    assert first_key.attention_sigma_schedule[0].high == 0.3
