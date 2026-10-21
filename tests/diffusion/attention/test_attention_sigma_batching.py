# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.diffusion.attention.test_attention_schedule_context import (
    _make_runner,
    _names,
    _request,
    _runner_config,
)
from tests.diffusion.attention.test_attention_schedule_context import (
    attention_env as _attention_env,
)
from tests.diffusion.attention.test_attention_schedule_context import (
    runner_platform as _runner_platform,
)
from vllm_omni.diffusion.attention.schedule import resolve_batch_attention_sigma_schedule
from vllm_omni.diffusion.data import AttentionScheduleConfig, DiffusionOutput
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
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
        resolve_batch_attention_sigma_schedule([first.sampling_params, second.sampling_params], _service())


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
