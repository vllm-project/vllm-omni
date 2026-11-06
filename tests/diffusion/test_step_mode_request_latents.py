# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Step mode must start from request-supplied initial latents, as request mode does."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

PIPELINES = [
    pytest.param(QwenImagePipeline, id="qwen-image"),
    pytest.param(QwenImage21Pipeline, id="qwen-image-2.1"),
]


class _StopAfterContextError(Exception):
    pass


def _sampling(latents=None):
    return SimpleNamespace(
        height=32,
        width=32,
        num_inference_steps=2,
        sigmas=None,
        max_sequence_length=None,
        num_outputs_per_prompt=1,
        generator=None,
        latents=latents,
        true_cfg_scale=None,
        guidance_scale_provided=False,
        guidance_scale=1.0,
        output_type=None,
    )


def _make_pipeline(pipeline_class: type) -> tuple[Any, dict]:
    pipeline: Any = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.tokenizer_max_length = 1024
    pipeline._max_length = 1024
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise _StopAfterContextError

    pipeline._prepare_generation_context = _capture
    return pipeline, captured


def _step_mode_latents(pipeline_class: type, latents):
    pipeline, captured = _make_pipeline(pipeline_class)
    with pytest.raises(_StopAfterContextError):
        pipeline.prepare_encode(SimpleNamespace(prompt="a red bicycle", sampling=_sampling(latents)))
    return captured["latents"]


def _request_mode_latents(pipeline_class: type, latents):
    pipeline, captured = _make_pipeline(pipeline_class)
    request = SimpleNamespace(request_id="req-0", prompt="a red bicycle", sampling_params=_sampling(latents))
    with pytest.raises(_StopAfterContextError):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))
    return captured["latents"]


@pytest.mark.parametrize("pipeline_class", PIPELINES)
def test_step_mode_uses_the_latents_request_mode_uses(pipeline_class):
    latents = torch.randn(1, 16, 64)

    step_latents = _step_mode_latents(pipeline_class, latents)

    assert step_latents is latents
    torch.testing.assert_close(step_latents, _request_mode_latents(pipeline_class, latents), rtol=0, atol=0)


@pytest.mark.parametrize("pipeline_class", PIPELINES)
def test_step_mode_without_request_latents_leaves_noise_to_the_pipeline(pipeline_class):
    assert _step_mode_latents(pipeline_class, None) is None
