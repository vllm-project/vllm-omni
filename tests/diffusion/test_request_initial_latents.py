# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pipelines must start from request-supplied initial latents in every execution mode."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

STEP_MODE_PIPELINES = [
    pytest.param(QwenImagePipeline, id="qwen-image"),
    pytest.param(QwenImage21Pipeline, id="qwen-image-2.1"),
]


class _StopAfterCaptureError(Exception):
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
        extra_args={},
    )


def _request_batch(latents) -> DiffusionRequestBatch:
    request = SimpleNamespace(request_id="req-0", prompt="a red bicycle", sampling_params=_sampling(latents))
    return DiffusionRequestBatch(requests=[request])


def _capture_into(captured: dict):
    def _capture(**kwargs):
        captured.update(kwargs)
        raise _StopAfterCaptureError

    return _capture


def _make_qwen_pipeline(pipeline_class: type) -> tuple[Any, dict]:
    pipeline: Any = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.tokenizer_max_length = 1024
    pipeline._max_length = 1024
    captured: dict = {}
    pipeline._prepare_generation_context = _capture_into(captured)
    return pipeline, captured


def _step_mode_latents(pipeline_class: type, latents):
    pipeline, captured = _make_qwen_pipeline(pipeline_class)
    with pytest.raises(_StopAfterCaptureError):
        pipeline.prepare_encode(SimpleNamespace(prompt="a red bicycle", sampling=_sampling(latents)))
    return captured["latents"]


def _request_mode_latents(pipeline_class: type, latents):
    pipeline, captured = _make_qwen_pipeline(pipeline_class)
    with pytest.raises(_StopAfterCaptureError):
        pipeline.forward(_request_batch(latents))
    return captured["latents"]


@pytest.mark.parametrize("pipeline_class", STEP_MODE_PIPELINES)
def test_step_mode_uses_the_latents_request_mode_uses(pipeline_class):
    latents = torch.randn(1, 16, 64)

    step_latents = _step_mode_latents(pipeline_class, latents)

    assert step_latents is latents
    torch.testing.assert_close(step_latents, _request_mode_latents(pipeline_class, latents), rtol=0, atol=0)


@pytest.mark.parametrize("pipeline_class", STEP_MODE_PIPELINES)
def test_step_mode_without_request_latents_leaves_noise_to_the_pipeline(pipeline_class):
    assert _step_mode_latents(pipeline_class, None) is None


def _flux2_prepare_latents_kwargs(latents) -> dict:
    pipeline: Any = object.__new__(Flux2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._execution_device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=64))
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.check_cfg_parallel_validity = lambda *args: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(1, 4, 8), torch.zeros(1, 4, 4))
    captured: dict = {}
    pipeline.prepare_latents = _capture_into(captured)
    with pytest.raises(_StopAfterCaptureError):
        pipeline.forward(_request_batch(latents))
    return captured


def test_flux2_starts_from_request_latents():
    latents = torch.randn(1, 64, 2, 2)

    assert _flux2_prepare_latents_kwargs(latents)["latents"] is latents


def test_flux2_without_request_latents_leaves_noise_to_the_pipeline():
    assert _flux2_prepare_latents_kwargs(None)["latents"] is None
