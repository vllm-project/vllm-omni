# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.deepseek_janus.pipeline_janus_vq import JanusVQDecodePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeVQModel:
    def decode_code(self, tokens, shape):
        assert tuple(tokens.shape) == (1, 576)
        assert shape == [1, 8, 24, 24]
        return torch.zeros((1, 3, 384, 384), dtype=torch.float32)


def _build_pipeline() -> JanusVQDecodePipeline:
    pipe = JanusVQDecodePipeline.__new__(JanusVQDecodePipeline)
    nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe._vq_model = _FakeVQModel()
    pipe._stage_durations = {}
    return pipe


def test_vq_pipeline_reads_image_tokens_from_prompt_extra() -> None:
    pipe = _build_pipeline()
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "p", "extra": {"image_tokens": list(range(576)), "img_size": 384}}],
        sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1),
        request_id="req-1",
    )

    output = pipe.forward(req)

    assert output.error is None
    assert output.aborted is False
    assert output.custom_output["num_images"] == 1
    assert output.output is not None


def test_vq_pipeline_prefers_sampling_params_extra_step_kwargs() -> None:
    pipe = _build_pipeline()
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "p", "extra": {"image_tokens": [999], "img_size": 128}}],
        sampling_params=OmniDiffusionSamplingParams(
            num_outputs_per_prompt=1,
            extra_step_kwargs={"image_tokens": list(range(576)), "img_size": 384},
        ),
        request_id="req-2",
    )

    output = pipe.forward(req)

    assert output.error is None
    assert output.custom_output["num_images"] == 1
