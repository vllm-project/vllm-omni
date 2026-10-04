# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for OmniGen2 request defaults."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _GuidanceResolvedError(Exception):
    pass


def _stop_after_guidance_resolution(*args, **kwargs):
    raise _GuidanceResolvedError


@pytest.mark.parametrize(
    ("sampling_kwargs", "forward_kwargs", "expected_scales"),
    [
        ({}, {}, (4.0, 1.0)),
        ({"guidance_scale": 5.0}, {}, (5.0, 1.0)),
        ({"guidance_scale": 0.0}, {}, (0.0, 1.0)),
        ({"guidance_scale_2": 0.0}, {}, (4.0, 0.0)),
        ({"guidance_scale": 5.0, "guidance_scale_2": 2.0}, {}, (5.0, 2.0)),
        ({}, {"image_guidance_scale": 3.0}, (4.0, 3.0)),
    ],
)
def test_image_guidance_distinguishes_explicit_value_from_omission(sampling_kwargs, forward_kwargs, expected_scales):
    pipeline = object.__new__(OmniGen2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.default_sample_size = 128
    pipeline.vae_scale_factor = 8
    pipeline.device = torch.device("cpu")
    pipeline.encode_prompt = _stop_after_guidance_resolution

    sampling_params = OmniDiffusionSamplingParams(height=64, width=64, **sampling_kwargs)
    request = OmniDiffusionRequest(
        prompt={
            "prompt": "edit the image",
            "negative_prompt": "blurred",
            "additional_information": {"preprocessed_images": [object()]},
        },
        sampling_params=sampling_params,
        request_id="request-0",
    )

    with pytest.raises(_GuidanceResolvedError):
        pipeline.forward(DiffusionRequestBatch([request]), **forward_kwargs)

    assert (pipeline.text_guidance_scale, pipeline.image_guidance_scale) == expected_scales
