# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E smoke test for Wan2.1 AutoRound MXFP4 inference."""

import os

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

QUANTIZED_MODEL = os.environ.get(
    "WAN21_AUTOROUND_MXFP4_MODEL",
    "INCModel/Wan2.1-T2V-1.3B-Diffusers-MXFP4-AutoRound",
)

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"xpu": "B60"})
@pytest.mark.parametrize(
    "omni_runner",
    [(QUANTIZED_MODEL, None)],
    indirect=True,
)
def test_wan21_autoround_mxfp4_generates_video(omni_runner, omni_runner_handler):
    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_frames=5,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )
    response = omni_runner_handler.send_diffusion_request(
        {
            "prompt": "A red toy car driving on a table",
            "sampling_params": sampling_params,
        }
    )

    assert response.success, response.error_message
    assert response.images is not None and len(response.images) == 1
    frames = np.asarray(response.images[0])
    assert frames.shape[1:4] == (5, 256, 256)
    assert frames.std() > 0.01
