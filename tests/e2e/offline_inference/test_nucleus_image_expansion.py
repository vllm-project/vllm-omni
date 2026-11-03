# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""End-to-end test for Nucleus-Image text-to-image generation."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "NucleusAI/Nucleus-Image"

_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "parallel_config": DiffusionParallelConfig(
            tensor_parallel_size=2,
        ),
    },
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.slow,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
def test_nucleus_image_text_to_image(offline_client: OfflineOmniClient) -> None:
    request_config = {
        "model": MODEL,
        "prompt": "A cat holding a sign that says hello world",
        "sampling_params": OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=2,
            guidance_scale=4.0,
            seed=42,
        ),
    }
    offline_client.send_diffusion_request(request_config)
