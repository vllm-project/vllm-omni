# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for ErnieImage text-to-image generation.

Equivalent to running:
    vllm serve baidu/ERNIE-Image --omni
"""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "baidu/ERNIE-Image"

# (model, stage_configs_path, extra_omni_kwargs) for ``omni_runner`` indirect parametrize
_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "parallel_config": DiffusionParallelConfig(
            tensor_parallel_size=2,
        ),
        "enable_cpu_offload": True,
    },
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_ernie_image_text_to_image(omni_runner_handler: OmniRunnerHandler) -> None:
    request_config = {
        "model": omni_runner_handler.runner.model_name,
        "prompt": "A photo of a cat sitting on a laptop",
        "sampling_params": OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=4.0,
            seed=42,
        ),
    }
    omni_runner_handler.send_diffusion_request(request_config)
