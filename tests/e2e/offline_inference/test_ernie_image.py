# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for ErnieImage text-to-image generation.

Equivalent to running:
    vllm serve baidu/ERNIE-Image --omni
"""

import pytest
from PIL import Image

from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "baidu/ERNIE-Image"


@pytest.mark.core_model
@pytest.mark.diffusion
def test_ernie_image_text_to_image():
    with OmniRunner(
        MODEL,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
        ),
        enable_cpu_offload=True,
    ) as runner:
        omni_outputs = list(
            runner.omni.generate(
                prompts=["A photo of a cat sitting on a laptop"],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_inference_steps=2,
                    guidance_scale=4.0,
                    seed=42,
                ),
            )
        )

    assert len(omni_outputs) > 0
    output = omni_outputs[0]
    images = None
    if output.images:
        images = output.images
    elif hasattr(output, "request_output") and output.request_output:
        for stage_out in output.request_output:
            if hasattr(stage_out, "images") and stage_out.images:
                images = stage_out.images
                break

    assert images is not None
    assert len(images) > 0
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)
