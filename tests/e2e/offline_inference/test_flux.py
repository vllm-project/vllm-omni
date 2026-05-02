# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Flux1 Schnell."""

import pytest
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "black-forest-labs/FLUX.1-schnell"


@pytest.mark.core_model
@pytest.mark.diffusion
def test_flux_schnell_text_to_image():
    """Test FLUX.1-schnell text-to-image generation."""
    omni = Omni(model=MODEL)

    omni_outputs = list(
        omni.generate(
            prompts=["A photo of a cat sitting on a laptop"],
            sampling_params_list=OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=2,
                seed=42,
            ),
        )
    )

    assert len(omni_outputs) > 0
    images = omni_outputs[0].images
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)
