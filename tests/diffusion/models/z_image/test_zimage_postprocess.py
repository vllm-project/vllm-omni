# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor

from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    _postprocess_to_pil_via_uint8,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_uint8_postprocess_matches_diffusers(dtype: torch.dtype):
    processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=True)
    random_images = torch.randn((2, 3, 17, 19), generator=torch.Generator().manual_seed(42), dtype=dtype)
    edge_values = torch.tensor(
        [-4.0, -1.01, -1.0, -0.5, 0.0, 0.5, 1.0, 1.01, 4.0],
        dtype=dtype,
    )
    edge_images = edge_values.repeat(3, 1).reshape(1, 3, 1, -1)

    for images in (random_images, edge_images):
        expected = processor.postprocess(images)
        actual = _postprocess_to_pil_via_uint8(images, processor)

        assert len(expected) == len(actual)
        assert all(
            np.array_equal(np.asarray(left), np.asarray(right)) for left, right in zip(expected, actual, strict=True)
        )
