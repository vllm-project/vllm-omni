# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HunyuanImage3 AR image RGB conversion."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3Processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FixedResolutionGroup:
    def get_target_size(self, width: int, height: int):
        return width, height

    def get_base_size_and_ratio_index(self, width: int, height: int):
        return 1024, 0


def test_ar_processor_uses_pil_rgb_conversion_for_rgba_images():
    captured: dict[str, object] = {}

    def fake_vit_processor(image: Image.Image):
        captured["mode"] = image.mode
        captured["pixel"] = image.getpixel((0, 0))
        return {
            "pixel_values": torch.zeros((1, 1, 3), dtype=torch.float32),
            "pixel_attention_mask": torch.ones((1, 1), dtype=torch.bool),
            "spatial_shapes": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def fake_vae_processor(_image: Image.Image):
        return torch.zeros((1, 3, 1, 1), dtype=torch.float32)

    processor = object.__new__(HunyuanImage3Processor)
    processor.hf_config = SimpleNamespace(
        vit={"num_channels": 3},
        vae_downsample_factor=(1, 1),
        patch_size=1,
    )
    processor.reso_group = _FixedResolutionGroup()
    processor.vision_encoder_processor = fake_vit_processor
    processor.vae_processor = fake_vae_processor

    processor.process_image(Image.new("RGBA", (1, 1), color=(255, 0, 0, 0)))

    assert captured["mode"] == "RGB"
    assert captured["pixel"] == (255, 0, 0)
