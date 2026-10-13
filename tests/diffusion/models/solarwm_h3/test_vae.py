# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm_omni.diffusion.models.solarwm_h3.vae import SolarH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _VideoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = {
            "latent_channels": 2,
            "latents_mean": [0.13, -0.27],
            "latents_std": [0.7, 1.3],
        }
        self.sample = torch.tensor([0.10003, 0.23457, 0.34567, 0.45678, -0.12345, -0.23456, -0.34567, -0.45678])

    def encode(self, pixels, return_dict):
        assert return_dict is False
        self.pixels = pixels

        def sample(generator):
            assert generator.initial_seed() == 42 and generator.device.type == "cpu"
            return self.sample.view(1, 2, 1, 2, 2)

        return (SimpleNamespace(sample=sample),)


def test_anchor_matches_solar_rounding_normalization_and_channel_major_patch_order():
    model = _VideoVAE()
    vae = SolarH3VideoVAE(model)
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3) * 17
    image = Image.fromarray(pixels)
    rows = vae.encode_image(image, height=2, width=2)
    rounded = model.sample.half().float().reshape(2, 4)
    expected = ((rounded - torch.tensor([0.13, -0.27])[:, None]) / torch.tensor([0.7, 1.3])[:, None]).bfloat16()
    torch.testing.assert_close(rows, expected.reshape(1, 8), rtol=0, atol=0)
    assert rows.dtype == torch.bfloat16 and rows.is_contiguous()
    rgb = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    expected_pixels = (rgb - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[
        :, None, None
    ]
    torch.testing.assert_close(model.pixels, expected_pixels.view(1, 3, 1, 2, 2), rtol=0, atol=0)
    unrounded = (model.sample.reshape(2, 4) - torch.tensor([0.13, -0.27])[:, None]) / torch.tensor([0.7, 1.3])[:, None]
    assert not torch.equal(unrounded.bfloat16().reshape(1, 8), rows)


def test_anchor_rejects_unprepared_image_shape():
    vae = SolarH3VideoVAE(_VideoVAE())
    with pytest.raises(ValueError, match="already match"):
        vae.encode_image(Image.new("RGB", (4, 4)), height=2, width=2)


def test_pixel_revert_and_latent_denormalization_use_fp32_stats():
    vae = SolarH3VideoVAE(_VideoVAE())
    latent = torch.tensor([0.33, -0.42], dtype=torch.bfloat16).reshape(1, 2, 1, 1, 1)
    result = vae._denormalize_latent(latent)
    expected = latent.float() * torch.tensor([0.7, 1.3]).view(1, 2, 1, 1, 1)
    expected += torch.tensor([0.13, -0.27]).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    raw = torch.tensor([-10, 0, 10], dtype=torch.float16).view(1, 3, 1, 1, 1)
    pixels = vae._normalize_decoded_frames(raw)
    torch.testing.assert_close(pixels.flatten(), torch.tensor([0, 0.456, 1]), rtol=0, atol=0)
    assert pixels.dtype == torch.float32


def test_component_statistics_ignore_global_bf16_dtype():
    vae = SolarH3VideoVAE(_VideoVAE())
    image = Image.new("RGB", (2, 2), color=(41, 121, 238))
    latent = torch.tensor([0.33, -0.42], dtype=torch.bfloat16).reshape(1, 2, 1, 1, 1)
    expected_anchor = vae.encode_image(image, height=2, width=2)
    expected_decode = vae._denormalize_latent(latent)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        actual_anchor = vae.encode_image(image, height=2, width=2)
        actual_decode = vae._denormalize_latent(latent)
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.testing.assert_close(actual_anchor, expected_anchor, rtol=0, atol=0)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=0, atol=0)
