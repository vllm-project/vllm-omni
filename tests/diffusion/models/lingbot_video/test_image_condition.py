# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.models.lingbot_video.image_condition import (
    apply_clean_prefix,
    geometry_align_image,
    prepare_ti2v_image_condition,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _gradient_image(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    rgb = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 127, dtype=np.uint8),
        ],
        axis=-1,
    )
    return Image.fromarray(rgb, mode="RGB")


@pytest.mark.parametrize("source_size", [(160, 64), (64, 160), (96, 96)])
def test_geometry_align_image_supports_source_aspect_ratios(source_size):
    pixel = geometry_align_image(_gradient_image(*source_size), 64, 96)

    assert pixel.shape == (1, 3, 1, 64, 96)
    assert pixel.dtype == torch.float32
    assert 0 <= pixel.min() <= pixel.max() <= 1


@pytest.mark.parametrize(
    ("source_size", "target_size"),
    [((160, 64), (64, 96)), ((64, 160), (96, 64))],
)
def test_geometry_align_image_center_crops_without_stretching(source_size, target_size):
    width, height = source_size
    target_height, target_width = target_size
    rgb = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
    if width > target_width:
        left = (width - target_width) // 2
        rgb[:, left : left + target_width] = (0, 255, 0)
    else:
        top = (height - target_height) // 2
        rgb[top : top + target_height] = (0, 255, 0)

    pixel = geometry_align_image(
        Image.fromarray(rgb, mode="RGB"),
        target_height,
        target_width,
    )

    assert pixel.shape == (1, 3, 1, target_height, target_width)
    expected_green = torch.tensor((0.0, 1.0, 0.0)).view(1, 3, 1, 1, 1)
    assert torch.equal(pixel, expected_green.expand_as(pixel))


class _LatentDistribution:
    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def sample(self, generator=None):
        del generator
        return self.latent


class _CapturingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            latents_mean=[0.0, 0.0, 0.0],
            latents_std=[1.0, 1.0, 1.0],
        )
        self.encoded_pixel = None

    def encode(self, pixel):
        self.encoded_pixel = pixel.detach().clone()
        return SimpleNamespace(latent_dist=_LatentDistribution(pixel))


def test_ti2v_vlm_and_vae_paths_share_geometry_aligned_image():
    vae = _CapturingVAE()
    condition = prepare_ti2v_image_condition(
        _gradient_image(160, 64),
        height=64,
        width=96,
        vae=vae,
        vision_patch_size=16,
        device=torch.device("cpu"),
    )

    vae_pixel = (vae.encoded_pixel + 1.0) / 2.0
    vae_image = vae_pixel[0, :, 0].permute(1, 2, 0).mul(255).byte().numpy()
    assert condition.vlm_image.size == (96, 64)
    np.testing.assert_allclose(
        np.asarray(condition.vlm_image),
        vae_image,
        rtol=0,
        atol=1,
    )


def test_apply_clean_prefix_only_overwrites_condition_frames():
    latents = torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 2, 2)
    original_tail = latents[:, :, 2:].clone()
    clean = torch.full((1, 1, 2, 2, 2), -3.0)

    returned = apply_clean_prefix(latents, clean)

    assert returned is latents
    assert torch.equal(latents[:, :, :2], clean)
    assert torch.equal(latents[:, :, 2:], original_tail)


def test_apply_clean_prefix_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="shape must match"):
        apply_clean_prefix(
            torch.zeros(1, 2, 3, 4, 4),
            torch.zeros(1, 3, 1, 4, 4),
        )
