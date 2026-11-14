# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BAGEL input images must be preprocessed exactly like the reference inferencer.

``_ReferenceResize`` transcribes ``MaxLongEdgeMinShortEdgeResize`` from
``data/transforms.py`` of ByteDance-Seed/Bagel; the reference pipeline is
``ImageTransform(1024, 512, 16)`` for the VAE and ``ImageTransform(980, 224, 14)``
for the ViT, applied to the VAE-resized image.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.bagel.image_transforms import (
    resize_for_vae,
    resize_for_vit,
    to_rgb,
    to_tensor,
    vae_size,
    vit_size,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

torchvision = pytest.importorskip("torchvision")
from torchvision import transforms  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

OFFICIAL_TEST_IMAGES = [(800, 1024), (478, 640), (500, 663)]


class _ReferenceResize:
    def __init__(self, max_size: int, min_size: int, stride: int, max_pixels: int = 14 * 14 * 9 * 1024):
        self.max_size, self.min_size, self.stride, self.max_pixels = max_size, min_size, stride, max_pixels

    def _make_divisible(self, value, stride):
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        return self._make_divisible(new_width, self.stride), self._make_divisible(new_height, self.stride)

    def size(self, width: int, height: int) -> tuple[int, int]:
        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)
        if new_width * new_height > self.max_pixels:
            scale = self.max_pixels / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)
        return new_width, new_height

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = self.size(*image.size)
        return TF.resize(image, (height, width), InterpolationMode.BICUBIC, antialias=True)


REFERENCE_VAE = _ReferenceResize(1024, 512, 16)
REFERENCE_VIT = _ReferenceResize(980, 224, 14)
NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def _image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [*OFFICIAL_TEST_IMAGES, (1024, 1024), (1920, 1080), (300, 300), (2048, 512), (16, 16)])
def test_sizes_match_the_reference(size):
    assert vae_size(*size) == REFERENCE_VAE.size(*size)
    assert vit_size(*vae_size(*size)) == REFERENCE_VIT.size(*REFERENCE_VAE.size(*size))


def test_sizes_match_the_reference_across_a_sweep():
    for width in range(16, 2200, 37):
        for height in range(16, 2200, 53):
            vae = vae_size(width, height)
            assert vae == REFERENCE_VAE.size(width, height), (width, height)
            assert vit_size(*vae) == REFERENCE_VIT.size(*vae), (width, height)


def test_short_edge_is_brought_up_to_512():
    assert vae_size(478, 640) == (512, 688)
    assert vae_size(300, 300) == (512, 512)


def test_vit_keeps_the_aspect_ratio():
    assert vit_size(*vae_size(800, 1024)) == (770, 980)
    assert vit_size(*vae_size(1920, 1080)) == (980, 546)


@pytest.mark.parametrize("size", OFFICIAL_TEST_IMAGES)
def test_pixels_match_the_reference_transforms(size):
    image = _image(*size)

    vae_image = resize_for_vae(image)
    vit_image = resize_for_vit(vae_image)

    reference_vae = REFERENCE_VAE(image)
    assert torch.equal(to_tensor(vae_image), NORMALIZE(reference_vae))
    assert torch.equal(to_tensor(vit_image), NORMALIZE(REFERENCE_VIT(reference_vae)))


def test_rgba_is_composited_on_white_like_pil_img2rgb():
    image = Image.new("RGBA", (4, 4), (10, 20, 30, 0))
    image.putpixel((0, 0), (10, 20, 30, 255))

    rgb = to_rgb(image)

    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (10, 20, 30)
    assert rgb.getpixel((1, 1)) == (255, 255, 255)


def test_rejects_empty_images():
    with pytest.raises(ValueError, match="positive"):
        vae_size(0, 10)
