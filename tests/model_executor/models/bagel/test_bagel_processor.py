# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AR-stage processor must feed the reference pixels and a matching token count.

Input images are converted and resized like the reference inferencer
(``ImageTransform(1024, 512, 16)`` for the VAE, ``ImageTransform(980, 224, 14)``
of that result for the ViT). The number of placeholder tokens has to equal the
number of embeddings the model produces for the same image, and the processor
itself must not depend on per-request values such as the output size, which
vLLM uses as a cache key.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.bagel.image_transforms import resize_for_vae, resize_for_vit, to_tensor
from vllm_omni.model_executor.models.bagel.bagel import OmniBagelProcessingInfo, OmniBagelProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GEOMETRY = {"vae_max_size": 1024, "vae_stride": 16, "vit_max_size": 980, "vit_stride": 14}
SIZES = [(800, 1024), (478, 640), (500, 663), (1920, 1080), (1024, 1024)]


def _image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(width * 7919 + height)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _processor() -> OmniBagelProcessor:
    processor = object.__new__(OmniBagelProcessor)
    processor.tokenizer = SimpleNamespace(init_kwargs={})
    processor._merge_kwargs = lambda *args, **kwargs: {"text_kwargs": {}}
    return processor


def _info() -> OmniBagelProcessingInfo:
    info = object.__new__(OmniBagelProcessingInfo)
    config = SimpleNamespace(
        vae_config={"downsample": 8},
        latent_patch_size=2,
        max_latent_size=64,
        vit_config=SimpleNamespace(patch_size=14),
        vit_max_num_patch_per_side=70,
    )
    info.get_hf_config = lambda: config
    info.ctx = Mock()
    return info


@pytest.mark.parametrize("size", SIZES)
def test_img2img_pixels_match_the_reference(size):
    image = _image(*size)

    out = _processor()(images=[image], is_img2img=True, **GEOMETRY)

    vae_image = resize_for_vae(image)
    assert torch.equal(out["pixel_values"][0], to_tensor(vae_image))
    assert torch.equal(out["vit_pixel_values"][0], to_tensor(resize_for_vit(vae_image)))


@pytest.mark.parametrize("size", SIZES)
def test_understanding_pixels_are_the_reference_vit_input(size):
    image = _image(*size)

    out = _processor()(images=[image], is_img2img=False, **GEOMETRY)

    assert "vit_pixel_values" not in out
    assert torch.equal(out["pixel_values"][0], to_tensor(resize_for_vit(resize_for_vae(image))))


def test_images_of_different_sizes_stay_separate():
    out = _processor()(images=[_image(800, 1024), _image(478, 640)], is_img2img=True, **GEOMETRY)

    assert [tuple(t.shape) for t in out["pixel_values"]] == [(3, 1024, 800), (3, 688, 512)]
    assert [tuple(t.shape) for t in out["vit_pixel_values"]] == [(3, 980, 770), (3, 686, 518)]


def test_geometry_comes_from_the_checkpoint_config():
    assert _info().get_image_geometry() == GEOMETRY


@pytest.mark.parametrize("size", SIZES)
def test_token_counts_match_the_processed_pixels(size):
    info = _info()
    image = _image(*size)
    out = _processor()(images=[image], is_img2img=True, **GEOMETRY)

    (vae_w, vae_h), (vit_w, vit_h) = info.get_image_block_sizes(*size)

    assert tuple(out["pixel_values"][0].shape[1:]) == (vae_h, vae_w)
    assert tuple(out["vit_pixel_values"][0].shape[1:]) == (vit_h, vit_w)
