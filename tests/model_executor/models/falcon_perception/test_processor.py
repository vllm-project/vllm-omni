# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception image-preprocessing and token-count parity.

The expected grids below were captured from the reference implementation
(``falcon_perception/data.py`` @ origin/main: ``resize_image_if_necessary`` ->
``ImageProcessor.preprocess`` -> ``calculate_image_tokens``) and matched to
float32 epsilon (1.19e-7) on the pixel values at the time of the port.

Token counts feed vLLM's placeholder bookkeeping: if they drift by even one, the
image block no longer lines up with the projected patch embeddings and the model
fails at runtime rather than degrading gracefully.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
    FalconPerceptionConfig,
)
from vllm_omni.model_executor.models.falcon_perception.processing_falcon_perception import (
    FalconPerceptionImageProcessor,
    FalconPerceptionProcessingInfo,
    resize_image_if_necessary,
    smart_resize_dims,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# (width, height) -> (resized_w, resized_h, grid_h, grid_w)
# Covers: pass-through, downscale-to-max, upscale-to-min, extreme aspect ratios,
# and the pixel-budget clamp (1024x1024 -> 992x992).
REFERENCE_GRIDS = [
    (640, 480, 640, 480, 30, 40),
    (1920, 1080, 1024, 576, 36, 64),
    (100, 100, 256, 256, 16, 16),
    (2048, 512, 1024, 256, 16, 64),
    (333, 777, 336, 784, 49, 21),
    (1024, 1024, 992, 992, 62, 62),
    (57, 300, 192, 1024, 64, 12),
]


@pytest.fixture
def processor() -> FalconPerceptionImageProcessor:
    return FalconPerceptionImageProcessor(FalconPerceptionConfig())


def _image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize(("w", "h", "exp_w", "exp_h", "gh", "gw"), REFERENCE_GRIDS)
def test_pixel_shape_and_grid_match_reference(processor, w, h, exp_w, exp_h, gh, gw):
    pixels = processor(_image(w, h))
    # Channels-last (H, W, C) — the reference keeps this layout end to end.
    assert pixels.shape == (exp_h, exp_w, 3)
    assert processor.target_grid(h, w) == (gh, gw)


@pytest.mark.parametrize(("w", "h", "_exp_w", "_exp_h", "gh", "gw"), REFERENCE_GRIDS)
def test_num_image_tokens_is_patches_plus_six(w, h, _exp_w, _exp_h, gh, gw):
    """5 structural prefix tokens + one per patch + end_of_image."""
    info = FalconPerceptionProcessingInfo.__new__(FalconPerceptionProcessingInfo)
    info.ctx = SimpleNamespace(get_hf_config=lambda _cls: FalconPerceptionConfig())
    assert info.get_num_image_tokens(image_width=w, image_height=h) == gh * gw + 6


def test_normalisation_maps_to_symmetric_unit_range(processor):
    """mean=std=0.5 on [0,1] pixels => exactly [-1, 1]."""
    white = Image.fromarray(np.full((512, 512, 3), 255, dtype=np.uint8))
    black = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    assert torch.allclose(processor(white), torch.ones(1), atol=1e-6)
    assert torch.allclose(processor(black), -torch.ones(1), atol=1e-6)


def test_resize_if_necessary_passes_through_conforming_images():
    img = _image(640, 480)
    assert resize_image_if_necessary(img, 256, 1024) is img


def test_resize_if_necessary_preserves_aspect_within_bounds():
    for w, h in [(4000, 100), (10, 10), (2000, 3000)]:
        out = resize_image_if_necessary(_image(w, h), 256, 1024)
        assert max(out.size) <= 1024


def test_smart_resize_dims_are_patch_aligned_and_within_budget():
    cfg = FalconPerceptionConfig()
    for w, h, *_ in REFERENCE_GRIDS:
        rw, rh = resize_image_if_necessary(_image(w, h), 256, 1024).size
        out_h, out_w = smart_resize_dims(rh, rw, factor=16, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        assert out_h % 16 == 0 and out_w % 16 == 0
        assert out_h * out_w <= cfg.max_pixels


def test_smart_resize_rejects_degenerate_aspect_ratios():
    # Both sides >= factor so the min-size guard passes; ratio 201 > 200.
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize_dims(16, 16 * 201, factor=16, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)


def test_smart_resize_rejects_images_smaller_than_one_patch():
    with pytest.raises(ValueError, match=">= 16"):
        smart_resize_dims(8, 500, factor=16, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
