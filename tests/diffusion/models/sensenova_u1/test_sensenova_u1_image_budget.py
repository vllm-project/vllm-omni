# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression for the pixel-budget invariant behind the SenseNova-U1 admission cap.

``SenseNovaU1Pipeline._prepare_input_images`` smart-resizes every reference
with ``min_pixels=512*512`` and ``max_pixels=min(2048*2048, 4096*4096 // num)``.
The upscale floor and the per-image budget share only stay reconciled while
``num <= 4096*4096 // 512*512 == 64``; beyond it the floor exceeds the share
and ``_smart_resize``'s upscale branch pushes every small reference to 512x512,
doubling the documented global pixel budget. The admission cap pins the
pipeline to the valid range; these tests pin the invariant and the resulting
total-pixel bound at the cap boundary.
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

GLOBAL_PIXEL_BUDGET = 4096 * 4096
UPSCALE_FLOOR = 512 * 512


def _admission_cap() -> int:
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    return get_diffusion_model_metadata("SenseNovaU1Pipeline").max_multimodal_image_inputs


def _bounds(num_images: int) -> tuple[int, int]:
    """Mirror _prepare_input_images: (min_pixels, max_pixels) per reference."""
    max_pixels = min(2048 * 2048, GLOBAL_PIXEL_BUDGET // num_images)
    return min(UPSCALE_FLOOR, max_pixels), max_pixels


def test_cap_keeps_resize_bounds_reconciled():
    """The admission cap must not exceed the count where the per-image upscale
    floor (512x512) still fits inside the per-image share of the global
    budget - beyond it min_pixels > max_pixels and _smart_resize upscales
    every small reference past the budget."""
    assert _admission_cap() <= GLOBAL_PIXEL_BUDGET // UPSCALE_FLOOR


def test_small_references_stay_within_global_budget_at_the_cap():
    """At the cap boundary, N small references resize to at most the per-image
    budget share, so the total stays within the documented 4096x4096 budget.

    ceil-to-32 rounding on the upscale branch can exceed the target share by
    at most (sqrt(share) + 31)^2 / share (~13% worst case for non-aligned
    sizes such as 110x110); factor-aligned sizes such as 256x256 land exactly
    on the share."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import _smart_resize

    num = _admission_cap()
    min_pixels, max_pixels = _bounds(num)
    # ceil-to-factor rounding on the upscale branch lands within one factor
    # step (32) per side: worst case (sqrt(share) + 32)^2 / share (~12.9%,
    # reached by non-aligned sizes whose scaled edge sits just above a
    # multiple of 32, e.g. a borderline 110x110 square).
    slack = ((max_pixels**0.5 + 32) ** 2) / max_pixels

    for height, width in [(256, 256), (200, 200), (110, 110), (64, 64), (300, 200), (1000, 900)]:
        total = 0
        for _ in range(num):
            h_bar, w_bar = _smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
            assert h_bar * w_bar <= max_pixels * slack
            total += h_bar * w_bar
        assert total <= GLOBAL_PIXEL_BUDGET * slack, (height, width, total)

    # The reviewer's exact scenario: factor-aligned 256x256 references land
    # precisely on the per-image share, so N of them consume exactly the
    # documented budget (not 2x of it, as at the previous 128 cap).
    h_bar, w_bar = _smart_resize(256, 256, min_pixels=min_pixels, max_pixels=max_pixels)
    assert h_bar * w_bar * num == GLOBAL_PIXEL_BUDGET
