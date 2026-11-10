# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression for the pixel-budget invariant behind the SenseNova-U1 admission cap.

``SenseNovaU1Pipeline._prepare_input_images`` smart-resizes every reference
with ``min_pixels=512*512`` and ``max_pixels=min(2048*2048, 4096*4096 // num)``.
``_smart_resize``'s upscale branch ceils to the 32-px factor, which can
overshoot ``max_pixels`` - moderately for squares, severely for long/thin
aspect ratios (64x1000 against a 512x512 share used to land at 160x2048, 1.25x
the budget; 32x6304 at 64x7200, 1.76x). The resize now re-fits by flooring
whenever the ceil overshoots, so the per-image budget is a hard bound. The
admission cap pins the pipeline to the count where the upscale floor still
fits the per-image share; these tests pin the cap and the exact bounds.
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
    """The admission cap must equal the count where the per-image upscale
    floor (512x512) still fits inside the per-image share of the global
    budget - beyond it min_pixels > max_pixels and every small reference
    saturates its share before the budget even splits."""
    assert _admission_cap() == GLOBAL_PIXEL_BUDGET // UPSCALE_FLOOR


def test_resize_never_exceeds_per_image_budget_at_the_cap():
    """At the cap boundary, every reference resize stays within its per-image
    share of the global budget - exactly, with no rounding slack - so N of
    them can never exceed the documented 4096x4096 total."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import _smart_resize

    num = _admission_cap()
    min_pixels, max_pixels = _bounds(num)

    sizes = [
        (256, 256),  # factor-aligned square: lands exactly on the share
        (200, 200),  # scales to exactly 512x512
        (110, 110),  # ceil lands exactly on 512x512
        (64, 64),  # smallest common input
        (300, 200),  # ceil overshoots, re-fit floors back under the share
        (1000, 900),
        (64, 1000),  # long/thin: the ceil overshoot used to reach 1.25x the share
        (32, 6304),  # extreme strip: the ceil overshoot used to reach 1.76x
    ]
    for height, width in sizes:
        total = 0
        for _ in range(num):
            h_bar, w_bar = _smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
            assert h_bar * w_bar <= max_pixels, (height, width, h_bar, w_bar)
            total += h_bar * w_bar
        assert total <= GLOBAL_PIXEL_BUDGET, (height, width, total)


def test_resize_overshoot_refit_exact_results():
    """Pin the re-fit results for the rectangular counter-examples and the
    factor-aligned square that consumes exactly the per-image share."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import _smart_resize

    num = _admission_cap()
    min_pixels, max_pixels = _bounds(num)

    expected = {
        # (height, width): (h_bar, w_bar) - previous unbounded result in comment.
        (64, 1000): (128, 2016),  # was 160x2048 = 1.25x the share
        (32, 6304): (32, 7168),  # was 64x7200 = 1.76x the share
        (256, 256): (512, 512),  # exactly the share, no re-fit needed
    }
    for (height, width), (h_bar, w_bar) in expected.items():
        assert _smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels) == (h_bar, w_bar)

    # 256x256 references consume exactly the documented budget at the cap.
    h_bar, w_bar = _smart_resize(256, 256, min_pixels=min_pixels, max_pixels=max_pixels)
    assert h_bar * w_bar * num == GLOBAL_PIXEL_BUDGET
