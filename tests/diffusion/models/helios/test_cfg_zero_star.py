# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline, combine_cfg_zero_star, optimized_scale

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _reference_combine(positive: torch.Tensor, negative: torch.Tensor, scale: float, zero_init: bool) -> torch.Tensor:
    """Pre-fix inline math from _denoise_stage1_step / _denoise_stage2_step."""
    batch_size = positive.shape[0]
    positive_flat = positive.view(batch_size, -1)
    negative_flat = negative.view(batch_size, -1)
    alpha_cfg = optimized_scale(positive_flat, negative_flat)
    alpha_cfg = alpha_cfg.view(batch_size, *([1] * (positive.ndim - 1))).to(positive.dtype)
    if zero_init:
        return positive * 0.0
    return negative * alpha_cfg + scale * (positive - negative * alpha_cfg)


@pytest.mark.parametrize("scale", [1.0, 5.0])
def test_combine_matches_reference_math(scale):
    torch.manual_seed(0)
    positive = torch.randn(1, 4, 12, 8, 16, dtype=torch.bfloat16)
    negative = torch.randn(1, 4, 12, 8, 16, dtype=torch.bfloat16)
    combined = combine_cfg_zero_star(positive, negative, scale, zero_init=False)
    assert torch.equal(combined, _reference_combine(positive, negative, scale, zero_init=False))


def test_zero_init_returns_zeroed_positive():
    torch.manual_seed(0)
    positive = torch.randn(1, 4, 12, 8, 16, dtype=torch.bfloat16)
    negative = torch.randn(1, 4, 12, 8, 16, dtype=torch.bfloat16)
    combined = combine_cfg_zero_star(positive, negative, 5.0, zero_init=True)
    assert torch.all(combined == 0)
    assert combined.shape == positive.shape


@pytest.mark.parametrize("zero_init", [False, True])
def test_combine_accepts_tuple_from_shared_cfg_path(zero_init):
    positive = torch.randn(1, 4, 12, 8, 16, dtype=torch.bfloat16)
    negative = torch.randn_like(positive)
    combined = HeliosPipeline.combine_cfg_noise(
        object.__new__(HeliosPipeline),
        (positive,),
        (negative,),
        5.0,
        kwargs={"helios_cfg_zero_star": {"zero_init": zero_init}},
    )
    assert torch.equal(combined, _reference_combine(positive, negative, 5.0, zero_init))
