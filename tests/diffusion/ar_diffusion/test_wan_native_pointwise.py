# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kernel rounding must match separate eager BF16 operations exactly."""

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("transposed", [False, True])
def test_native_residual_preserves_intermediate_bf16_rounding(transposed):
    from benchmarks.ar_diffusion.native_pointwise import residual

    torch.manual_seed(72)
    x = torch.randn((1, 1536, 257) if transposed else (1, 257, 1536), device="cuda", dtype=torch.bfloat16)
    if transposed:
        x = x.transpose(1, 2)
    delta = torch.randn_like(x)
    gate = torch.randn((1, 1, 1536), device="cuda", dtype=torch.bfloat16)
    actual = residual(x, delta, gate)
    assert torch.equal(actual, x + delta * gate)
    assert actual.is_contiguous()


def test_native_modulation_preserves_all_three_bf16_roundings():
    from benchmarks.ar_diffusion.native_pointwise import modulation

    torch.manual_seed(93)
    x = torch.randn((1, 257, 1536), device="cuda", dtype=torch.bfloat16)
    scale = torch.randn((1, 1, 1536), device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    assert torch.equal(modulation(x, scale, shift), x * (1 + scale) + shift)
