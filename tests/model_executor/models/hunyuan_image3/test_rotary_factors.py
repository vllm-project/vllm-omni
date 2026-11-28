# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage-3.0 shares the same interleaved RoPE factors across decoder layers."""

import pytest
import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("two_dimensional", [False, True])
def test_shared_rotary_factors(dtype, two_dimensional):
    dim, length = 128, 17
    rope = HunyuanImage3RotaryEmbedding(dim)
    indices = torch.arange(length)
    positions = torch.stack((indices, indices * 2, indices // 3)) if two_dimensional else indices
    gen = torch.Generator().manual_seed(17)
    projected = torch.randn(length, 12 * dim, generator=gen, dtype=dtype)

    factors = rope.cos_sin(positions)
    for offset in (0, 1):
        # Consecutive layers consume different Q/K projections at identical positions.
        q, k, _ = (projected + offset).split((8 * dim, 2 * dim, 2 * dim), dim=-1)
        assert not q.is_contiguous() and not k.is_contiguous()
        actual = rope(factors, q, k)
        expected = rope(positions, q, k)
        for out, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
