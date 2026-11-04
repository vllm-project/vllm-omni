# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, AttentionRanges
from vllm_omni.diffusion.attention.layer import Attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 4])
@torch.no_grad()
def test_compiled_flex_ranges_match_fp32_sdpa(dtype, kv_heads):
    generator = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randn(1, 19, 4, 128, device="cuda", dtype=dtype, generator=generator)
    k, v = [torch.randn(1, 23, kv_heads, 128, device="cuda", dtype=dtype, generator=generator) for _ in range(2)]
    qm, km = torch.ones(1, 19, device="cuda", dtype=torch.bool), torch.ones(1, 23, device="cuda", dtype=torch.bool)
    qm[:, 2], km[:, 5] = False, False
    rectangles = [[(0, 10, 0, 13), (10, 19, 8, 23)]]
    attention = Attention(
        4, 128, causal=False, softmax_scale=128**-0.5, num_kv_heads=kv_heads, default_backend="FLEX_ATTENTION"
    )
    actual = attention(q, k, v, AttentionMetadata(range_mask=AttentionRanges(rectangles, qm, km)))
    pair_mask = torch.zeros(1, 1, 19, 23, device="cuda", dtype=torch.bool)
    pair_mask[:, :, :10, :13], pair_mask[:, :, 10:, 8:] = True, True
    pair_mask &= qm[:, None, :, None] & km[:, None, None, :]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().repeat_interleave(4 // kv_heads, 2).transpose(1, 2),
            v.float().repeat_interleave(4 // kv_heads, 2).transpose(1, 2),
            attn_mask=pair_mask,
        ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(actual[:, 2]) == 0
    assert torch.isfinite(actual).all()
