# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 4])
@torch.no_grad()
def test_compiled_ranges_match_fp32_sdpa(dtype, kv_heads):
    generator = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randn(1, 19, 4, 128, device="cuda", dtype=dtype, generator=generator)
    k, v = [torch.randn(1, 23, kv_heads, 128, device="cuda", dtype=dtype, generator=generator) for _ in range(2)]
    qm, km = torch.ones(1, 19, device="cuda", dtype=torch.bool), torch.ones(1, 23, device="cuda", dtype=torch.bool)
    qm[:, 2], km[:, 5] = False, False
    ranges = [[(0, 10, 0, 13), (10, 19, 8, 23)]]
    actual = RangeAttention(128**-0.5)(q, k, v, ranges=ranges, query_mask=qm, key_mask=km)
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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_prebuilt_mask_preserves_static_kernel_numerics():
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(1, 9600, 32, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k, v = [torch.randn(1, 9600, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]

    def mask_mod(batch, head, query, key):
        return (query // 640 - key // 640).abs() <= 1

    mask = create_block_mask(mask_mod, None, None, 9600, 9600, device="cuda", _compile=True)
    actual = RangeAttention(128**-0.5)(q, k, v, block_mask=mask)
    expected = torch.compile(flex_attention, dynamic=False)(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask, enable_gqa=True
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
