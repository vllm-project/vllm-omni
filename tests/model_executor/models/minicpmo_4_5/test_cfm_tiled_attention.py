# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiled attention must preserve masking and pooled-row ownership in FP16."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.cfm_attention import cfm_attention

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("frames,history", [(1, 17), (23, 65), (50, 513)])
def test_tiled_attention_matches_math_sdpa(dtype, pooled, masked, frames, history):
    generator = torch.Generator(device="cuda").manual_seed(43)
    batch, heads, width = 3, 4, 64
    q = torch.randn(batch, heads, frames, width, device="cuda", dtype=dtype, generator=generator)
    # K and V are interleaved in the real resident pool, so both views are strided.
    storage = torch.randn(
        7 if pooled else batch, heads, history, 2 * width, device="cuda", dtype=dtype, generator=generator
    )
    k, v = storage[..., :width], storage[..., width:]
    rows = torch.tensor([5, 1, 5], device="cuda") if pooled else None
    mask = None
    if masked:
        mask = torch.rand(batch, frames, history, device="cuda", generator=generator) > 0.3
        mask[0, 0] = False
    expected_k = k.index_select(0, rows) if pooled else k
    expected_v = v.index_select(0, rows) if pooled else v
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.float(),
            expected_k.float(),
            expected_v.float(),
            attn_mask=mask.unsqueeze(1) if mask is not None else None,
        ).to(dtype)
    actual = cfm_attention(q, k, v, mask, rows)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(actual).all()
    if masked:
        assert torch.count_nonzero(actual[0, :, 0]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tiled_attention_rejects_mixed_cache_dtype(dtype):
    q = torch.empty(1, 1, 1, 16, dtype=dtype)
    cache_dtype = torch.float16 if dtype == torch.float32 else torch.float32
    k = torch.empty_like(q, dtype=cache_dtype)
    with pytest.raises(ValueError, match="matching float32 or float16"):
        cfm_attention(q, k, k)
