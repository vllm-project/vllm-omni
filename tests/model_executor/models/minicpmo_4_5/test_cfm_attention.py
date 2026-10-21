# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CFM attention preserves masking and pooled row selection at both precisions."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.cfm_attention import cfm_attention

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("frames", [1, 50])
def test_cfm_attention_matches_sdpa(dtype, pooled, frames):
    torch.manual_seed(17)
    q = torch.randn(2, 4, frames, 64, device="cuda", dtype=dtype)
    kv = torch.randn(4 if pooled else 2, 4, 137, 128, device="cuda", dtype=dtype)
    k, v = kv[..., :64], kv[..., 64:]
    rows = torch.tensor([3, 1], device="cuda") if pooled else None
    mask = torch.rand(2, frames, 137, device="cuda") > 0.3
    mask[0, 0] = False
    expected = F.scaled_dot_product_attention(
        q.float(),
        (k if rows is None else k[rows]).float(),
        (v if rows is None else v[rows]).float(),
        attn_mask=mask.unsqueeze(1),
    )
    actual = cfm_attention(q, k, v, mask, rows)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-3)
    assert torch.count_nonzero(actual[0, :, 0]) == 0
