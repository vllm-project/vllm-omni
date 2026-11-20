# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The single-launch half-split Q+K path is bit-identical to the per-tensor kernel it replaces."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers import fused_qk_norm_rope as fqk

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or not fqk.HAS_TRITON, reason="CUDA and Triton required"),
]


@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("tokens", "heads"), [(1, 56), (515, 56), (2048, 7), (300, 13)])
def test_token_path_matches_per_tensor_kernel(table_dtype: torch.dtype, tokens: int, heads: int) -> None:
    torch.manual_seed(0)
    head_dim, rotary = fqk._FUSED_HEAD_DIM, fqk._FUSED_ROTARY_DIM
    qkv = 3 * torch.randn(tokens, 3 * heads * head_dim, device="cuda", dtype=torch.bfloat16)
    q = qkv[:, : heads * head_dim].view(tokens, heads, head_dim)
    k = qkv[:, heads * head_dim : 2 * heads * head_dim].view(tokens, heads, head_dim)
    q_weight, k_weight = (
        torch.empty(head_dim, device="cuda", dtype=torch.bfloat16).uniform_(0.5, 1.5) for _ in range(2)
    )
    angles = torch.rand(tokens, rotary // 2, device="cuda") * 6.28
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(table_dtype)
    assert fqk._token_qk_norm_rope_supported(q, k)

    expected_q = fqk._launch_fused_rms_norm_rope(q, q_weight, table, 1e-6)
    expected_k = fqk._launch_fused_rms_norm_rope(k, k_weight, table, 1e-6)
    actual_q, actual_k = fqk._launch_token_qk_norm_rope(q, k, q_weight, k_weight, table, 1e-6)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
