# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return current_omni_platform.is_npu() and bool(hasattr(torch, "npu") and torch.npu.is_available())


@pytest.mark.skipif(not _npu_available(), reason="requires an available Ascend NPU")
def test_sdpa_npu_native_gqa_matches_explicit_kv_expansion() -> None:
    torch.manual_seed(0)
    device = torch.device("npu")
    query = torch.randn(1, 5, 4, 8, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 5, 2, 8, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.tensor([[True, True, True, False, False]], device=device)
    metadata = AttentionMetadata(attn_mask=mask)
    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)

    actual = impl.forward_npu(query, key, value, metadata)

    query_bnsd = query.permute(0, 2, 1, 3)
    key_bnsd = key.permute(0, 2, 1, 3).repeat_interleave(2, dim=1)
    value_bnsd = value.permute(0, 2, 1, 3).repeat_interleave(2, dim=1)
    full_qk_mask = mask[:, None, None, :].expand(1, 1, 5, 5).contiguous()
    expected = F.scaled_dot_product_attention(
        query_bnsd,
        key_bnsd,
        value_bnsd,
        attn_mask=full_qk_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=0.5,
        enable_gqa=False,
    ).permute(0, 2, 1, 3)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
