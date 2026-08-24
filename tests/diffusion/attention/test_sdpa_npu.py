# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_sdpa_npu_native_causal_gqa_matches_expanded_kv() -> None:
    """Exercise torch-npu causal GQA with compressed K/V heads."""
    torch.manual_seed(0)
    batch_size, seq_len = 1, 16
    num_heads, num_kv_heads, head_size = 8, 2, 128
    softmax_scale = head_size**-0.5
    device = torch.device("npu")
    dtype = torch.bfloat16

    query = torch.randn(batch_size, seq_len, num_heads, head_size, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn_like(key)

    repeat_num = num_heads // num_kv_heads
    query_bnsd = query.permute(0, 2, 1, 3)
    key_bnsd = key.permute(0, 2, 1, 3)
    value_bnsd = value.permute(0, 2, 1, 3)
    native_gqa = F.scaled_dot_product_attention(
        query_bnsd,
        key_bnsd,
        value_bnsd,
        dropout_p=0.0,
        is_causal=True,
        scale=softmax_scale,
        enable_gqa=True,
    )

    expanded_key = key_bnsd.repeat_interleave(repeat_num, dim=1)
    expanded_value = value_bnsd.repeat_interleave(repeat_num, dim=1)
    expected = F.scaled_dot_product_attention(
        query_bnsd,
        expanded_key,
        expanded_value,
        dropout_p=0.0,
        is_causal=True,
        scale=softmax_scale,
    )

    impl = SDPAImpl(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        softmax_scale=softmax_scale,
        causal=True,
    )
    actual = impl.forward_npu(query, key, value)

    torch.testing.assert_close(native_gqa, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual, native_gqa.permute(0, 2, 1, 3), atol=2e-2, rtol=2e-2)
