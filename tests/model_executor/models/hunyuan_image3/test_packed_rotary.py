# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("tokens", [1, 13])
@pytest.mark.parametrize("query_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_packed_rotary_matches_separate_query_and_key(tokens, query_dtype, mixed_dtype):
    head_dim = 128
    query_width, kv_width = 8 * head_dim, 2 * head_dim
    rope = HunyuanImage3RotaryEmbedding(head_dim)
    pos = torch.arange(tokens)
    positions = torch.stack([pos, pos * 2 - 5, pos // 3 + 9])
    generator = torch.Generator().manual_seed(17)
    qkv = torch.randn(tokens, query_width + 2 * kv_width, generator=generator).to(query_dtype)
    q, k, v = qkv.split((query_width, kv_width, kv_width), dim=-1)
    expected_q, expected_k = rope(positions, q, k)
    hidden_dtype = torch.float16 if mixed_dtype else query_dtype
    factors = rope.prepare_factors(positions, hidden_dtype)
    actual_q, actual_k, actual_v = rope.apply_packed(factors, qkv, query_width, kv_width)

    for actual, expected in ((actual_q, expected_q), (actual_k, expected_k)):
        assert actual.is_contiguous()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert actual_v.data_ptr() == v.data_ptr()
    assert actual_v.stride() == v.stride()
