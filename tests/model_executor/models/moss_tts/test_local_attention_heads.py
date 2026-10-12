# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Head-group MOSS Local lookup attention for every group size and position."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

HEADS, DIM, NVQ, VOCAB = 32, 80, 12, 1024
HIDDEN = HEADS * DIM


@pytest.mark.parametrize("batch", [1, 7, 64, 128])
@pytest.mark.parametrize("position", [0, 1, 6, 11])
@pytest.mark.parametrize("lookup", [False, True])
@pytest.mark.parametrize("group", [4, 8, 32])
@torch.inference_mode()
def test_head_groups_match_sdpa_and_write_exact_kv(batch, position, lookup, group):
    from vllm_omni.model_executor.models.moss_tts.local_kernels import lookup_attention

    gen = torch.Generator(device="cuda").manual_seed(batch * 31 + position)
    rows = VOCAB if lookup else batch
    qkv = torch.randn((rows, 3 * HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    embedding = torch.randn((VOCAB, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    key = torch.randn((batch, HEADS, NVQ, DIM), device="cuda", generator=gen).to(torch.bfloat16)
    value = torch.randn_like(key)
    tokens = torch.randint(0, VOCAB, (batch,), device="cuda", generator=gen) if lookup else None

    selected = qkv[tokens] if lookup else qkv
    q, k, v = (x.reshape(batch, HEADS, 1, DIM) for x in selected.split(HIDDEN, -1))
    expected_k, expected_v = key.clone(), value.clone()
    expected_k[:, :, position : position + 1] = k
    expected_v[:, :, position : position + 1] = v
    expected = F.scaled_dot_product_attention(
        q, expected_k[:, :, : position + 1], expected_v[:, :, : position + 1]
    ).reshape(batch, HIDDEN)

    out, residual = lookup_attention(
        qkv, key, value, position, tokens=tokens, embedding=embedding if lookup else None, group=group
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert torch.equal(key, expected_k) and torch.equal(value, expected_v)
    if lookup:
        assert torch.equal(residual, embedding[tokens])
    if position == 0:
        assert torch.equal(out, v.reshape(batch, HIDDEN))
