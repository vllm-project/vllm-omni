# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


@pytest.mark.parametrize(
    "batch,queries,keys,dim", [(2, 16, 16, 16), (1, 17, 17, 64), (3, 50, 350, 64), (8, 64, 368, 64)]
)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("strided_dim", [False, True])
def test_fp32_tiled_attention_mask_and_graph(batch, queries, keys, dim, masked, strided_dim):
    from vllm_omni.model_executor.models.minicpmo_4_5.cfm_attention import cfm_attention

    torch.manual_seed(773)
    q = torch.randn(batch, queries, 8, dim, device="cuda").transpose(1, 2)
    # Like unpacking the model's interleaved K/V history cache.
    kv = torch.randn(batch, 8, keys, dim * 2, device="cuda")
    k, v = kv.chunk(2, dim=-1)
    mask = torch.rand(batch, queries, keys, device="cuda") > 0.3 if masked else None
    if mask is not None:
        mask[:, 0] = False
    if strided_dim:
        q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        k = k.transpose(-1, -2).contiguous().transpose(-1, -2)
        v = v.transpose(-1, -2).contiguous().transpose(-1, -2)
        if mask is not None:
            mask = mask.transpose(-1, -2).contiguous().transpose(-1, -2)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None] if masked else None
        )
    actual = cfm_attention(q, k, v, mask)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cfm_attention(q, k, v, mask)
    if masked:
        mask.logical_not_()
    q.mul_(0.3)
    expected = cfm_attention(q, k, v, mask)
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)


def test_masked_prefix_keeps_later_negative_logits_normalized():
    from vllm_omni.model_executor.models.minicpmo_4_5.cfm_attention import cfm_attention

    q = torch.full((1, 2, 4, 64), 10.0, device="cuda")
    k = torch.full((1, 2, 130, 64), -10.0, device="cuda")
    v = torch.ones_like(k)
    mask = torch.zeros(1, 4, 130, dtype=torch.bool, device="cuda")
    mask[:, 1:, 128:] = True
    expected = torch.ones_like(q)
    expected[:, :, 0] = 0
    torch.testing.assert_close(cfm_attention(q, k, v, mask), expected, atol=2e-5, rtol=2e-5)
