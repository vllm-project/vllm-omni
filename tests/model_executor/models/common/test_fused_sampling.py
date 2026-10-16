# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


def _reference(logits, uniforms, top_k, top_p):
    if top_k > 0:
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1:
        values, indices = logits.sort(dim=-1, descending=True)
        probs = values.softmax(dim=-1, dtype=torch.float32)
        values[(probs.cumsum(-1) - probs) >= top_p] = -torch.inf
        logits = values.scatter(1, indices, values)
    return (logits.float() - torch.log(-torch.log(uniforms))).argmax(-1, keepdim=True)


@pytest.mark.parametrize("batch,vocab", [(1, 128), (7, 2048), (64, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k,top_p", [(-1, 0.8), (0, 1.0), (50, 0.8), (50, 0.01)])
def test_fixed_uniform_sampling_and_graph(batch, vocab, dtype, top_k, top_p):
    from vllm_omni.model_executor.models.common.fused_sampling import sample_top_k_top_p_gumbel

    torch.manual_seed(173)
    logits = torch.randn(batch, vocab, device="cuda", dtype=dtype)
    # The MTP caller supplies a noncontiguous row slice of [B, 15, vocab].
    uniforms = torch.rand(batch, 15, vocab, device="cuda")[:, 3].clamp_min_(1e-7)
    expected = _reference(logits, uniforms, top_k, top_p)
    actual = sample_top_k_top_p_gumbel(logits, uniforms, top_k=top_k, top_p=top_p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sample_top_k_top_p_gumbel(logits, uniforms, top_k=top_k, top_p=top_p)
    logits.add_(torch.randn_like(logits) * 0.1)
    uniforms.uniform_(1e-7, 1 - 1e-7)
    graph.replay()
    torch.testing.assert_close(captured, _reference(logits, uniforms, top_k, top_p), rtol=0, atol=0)


@pytest.mark.parametrize("top_k,top_p", [(3, 0.5), (3, 1.0), (0, 0.8)])
def test_threshold_ties_and_degenerate_rows(top_k, top_p):
    from vllm_omni.model_executor.models.common.fused_sampling import sample_top_k_top_p_gumbel

    logits = torch.tensor(
        [[2, 2, 2, 2, 2, 1, 0, -1], [1] * 8, [-torch.inf] * 8, [torch.nan, 1, 2, 3, 4, 5, 6, 7]],
        device="cuda",
        dtype=torch.float32,
    )
    uniforms = torch.tensor([[0.4, 0.7, 0.8, 0.9, 0.2, 0.3, 0.6, 0.5]], device="cuda").expand(4, -1)
    actual = sample_top_k_top_p_gumbel(logits, uniforms, top_k=top_k, top_p=top_p)
    torch.testing.assert_close(actual, _reference(logits, uniforms, top_k, top_p), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_nucleus_boundaries(dtype):
    from vllm_omni.model_executor.models.common.fused_sampling import sample_top_k_top_p_gumbel

    torch.manual_seed(1923)
    logits = torch.randn(1, 2048, device="cuda", dtype=dtype)
    values, indices = logits.sort(descending=True)
    probabilities = values.softmax(-1, dtype=torch.float32)
    before = probabilities.cumsum(-1) - probabilities
    for rank in [1, 19, 49, 1000, 2000]:
        boundary = before[0, rank]
        uniforms = torch.full_like(logits, 0.001, dtype=torch.float32)
        uniforms[0, indices[0, rank]] = 1 - 1e-7
        for direction in [-torch.inf, torch.inf]:
            top_p = torch.nextafter(boundary, boundary.new_tensor(direction)).item()
            actual = sample_top_k_top_p_gumbel(logits, uniforms, top_k=0, top_p=top_p)
            torch.testing.assert_close(actual, _reference(logits, uniforms, 0, top_p), rtol=0, atol=0)


@pytest.mark.parametrize("top_p", [0.8, 1.0])
def test_strided_logits_and_non_power_of_two_vocabulary(top_p):
    from vllm_omni.model_executor.models.common.fused_sampling import sample_top_k_top_p_gumbel

    for seed in range(16):
        torch.manual_seed(seed)
        logits = torch.randn(128, 2050, device="cuda")[:, ::2]
        uniforms = torch.rand(128, 3, 1025, device="cuda")[:, 1].clamp_min_(1e-7)
        actual = sample_top_k_top_p_gumbel(logits, uniforms, top_k=-1, top_p=top_p)
        torch.testing.assert_close(actual, _reference(logits, uniforms, -1, top_p), rtol=0, atol=0)
