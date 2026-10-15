# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selector tests require CUDA/Triton, independently of FA4 and Hopper."""

import math

import pytest
import torch

from tests.helpers.block_sparse import make_attention_inputs

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture
def cuda_selector():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for Triton routing")
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("batch,kv_heads", [(1, 4), (2, 2)])
@pytest.mark.cuda
@torch.inference_mode()
def test_gqa_routing_and_prefix(cuda_selector, batch, kv_heads):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    torch.manual_seed(17)
    q = torch.randn(batch, 129, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 4226, kv_heads, 128, device="cuda", dtype=q.dtype)
    selector = SubBlockTopK({"target_sparsity": 0.75})
    selector.prepare((64, 64), 128, q.device)
    expected = selector.select(q, k.repeat_interleave(4 // kv_heads, dim=2), 0.1, 0).indices
    torch.testing.assert_close(selector.select(q, k, 0.1, 0).indices, expected, atol=0, rtol=0)
    protected = selector.select(q, k, 0.1, 129).indices
    assert protected.shape == (batch, 4, 3, 19)
    assert torch.all(protected[..., 1:] > protected[..., :-1])
    torch.testing.assert_close(
        protected[..., :3], torch.arange(3, device="cuda", dtype=torch.int32).expand_as(protected[..., :3])
    )
    all_protected = selector.select(q, k, 0.1, k.shape[1])
    expected_all = torch.arange(67, dtype=torch.int32, device=q.device).expand(batch, 4, 3, 67)
    torch.testing.assert_close(all_protected.indices, expected_all, atol=0, rtol=0)
    with pytest.raises(ValueError, match="nonempty keys"):
        selector.select(q, k[:, :0], 0.1, 0)


@pytest.mark.cuda
@pytest.mark.parametrize("head_size", [8, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_pooling_and_scores_against_reference(cuda_selector, head_size, dtype):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import _fused_pool, _fused_scores

    torch.manual_seed(23)
    q = torch.randn(1, 65, 2, head_size, device="cuda", dtype=dtype)
    k = torch.randn(1, 129, 2, head_size, device="cuda", dtype=q.dtype)

    def pool(x, cells, scale):
        out = torch.zeros(x.shape[2], cells, head_size, device="cuda", dtype=x.dtype)
        for i in range((x.shape[1] + 15) // 16):
            out[:, i] = (x[0, i * 16 : (i + 1) * 16].float().mean(0) * scale).to(x.dtype)
        return out

    qp, kp = (
        torch.empty(2, 8, head_size, device="cuda", dtype=q.dtype),
        torch.empty(2, 12, head_size, device="cuda", dtype=q.dtype),
    )
    scale = 0.1 * math.log2(math.e)
    _fused_pool(q, 8, 16, qp, scale)
    _fused_pool(k, 12, 16, kp)
    torch.testing.assert_close(qp, pool(q, 8, scale), atol=0, rtol=0)
    torch.testing.assert_close(kp, pool(k, 12, 1), atol=0, rtol=0)
    scores = torch.empty(2, 2, 3, device="cuda", dtype=torch.float32)
    _fused_scores(qp, kp, scores, n_k=4, n_valid=9, n_q=4, m_valid=5)
    dots = qp.float() @ kp.float().transpose(-1, -2)
    dots[:, 5:, :] = -float("inf")
    dots[:, :, 9:] = -float("inf")
    dots = dots.reshape(2, 2, 4, 3, 4).permute(0, 1, 3, 2, 4).reshape(2, 2, 3, 16)
    reference = torch.logsumexp(dots * math.log(2), dim=-1)
    torch.testing.assert_close(scores, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "sparsity,kv_len,prefix,kept",
    [(0.0, 1089, 0, 18), (0.75, 257, 0, 5), (0.75, 1089, 1089, 18), (0.75, 1089, 1025, 18)],
)
@torch.inference_mode()
def test_known_selection_skips_scoring(cuda_selector, monkeypatch, sparsity, kv_len, prefix, kept):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    q, k, _ = make_attention_inputs(kv_len=kv_len)
    selector = SubBlockTopK({"target_sparsity": sparsity})
    selector.prepare((64, 64), 128, q.device)

    def unexpected_scoring(*args, **kwargs):
        pytest.fail("A fully determined selection must not compute scores")

    monkeypatch.setattr(SubBlockTopK, "_compute_scores", staticmethod(unexpected_scoring))
    selection = selector.select(q, k, 128**-0.5, prefix)
    expected = torch.arange(kept, dtype=torch.int32, device=q.device).expand(2, 4, 3, kept)
    torch.testing.assert_close(selection.indices, expected, atol=0, rtol=0)
    torch.testing.assert_close(selection.counts, torch.full_like(selection.counts, kept), atol=0, rtol=0)
    assert selection.indices.is_contiguous() and selection.counts.is_contiguous()
    if sparsity == 0.0:
        compiled = torch.compile(selector.select, fullgraph=True)
        actual = compiled(q, k, 128**-0.5, prefix)
        torch.testing.assert_close(actual.indices, selection.indices, atol=0, rtol=0)
        torch.testing.assert_close(actual.counts, selection.counts, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("prefix", [0, 64, 65, 2048])
@torch.inference_mode()
def test_prefix_does_not_consume_candidate_budget(cuda_selector, prefix):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    prefix_blocks = (prefix + 63) // 64
    # Keep 64 unprotected blocks so their retained budget stays at 16.
    q, k, _ = make_attention_inputs(kv_len=prefix_blocks * 64 + 4096)
    selector = SubBlockTopK({"target_sparsity": 0.75})
    selector.prepare((64, 64), 128, q.device)
    selection = selector.select(q, k, 128**-0.5, prefix)
    assert selection.indices.shape == (2, 4, 3, prefix_blocks + 16)
    assert torch.all(selection.counts == prefix_blocks + 16)
    assert torch.all(selection.indices[..., 1:] > selection.indices[..., :-1])
    assert torch.all(selection.indices[..., prefix_blocks:] >= prefix_blocks)
    expected_prefix = torch.arange(prefix_blocks, dtype=torch.int32, device=q.device)
    torch.testing.assert_close(
        selection.indices[..., :prefix_blocks], expected_prefix.expand(2, 4, 3, prefix_blocks), atol=0, rtol=0
    )
    if prefix == 65:
        compiled = torch.compile(selector.select, fullgraph=True)
        actual = compiled(q, k, 128**-0.5, prefix)
        torch.testing.assert_close(actual.indices, selection.indices, atol=0, rtol=0)
        torch.testing.assert_close(actual.counts, selection.counts, atol=0, rtol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("prefix,retained", [(0, 80), (514, 89), (19234, 301)])
def test_cosmos3_prefix_budget(prefix, retained):
    """The documented 301-block workload budgets unprotected candidates only."""
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    prefix_blocks = (prefix + 63) // 64
    assert prefix_blocks + SubBlockTopK.block_budget(301 - prefix_blocks, 0.75) == retained


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kv_heads,block_size,prefix,sparsity",
    [
        (1, (64, 64), 65, 0.75),
        (2, (128, 64), 0, 0.875),
        (4, (64, 128), 129, 0.5),
    ],
)
@torch.inference_mode()
def test_pinned_routing_reference(cuda_selector, kv_heads, block_size, prefix, sparsity):
    from tests.helpers.block_sparse import pinned_subblock_reference
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    q, k, _ = make_attention_inputs(q_len=129, kv_len=4226, kv_heads=kv_heads)
    scale = q.shape[-1] ** -0.5
    scores, expected = pinned_subblock_reference(q, k, scale, sparsity, prefix, block_size)
    selector = SubBlockTopK({"target_sparsity": sparsity})
    selector.prepare(block_size, q.shape[-1], q.device)
    torch.testing.assert_close(selector._compute_scores(q, k, scale, block_size), scores, atol=1e-5, rtol=1e-5)
    actual = selector.select(q, k, scale, prefix)
    torch.testing.assert_close(actual.indices, expected.indices, atol=0, rtol=0)
    torch.testing.assert_close(actual.counts, expected.counts, atol=0, rtol=0)


@pytest.mark.parametrize(
    "candidates,sparsity,expected", [(0, 0.75, 0), (3, 0.75, 3), (64, 0.75, 16), (292, 0.75, 80), (301, 0.75, 80)]
)
def test_candidate_budget_without_gpu(candidates, sparsity, expected):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    assert SubBlockTopK.block_budget(candidates, sparsity) == expected


@pytest.mark.cuda
@pytest.mark.parametrize("sparsity", [0.75, 0.95, 0.98])
@torch.inference_mode()
def test_pinned_routing_ties_and_high_sparsity(cuda_selector, sparsity):
    from tests.helpers.block_sparse import pinned_subblock_reference
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    q, k, _ = make_attention_inputs(q_len=17, kv_len=16385, heads=2, kv_heads=1)
    # Head zero has tied scores for all full blocks. Head one retains random scores.
    q[:, :, 0] = 0
    scale = q.shape[-1] ** -0.5
    _, expected = pinned_subblock_reference(q, k, scale, sparsity, 65, (64, 64))
    selector = SubBlockTopK({"target_sparsity": sparsity})
    selector.prepare((64, 64), q.shape[-1], q.device)
    actual = selector.select(q, k, scale, 65)
    torch.testing.assert_close(actual.indices, expected.indices, atol=0, rtol=0)
    torch.testing.assert_close(actual.counts, expected.counts, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["nan", "mixed", "positive_inf", "negative_inf", "extreme_finite"])
@torch.inference_mode()
def test_nonfinite_score_rows_have_complete_unique_selection(cuda_selector, kind):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import _fused_topk

    scores = torch.arange(32, device="cuda", dtype=torch.float32).repeat(2, 1)
    if kind == "nan":
        scores[0].fill_(float("nan"))
    elif kind == "mixed":
        scores[0, 17] = float("nan")
    elif kind == "extreme_finite":
        scores[0, ::2] = -torch.finfo(torch.float32).max
        scores[0, 1::2] = torch.finfo(torch.float32).max
    else:
        scores[0, 17] = float("inf") if kind == "positive_inf" else -float("inf")
    actual = _fused_topk(scores, 8)
    assert torch.all((actual >= 0) & (actual < 32))
    assert torch.all(actual[:, 1:] > actual[:, :-1])
    if kind != "extreme_finite":
        torch.testing.assert_close(actual[0], torch.arange(8, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(actual[1:], _fused_topk(scores[1:].contiguous(), 8))


@pytest.mark.cuda
@torch.inference_mode()
def test_finite_input_pooling_overflow_has_valid_counts_and_prefix(cuda_selector):
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    q = torch.zeros(1, 64, 1, 64, dtype=torch.float16, device="cuda")
    q[:, :16] = 46400
    k = torch.ones(1, 2048, 1, 64, dtype=q.dtype, device=q.device)
    assert torch.isfinite(q).all() and torch.isfinite(k).all()
    selector = SubBlockTopK({"target_sparsity": 0.75})
    selector.prepare((64, 64), 64, q.device)
    scores = selector._compute_scores(q, k, 1.0, (64, 64))
    assert not torch.isfinite(scores).all()
    for prefix in (0, 65):
        selection = selector.select(q, k, 1.0, prefix)
        capacity = selection.indices.shape[-1]
        assert torch.all(selection.counts == capacity)
        expected = torch.arange(capacity, device=q.device, dtype=torch.int32).expand_as(selection.indices)
        torch.testing.assert_close(selection.indices, expected)
