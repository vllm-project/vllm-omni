# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from unittest import mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout, VideoTokenSpan
from vllm_omni.diffusion.attention.backends.rainfusion_attn import RainFusionAttentionImpl
from vllm_omni.diffusion.attention.ops import rainfusion_hybrid as hybrid

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("grid", [(1, 8, 16), (3, 16, 24), (4, 10, 18), (1, 5, 9), (2, 5, 9)])
def test_cpu_spatial_tiling_is_bijective_and_preserves_remainder_first_frame(grid):
    indices = hybrid._tile_indices(7, grid)
    assert torch.equal(indices.sort().values, torch.arange(7, 7 + math.prod(grid)))
    if grid[1] % 8 or grid[2] % 8:
        assert torch.equal(indices[: grid[1] * grid[2]], torch.arange(7, 7 + grid[1] * grid[2]))


def test_multi_clip_packing_isolates_boundaries_and_covers_context():
    spans = ((10, (2, 5, 9)), (200, (2, 8, 16)))
    perm, protected = hybrid._packing_cpu(500, spans, 128)
    assert torch.equal(perm.sort().values, torch.arange(500))
    assert torch.equal(perm[128:256].sort().values, torch.arange(200, 328))
    assert 0 in protected and 1 in protected and 3 in protected


def test_rectangular_plan_never_adds_reference_queries_and_protects_reference_keys():
    plan = hybrid.get_hybrid_plan(1280, 1536, ((256, (4, 16, 16)),), ((512, (4, 16, 16)),), "cpu")
    all_queries = torch.cat((plan.sparse_queries, plan.dense_queries))
    assert torch.equal(all_queries.sort().values, torch.arange(1280))
    assert plan.sparse_query_rows == 768
    assert plan.dense_query_rows == 512
    assert plan.key_rows == 1536
    assert torch.equal(plan.protected_keys, torch.tensor([0, 1, 8, 9, 10, 11]))
    assert plan is hybrid.get_hybrid_plan(1280, 1536, ((256, (4, 16, 16)),), ((512, (4, 16, 16)),), "cpu")


def test_block_selection_keeps_context_first_frame_and_softmax_ties():
    plan = hybrid.get_hybrid_plan(1280, 1536, ((256, (4, 16, 16)),), ((512, (4, 16, 16)),), "cpu")
    q = torch.zeros(1, plan.sparse_query_rows, 2, 8)
    k = torch.zeros(1, plan.key_rows, 2, 8)
    idx, count = hybrid.select_blocks(q, k, plan, 0.8, 8**-0.5)
    # rf_v2 uses >= threshold, so tied scores retain all keys, not exactly topk.
    assert torch.equal(count, torch.full_like(count, 12))
    assert torch.equal(idx, torch.arange(12).reshape(1, 1, 12).expand(6, 2, 12))
    torch.manual_seed(11)
    q = torch.randn_like(q)
    k = torch.randn_like(k)
    idx, count = hybrid.select_blocks(q, k, plan, 0.8, 8**-0.5)
    for row in range(idx.shape[0]):
        for head in range(idx.shape[1]):
            selected = idx[row, head, : count[row, head]].tolist()
            assert set(plan.protected_keys.tolist()).issubset(selected)


def test_dense_and_sparse_calls_cover_only_real_queries(monkeypatch):
    plan = hybrid.get_hybrid_plan(1280, 1536, ((256, (4, 16, 16)),), ((512, (4, 16, 16)),), "cpu")
    torch.manual_seed(20)
    q = torch.randn(1, 1280, 2, 8)
    k = torch.randn(1, 1536, 2, 8)
    v = torch.randn_like(k)
    seen = {}

    def dense(q_dense, key, value, geometry):
        seen["dense_rows"] = q_dense.shape[1]
        torch.testing.assert_close(q_dense, q.index_select(1, geometry.dense_queries))
        return q_dense + 1

    def sparse(q_sparse, key, value, **kwargs):
        seen["sparse_rows"] = q_sparse.shape[0]
        assert kwargs["actual_seq_lengths"] == [768]
        assert kwargs["actual_seq_lengths_kv"] == [1536]
        assert kwargs["select_idx"].shape == (6, 2, 12)
        torch.testing.assert_close(key, k.index_select(1, plan.key_permutation).squeeze(0))
        return q_sparse + 2

    monkeypatch.setattr(hybrid, "_native_forward", sparse)
    out = hybrid.hybrid_attention(q, k, v, plan, dense, sparsity=0.8, scale=8**-0.5, kernel_dtype="bf16", kernel="rf2")
    assert seen["dense_rows"] + seen["sparse_rows"] == q.shape[1]
    torch.testing.assert_close(out.index_select(1, plan.dense_queries), q.index_select(1, plan.dense_queries) + 1)
    torch.testing.assert_close(out.index_select(1, plan.sparse_queries), q.index_select(1, plan.sparse_queries) + 2)


def test_backend_hybrid_preserves_padding_and_laser_scale(monkeypatch):
    impl = RainFusionAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8},
    )
    metadata = AttentionMetadata(
        extra={"max_seqlen_q": 1280, "max_seqlen_k": 1536, "laser_input_scale": 256.0},
        video_layout=VideoTokenLayout(
            used_len=1280,
            video_spans=(VideoTokenSpan(start=256, latent_grid=(4, 16, 16), role="target"),),
        ),
    )
    q = torch.randn(1, 1283, 2, 128)
    k = torch.randn(1, 1539, 2, 128)
    impl.dense_fallback.forward_npu = mock.Mock(side_effect=lambda q, k, v, m: q)
    plan = mock.Mock(video_spans=[{"start": 512, "latent_shape": [4, 16, 16]}])

    def run(q, k, v, geometry, dense, **kwargs):
        assert q.shape[1] == 1280 and k.shape[1] == 1536
        dense(q.index_select(1, geometry.dense_queries), k, v, geometry)
        return q

    monkeypatch.setattr(hybrid, "hybrid_attention", run)
    out = impl._forward_hybrid_npu(q, k, k, metadata, plan, 256)
    torch.testing.assert_close(out[:, :1280], q[:, :1280])
    assert torch.count_nonzero(out[:, 1280:]) == 0
    dense_metadata = impl.dense_fallback.forward_npu.call_args.args[3]
    assert dense_metadata.extra["laser_input_scale"] == 256.0
    assert dense_metadata.extra["max_seqlen_q"] == 512
    assert dense_metadata.extra["max_seqlen_k"] == 1536
    assert dense_metadata.attn_mask is None


def test_fast_kernel_cast_compensates_scale_and_restores_output(monkeypatch):
    plan = hybrid.get_hybrid_plan(1280, 1536, ((256, (4, 16, 16)),), ((512, (4, 16, 16)),), "cpu")
    q = torch.full((1, 1280, 2, 8), 0.25, dtype=torch.bfloat16)
    k = torch.full((1, 1536, 2, 8), 0.5, dtype=torch.bfloat16)
    v = torch.ones_like(k)

    def sparse(query, key, value, **kwargs):
        assert query.dtype == key.dtype == value.dtype == torch.float16
        assert kwargs["inner_precise"] == 1
        assert kwargs["scale"] == 8**-0.5 * 256**2
        assert torch.equal(query, torch.full_like(query, 0.25 / 256))
        assert torch.equal(key, torch.full_like(key, 0.5 / 256))
        return query

    monkeypatch.setattr(hybrid, "_native_forward", sparse)
    out = hybrid.hybrid_attention(q, k, v, plan, lambda q, k, v, p: q, sparsity=0.8, scale=8**-0.5, kernel="rf2")
    assert out.dtype == q.dtype
    assert torch.equal(out, q)


def test_binary_kernel_receives_same_mask_and_no_manufactured_queries(monkeypatch):
    plan = hybrid.get_hybrid_plan(1280, 1536, ((256, (4, 16, 16)),), ((512, (4, 16, 16)),), "cpu")
    q = torch.zeros(1, 1280, 2, 8, dtype=torch.bfloat16)
    k = torch.zeros(1, 1536, 2, 8, dtype=torch.bfloat16)
    v = torch.ones_like(k)

    def native(query, key, value, mask, scale, precise):
        assert query.shape[1] == 768 and key.shape[1] == 1536
        assert mask.shape == (1, 2, 6, 12)
        assert mask.all()  # Equal pooled scores preserve the original tie policy.
        assert query.dtype == torch.float16 and precise == 1
        return torch.ones_like(query) / 256

    monkeypatch.setattr(hybrid, "_native_mask_forward", native)
    out = hybrid.hybrid_attention(q, k, v, plan, lambda q, k, v, p: torch.ones_like(q), sparsity=0.8, scale=8**-0.5)
    assert torch.equal(out, torch.ones_like(q))
