# SPDX-License-Identifier: Apache-2.0
"""Tests for the ARDiffusionKVState paged-attention pipeline bridge."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionPagedLayerContext,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
POS = "positive"
NEG = "negative"


def make_state(num_layers=1, window_chunks=4, cross_attn_length=0, shared_local_index=False):
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window_chunks)
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=num_layers,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 24,
        kv_branches=(
            ARDiffusionKVBranchSpec(POS, 0),
            ARDiffusionKVBranchSpec(NEG, 0 if shared_local_index else 1),
        ),
        session_capacity=2,
        cross_attention_lengths={"text": cross_attn_length} if cross_attn_length else None,
        device=torch.device("cpu"),
    )
    pos = kv.begin_request("r-pos")
    neg = kv.begin_request("r-neg")
    return kv, ARDiffusionKVState(kv, "s1", {POS: pos, NEG: neg}, num_layers=num_layers)


def _prepare_and_commit(st: ARDiffusionKVState, kv_branch: str, n_chunks: int) -> None:
    ctx = st.get_kv_caches(kv_branch, seq_len=n_chunks * BLOCK, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    st.commit_paged_context(kv_branch)


def test_get_returns_paged_layer_contexts_when_nothing_committed():
    _, st = make_state(num_layers=2)
    for kv_branch in (POS, NEG):
        contexts = st.get_kv_caches(kv_branch, seq_len=BLOCK, commit_current=False)
        assert len(contexts) == 2
        assert all(isinstance(ctx, ARDiffusionPagedLayerContext) for ctx in contexts)
        assert contexts[0].history_block_ids == []


def test_managed_paged_context_commits_only_after_forward():
    kv, st = make_state(num_layers=1)
    contexts = st.get_kv_caches(POS, seq_len=2 * BLOCK, commit_current=True)
    ctx = contexts[0].forward_ctx

    assert st.adapter(POS).completed_chunks == 0
    assert kv.window_block_ids(st.adapter(POS)) == []
    ctx.ensure_video_slots(torch.device("cpu"))
    assert st.adapter(POS).completed_chunks == 0
    assert len(ctx.current_video_block_ids) == 2

    st.commit_paged_context(POS)

    assert st.adapter(POS).completed_chunks == 2
    assert st._committed[POS] == 2 * BLOCK
    assert len(kv.window_block_ids(st.adapter(POS))) == 2


def test_scratch_paged_context_does_not_commit():
    kv, st = make_state(num_layers=1)
    contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)
    ctx = contexts[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))

    assert ctx.current_video_block_ids == kv.scratch_block_ids(POS, 0, 1)
    st.commit_paged_context(POS)

    assert st.adapter(POS).completed_chunks == 0
    assert st._committed[POS] == 0
    assert kv.window_block_ids(st.adapter(POS)) == []


def test_eviction_bounds_resident_window():
    kv, st = make_state(num_layers=1, window_chunks=3)
    _prepare_and_commit(st, POS, 3)
    assert len(kv.window_block_ids(st.adapter(POS))) == 3

    _prepare_and_commit(st, POS, 2)
    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
    visible_blocks, video_len = ctx.video_block_table(torch.device("cpu"))
    assert len(visible_blocks) == 3
    assert video_len == 3 * BLOCK


def test_branches_are_independent():
    kv, st = make_state()
    _prepare_and_commit(st, POS, 1)
    assert len(kv.window_block_ids(st.adapter(POS))) == 1
    assert kv.window_block_ids(st.adapter(NEG)) == []


def test_reset_clears_session_window():
    kv, st = make_state(num_layers=1, window_chunks=4)
    _prepare_and_commit(st, POS, 2)
    _prepare_and_commit(st, NEG, 1)
    assert st._committed[POS] == 2 * BLOCK and st._committed[NEG] == BLOCK
    free_before = kv.manager.block_pool.get_num_free_blocks()

    st.reset()

    assert st._committed == {POS: 0, NEG: 0}
    assert st._paged_pending == {POS: None, NEG: None}
    assert kv.window_block_ids(st.adapter(POS)) == []
    assert kv.window_block_ids(st.adapter(NEG)) == []
    assert kv.manager.block_pool.get_num_free_blocks() > free_before
    _prepare_and_commit(st, POS, 1)
    assert len(kv.window_block_ids(st.adapter(POS))) == 1


def test_get_cross_attention_kv_raises_before_populate():
    _, st = make_state(cross_attn_length=8)
    for kv_branch in (POS, NEG):
        with pytest.raises(RuntimeError, match="before it was populated"):
            st.get_cross_attention_kv(kv_branch, "text")


def test_get_cross_attention_kv_returns_session_pool_dicts_when_populated():
    L = 8
    kv, st = make_state(num_layers=2, cross_attn_length=L)
    written = []
    for _ in range(2):
        k = torch.randn(1, L, N_HEADS, HEAD_DIM)
        v = torch.randn(1, L, N_HEADS, HEAD_DIM)
        written.append((k, v))
    st.populate_cross_attention(POS, "text", written)

    out = st.get_cross_attention_kv(POS, "text")
    assert len(out) == 2
    for i, (k, v) in enumerate(written):
        assert out[i]["is_init"] is True
        assert out[i]["k"].shape == (1, L, N_HEADS, HEAD_DIM)
        assert torch.equal(out[i]["k"], k)
        assert torch.equal(out[i]["v"], v)


def test_window_reset_can_keep_named_cross_attention_cache():
    kv, st = make_state(num_layers=1, cross_attn_length=8)
    k = torch.randn(1, 8, N_HEADS, HEAD_DIM)
    v = torch.randn(1, 8, N_HEADS, HEAD_DIM)
    st.populate_cross_attention(POS, "text", [(k, v)])
    old_adapter = st.adapter(POS)

    st.reset(keep_cross_attention=("text",))

    assert st.adapter(POS) is not old_adapter
    assert torch.equal(st.get_cross_attention_kv(POS, "text")[0]["k"], k)
    assert "s1" in kv._cross_sessions

    st.reset()
    assert "s1" not in kv._cross_sessions
    assert not st.is_cross_attention_populated(POS, "text")


def test_cross_attention_uses_logical_branch_when_local_slot_is_shared():
    _, st = make_state(num_layers=2, cross_attn_length=8, shared_local_index=True)
    assert st.kv_cache.num_local_kv_branches == 1
    pos_layers = [
        (torch.full((1, 8, N_HEADS, HEAD_DIM), 1.0), torch.full((1, 8, N_HEADS, HEAD_DIM), 2.0)) for _ in range(2)
    ]
    neg_layers = [
        (torch.full((1, 8, N_HEADS, HEAD_DIM), 3.0), torch.full((1, 8, N_HEADS, HEAD_DIM), 4.0)) for _ in range(2)
    ]

    st.populate_cross_attention(POS, "text", pos_layers)
    st.populate_cross_attention(NEG, "text", neg_layers)

    for layer in st.get_cross_attention_kv(POS, "text"):
        assert torch.count_nonzero(layer["k"] != 1.0) == 0
        assert torch.count_nonzero(layer["v"] != 2.0) == 0
    for layer in st.get_cross_attention_kv(NEG, "text"):
        assert torch.count_nonzero(layer["k"] != 3.0) == 0
        assert torch.count_nonzero(layer["v"] != 4.0) == 0


def test_cross_attention_population_is_transactional():
    kv, st = make_state(num_layers=2, cross_attn_length=8)
    original = [
        (torch.full((1, 8, N_HEADS, HEAD_DIM), value), torch.full((1, 8, N_HEADS, HEAD_DIM), -value))
        for value in (1.0, 2.0)
    ]
    replacement = (
        torch.full((1, 8, N_HEADS, HEAD_DIM), 9.0),
        torch.full((1, 8, N_HEADS, HEAD_DIM), -9.0),
    )

    def interrupted_population():
        yield replacement
        raise RuntimeError("projection failed")

    with pytest.raises(RuntimeError, match="projection failed"):
        st.populate_cross_attention(POS, "text", interrupted_population())
    assert not st.is_cross_attention_populated(POS, "text")
    assert "s1" not in kv._cross_sessions

    st.populate_cross_attention(POS, "text", original)

    with pytest.raises(RuntimeError, match="projection failed"):
        st.populate_cross_attention(POS, "text", interrupted_population())
    out = st.get_cross_attention_kv(POS, "text")
    for layer_index, layer in enumerate(out):
        assert torch.equal(layer["k"], original[layer_index][0])
        assert torch.equal(layer["v"], original[layer_index][1])

    for invalid_layers in (original[:1], [*original, replacement]):
        with pytest.raises(ValueError, match="expected 2 layers"):
            st.populate_cross_attention(POS, "text", invalid_layers)
        out = st.get_cross_attention_kv(POS, "text")
        for layer_index, layer in enumerate(out):
            assert torch.equal(layer["k"], original[layer_index][0])
            assert torch.equal(layer["v"], original[layer_index][1])

    invalid_shape = (torch.empty(1, 7, N_HEADS, HEAD_DIM), original[0][1])
    with pytest.raises(ValueError, match="expected k/v shape"):
        st.populate_cross_attention(POS, "text", [invalid_shape, original[1]])
    out = st.get_cross_attention_kv(POS, "text")
    assert torch.equal(out[0]["k"], original[0][0])


def test_closed_state_cannot_repopulate_cross_attention():
    kv, st = make_state(num_layers=1, cross_attn_length=8)
    layer = (torch.ones(1, 8, N_HEADS, HEAD_DIM), torch.ones(1, 8, N_HEADS, HEAD_DIM))
    st.close()

    with pytest.raises(RuntimeError, match="is closed"):
        st.populate_cross_attention(POS, "text", [layer])
    assert "s1" not in kv._cross_sessions


def test_kv_create_is_noop_engine_owns_allocation():
    from unittest.mock import MagicMock

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    p = DreamZeroPipeline.__new__(DreamZeroPipeline)
    p._ar_diffusion_kv_state = object()
    state = MagicMock()
    p._kv_create(state, 1, "float32", "cpu", 24, 4, 64)
    state.create_kv_caches.assert_not_called()


def test_get_kv_cache_requires_frame_aligned_seqlen():
    _, st = make_state()
    with pytest.raises(AssertionError, match="frame-aligned"):
        st.get_kv_caches(POS, seq_len=BLOCK + 1, commit_current=True)


def test_prepare_one_branch_allocates_nothing_for_the_other():
    """CFG-parallel laziness: a rank prepares only the kv_branch it runs."""
    kv, st = make_state(num_layers=1, window_chunks=4)
    pos_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    neg_ctx = st.get_kv_caches(NEG, seq_len=BLOCK, commit_current=True)[0].forward_ctx

    pos_ctx.prepare(device=torch.device("cpu"), action_len=0, query_len=BLOCK)

    assert pos_ctx._allocated_video
    assert not neg_ctx._allocated_video
    assert kv.window_block_ids(st.adapter(NEG)) == []
