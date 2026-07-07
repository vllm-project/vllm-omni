# SPDX-License-Identifier: Apache-2.0
"""Tests for the ARDiffusionKVState paged-attention pipeline bridge."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionPagedLayerContext,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def make_state(num_layers=1, window_chunks=4, cross_attn_length=0):
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
        cross_attn_length=cross_attn_length,
        device=torch.device("cpu"),
    )
    pos = kv.begin_request("r-pos")
    neg = kv.begin_request("r-neg")
    return kv, ARDiffusionKVState(kv, pos, neg, num_layers=num_layers)


def _prepare_and_commit(st: ARDiffusionKVState, is_negative: bool, n_chunks: int) -> None:
    ctx = st.get_kv_caches(is_negative, seq_len=n_chunks * BLOCK, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    st.commit_paged_context(is_negative)


def test_get_returns_paged_layer_contexts_when_nothing_committed():
    _, st = make_state(num_layers=2)
    for neg in (False, True):
        contexts = st.get_kv_caches(neg, seq_len=BLOCK, commit_current=False)
        assert len(contexts) == 2
        assert all(isinstance(ctx, ARDiffusionPagedLayerContext) for ctx in contexts)
        assert contexts[0].history_block_ids == []


def test_managed_paged_context_commits_only_after_forward():
    kv, st = make_state(num_layers=1)
    contexts = st.get_kv_caches(False, seq_len=2 * BLOCK, commit_current=True)
    ctx = contexts[0].forward_ctx

    assert st.pos.completed_chunks == 0
    assert kv.window_block_ids(st.pos) == []
    ctx.ensure_video_slots(torch.device("cpu"))
    assert st.pos.completed_chunks == 0
    assert len(ctx.current_video_block_ids) == 2

    st.commit_paged_context(False)

    assert st.pos.completed_chunks == 2
    assert st._committed[False] == 2 * BLOCK
    assert len(kv.window_block_ids(st.pos)) == 2


def test_scratch_paged_context_does_not_commit():
    kv, st = make_state(num_layers=1)
    contexts = st.get_kv_caches(False, seq_len=BLOCK, commit_current=False)
    ctx = contexts[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))

    assert ctx.current_video_block_ids == kv.scratch_block_ids(False, 0, 1)
    st.commit_paged_context(False)

    assert st.pos.completed_chunks == 0
    assert st._committed[False] == 0
    assert kv.window_block_ids(st.pos) == []


def test_eviction_bounds_resident_window():
    kv, st = make_state(num_layers=1, window_chunks=3)
    _prepare_and_commit(st, False, 3)
    assert len(kv.window_block_ids(st.pos)) == 3

    _prepare_and_commit(st, False, 2)
    ctx = st.get_kv_caches(False, seq_len=BLOCK, commit_current=False)[0].forward_ctx
    visible_blocks, video_len = ctx.video_block_table(torch.device("cpu"))
    assert len(visible_blocks) == 3
    assert video_len == 3 * BLOCK


def test_branches_are_independent():
    kv, st = make_state()
    _prepare_and_commit(st, False, 1)
    assert len(kv.window_block_ids(st.pos)) == 1
    assert kv.window_block_ids(st.neg) == []


def test_reset_clears_session_window():
    kv, st = make_state(num_layers=1, window_chunks=4)
    _prepare_and_commit(st, False, 2)
    _prepare_and_commit(st, True, 1)
    assert st._committed[False] == 2 * BLOCK and st._committed[True] == BLOCK
    free_before = kv.manager.block_pool.get_num_free_blocks()

    st.reset()

    assert st._committed == {False: 0, True: 0}
    assert st._paged_pending == {False: None, True: None}
    assert kv.window_block_ids(st.pos) == []
    assert kv.window_block_ids(st.neg) == []
    assert kv.manager.block_pool.get_num_free_blocks() > free_before
    _prepare_and_commit(st, False, 1)
    assert len(kv.window_block_ids(st.pos)) == 1


def test_get_cross_kv_caches_raises_before_populate():
    """Engine ownership guard: a cross-attn read before _kv_populate_cross must fail loud."""
    _, st = make_state()  # cross_attn_length == 0, nothing populated
    for neg in (False, True):
        with pytest.raises(RuntimeError, match="cross-attn read before"):
            st.get_cross_kv_caches(neg)
    # Guards the full AND: the populated flag alone is not enough without a pool.
    st._cross_text_populated[False] = True
    with pytest.raises(RuntimeError, match="cross-attn read before"):
        st.get_cross_kv_caches(False)


def test_get_cross_kv_caches_returns_pool_dicts_when_populated():
    L = 8
    kv, st = make_state(num_layers=2, cross_attn_length=L)
    written = []
    for i in range(2):
        k = torch.randn(1, L, N_HEADS, HEAD_DIM)
        v = torch.randn(1, L, N_HEADS, HEAD_DIM)
        kv.write_cross_kv(i, False, k, v)
        written.append((k, v))
    st._cross_text_populated[False] = True

    out = st.get_cross_kv_caches(False)
    assert len(out) == 2
    for i, (k, v) in enumerate(written):
        assert out[i]["is_init"] is True
        assert out[i]["k"].shape == (1, L, N_HEADS, HEAD_DIM)
        assert torch.equal(out[i]["k"], k)
        assert torch.equal(out[i]["v"], v)


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
        st.get_kv_caches(False, seq_len=BLOCK + 1, commit_current=True)


def test_prepare_one_branch_allocates_nothing_for_the_other():
    """CFG-parallel laziness: a rank prepares only the branch it runs."""
    kv, st = make_state(num_layers=1, window_chunks=4)
    pos_ctx = st.get_kv_caches(False, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    neg_ctx = st.get_kv_caches(True, seq_len=BLOCK, commit_current=True)[0].forward_ctx

    pos_ctx.prepare(device=torch.device("cpu"), action_len=0, query_len=BLOCK)

    assert pos_ctx._allocated_video
    assert not neg_ctx._allocated_video
    assert kv.window_block_ids(st.neg) == []


def test_failed_forward_tears_down_session(monkeypatch):
    """Review follow-up (hsliuustc0106): a forward that dies partway must not
    leave allocated-but-uncommitted KV behind — the session is torn down (pool
    blocks freed, model-local state dropped) and the error propagates."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

    kv, st = make_state(num_layers=1, window_chunks=4)
    _prepare_and_commit(st, False, 2)  # session owns pool blocks
    free_before_failure = kv.manager.block_pool.get_num_free_blocks()

    runner = object.__new__(ARDiffusionModelRunner)
    runner.kv_cache = kv
    runner._ar_diffusion_states = __import__("collections").OrderedDict({"s1": st})
    runner._max_ar_diffusion_states = 4
    runner.pipeline = SimpleNamespace(_states={"s1": object()}, _ar_diffusion_kv_state=None)
    runner.device = None
    runner._perf_e2e_times = []

    def boom(self, req):
        raise RuntimeError("layer 17 exploded")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", boom)
    req = SimpleNamespace(
        request_id="r0",
        sampling_params=SimpleNamespace(extra_args={"session_id": "s1"}),
    )

    with pytest.raises(RuntimeError, match="layer 17 exploded"):
        ARDiffusionModelRunner.execute_model(runner, req)

    assert "s1" not in runner._ar_diffusion_states
    assert "s1" not in runner.pipeline._states
    assert runner.pipeline._ar_diffusion_kv_state is None
    # close() freed the session's resident blocks.
    assert kv.manager.block_pool.get_num_free_blocks() > free_before_failure
