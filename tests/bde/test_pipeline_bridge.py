# SPDX-License-Identifier: Apache-2.0
"""Tests for the BDEKVState pipeline bridge (Step 5)."""

import pytest
import torch

from vllm_omni.experimental.bde.kv_cache import BDEKVCache, BDEKVConfig
from vllm_omni.experimental.bde.kv_cache.state import BDEKVState

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def make_state(num_layers=1, window_chunks=4, cross_attn_length=0):
    cfg = BDEKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window_chunks)
    kv = BDEKVCache(
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
    return kv, BDEKVState(kv, pos, neg, num_layers=num_layers)


def _window(n_chunks):
    """A model-style window tensor (2, 1, n_chunks*BLOCK, heads, head_dim)."""
    return torch.randn(2, 1, n_chunks * BLOCK, N_HEADS, HEAD_DIM)


# Approach 2: BDEKVState writes the new tokens into paged pool blocks and gathers
# the resident window back; output mirrors the model-local sliding window.


def test_get_returns_empty_window_when_nothing_committed():
    """The engine owns the read end-to-end: with nothing committed the adapter has
    no resident blocks, so gather returns zero-length windows — the concat-identity
    the model's attention expects on the first step — with no model-local fallback."""
    _, st = make_state()
    for neg in (False, True):
        windows = st.get_kv_caches(neg)
        assert len(windows) == 1
        assert windows[0].shape == (2, 1, 0, N_HEADS, HEAD_DIM)


def test_paged_write_then_gather_roundtrips():
    _, st = make_state(num_layers=1)
    win = _window(2)  # this forward appends 2 frame-chunks
    st.update_kv_cache(0, win, False, seq_len=2 * BLOCK)
    got = st.get_kv_caches(False)[0]  # (2, 1, W, heads, head_dim)
    assert got.shape == (2, 1, 2 * BLOCK, N_HEADS, HEAD_DIM)
    # The gathered window equals the tokens we wrote (bit-exact).
    assert torch.allclose(got[0, 0], win[0, 0], atol=0)
    assert torch.allclose(got[1, 0], win[1, 0], atol=0)


def test_gather_is_memoized_within_a_forward():
    """The resident window is frozen across a forward's denoise steps, so repeated
    get_kv_caches must reuse one gather; a write-back (layer 0) invalidates it."""
    _, st = make_state(num_layers=2)
    st.update_kv_cache(0, _window(2), False, seq_len=2 * BLOCK)  # commit one chunk
    first = st.get_kv_caches(False)
    again = st.get_kv_caches(False)
    assert again is first  # cache HIT: identical object, no re-gather
    # A new write-back grows the window -> next get must re-gather a fresh object.
    st.update_kv_cache(0, _window(1), False, seq_len=BLOCK)
    after = st.get_kv_caches(False)
    assert after is not first
    # Negative branch is cached independently.
    st.update_kv_cache(0, _window(1), True, seq_len=BLOCK)
    neg1 = st.get_kv_caches(True)
    assert st.get_kv_caches(True) is neg1
    assert neg1 is not after


def test_reset_clears_gather_cache():
    _, st = make_state(num_layers=1)
    st.update_kv_cache(0, _window(2), False, seq_len=2 * BLOCK)
    st.get_kv_caches(False)
    assert st._gather_cache[False] is not None
    st.reset()
    assert st._gather_cache == {False: None, True: None}


def test_eviction_bounds_resident_window():
    kv, st = make_state(num_layers=1, window_chunks=3)
    st.update_kv_cache(0, _window(3), False, seq_len=3 * BLOCK)  # resident 3/3
    st.update_kv_cache(0, _window(5), False, seq_len=2 * BLOCK)  # +2 -> evict to 3
    got = st.get_kv_caches(False)[0]
    assert got.shape[2] <= 3 * BLOCK  # window bounded


def test_branches_are_independent():
    _, st = make_state()
    st.update_kv_cache(0, _window(1), False, seq_len=BLOCK)
    # Negative branch never written -> empty (zero-length) window.
    assert st.get_kv_caches(True)[0].shape[2] == 0
    assert st.get_kv_caches(False)[0].shape[2] == BLOCK


def test_reset_clears_session_window():
    """reset() drops the resident window so the next forward starts fresh —
    mirrors DreamZero's window-boundary state.reset()."""
    kv, st = make_state(num_layers=1, window_chunks=4)
    st.update_kv_cache(0, _window(2), False, seq_len=2 * BLOCK)
    st.update_kv_cache(0, _window(1), True, seq_len=BLOCK)
    assert st._committed[False] == 2 * BLOCK and st._committed[True] == BLOCK
    free_before = kv.manager.block_pool.get_num_free_blocks()

    st.reset()

    # Both branches gather empty windows again, and pool blocks are freed.
    assert st._committed == {False: 0, True: 0}
    assert st._pending == {False: [], True: []}
    assert st.get_kv_caches(False)[0].shape[2] == 0
    assert st.get_kv_caches(True)[0].shape[2] == 0
    assert kv.manager.block_pool.get_num_free_blocks() > free_before
    # Adapters are live again under the same ids — a post-reset write works.
    st.update_kv_cache(0, _window(1), False, seq_len=BLOCK)
    assert st.get_kv_caches(False)[0].shape[2] == BLOCK


def test_get_cross_kv_caches_raises_before_populate():
    """Engine ownership guard: a cross-attn read before _kv_populate_cross must
    fail loud rather than fall back to the model (which would self-project KV)."""
    _, st = make_state()  # cross_attn_length == 0, nothing populated
    for neg in (False, True):
        with pytest.raises(RuntimeError, match="cross-attn read before"):
            st.get_cross_kv_caches(neg)
    # Guards the full AND: _cross_populated alone is not enough without a pool.
    st._cross_populated[False] = True
    with pytest.raises(RuntimeError, match="cross-attn read before"):
        st.get_cross_kv_caches(False)


def test_get_cross_kv_caches_returns_pool_dicts_when_populated():
    """Once the cross pool is filled and marked populated, the read returns the
    pool-backed dicts (is_init=True) bit-exactly — no model fallback involved."""
    L = 8
    kv, st = make_state(num_layers=2, cross_attn_length=L)
    written = []
    for i in range(2):
        k = torch.randn(1, L, N_HEADS, HEAD_DIM)
        v = torch.randn(1, L, N_HEADS, HEAD_DIM)
        kv.write_cross_kv(i, False, k, v)
        written.append((k, v))
    st._cross_populated[False] = True

    out = st.get_cross_kv_caches(False)
    assert len(out) == 2
    for i, (k, v) in enumerate(written):
        assert out[i]["is_init"] is True
        assert out[i]["k"].shape == (1, L, N_HEADS, HEAD_DIM)
        assert torch.equal(out[i]["k"], k)
        assert torch.equal(out[i]["v"], v)


def test_commit_is_noop():
    _, st = make_state()
    st.commit_chunk()  # paged mode keeps resident windows; commit just logs


def test_kv_state_noop():
    """When _bde_kv_state is None, proxy methods fall through — verified by
    the existing tests passing unchanged (the pipeline's default __init__
    sets _bde_kv_state = None)."""
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    p = DreamZeroPipeline.__new__(DreamZeroPipeline)
    assert p._bde_kv_state is None
    for attr in ("_kv_get", "_kv_create", "_kv_update", "_kv_commit"):
        assert hasattr(p, attr), f"proxy method {attr} missing"


def test_kv_create_owned_by_engine_under_bde():
    """_kv_create initializes model-local caches ONLY on the non-BDE path. Under
    BDE the pool owns all allocation, so state.create_kv_caches must NOT be called."""
    from unittest.mock import MagicMock

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    p = DreamZeroPipeline.__new__(DreamZeroPipeline)

    # Non-BDE: model-local caches are created (no recursion).
    p._bde_kv_state = None
    state = MagicMock()
    p._kv_create(state, 1, "float32", "cpu", 24, 4, 64)
    state.create_kv_caches.assert_called_once_with(1, "float32", "cpu", 24, 4, 64)

    # BDE active: allocation is engine-owned -> no model-local creation.
    p._bde_kv_state = object()
    state = MagicMock()
    p._kv_create(state, 1, "float32", "cpu", 24, 4, 64)
    state.create_kv_caches.assert_not_called()
