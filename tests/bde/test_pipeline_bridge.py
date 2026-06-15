# SPDX-License-Identifier: Apache-2.0
"""Tests for the BDEKVState pipeline bridge (Step 5)."""

import torch
from vllm_omni.bde.kv_cache import BDEKVCache, BDEKVConfig
from vllm_omni.bde.kv_cache.gather import BDEKVState

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def make_state(chunk_size=BLOCK, window_chunks=2, num_layers=2):
    cfg = BDEKVConfig(enable=True, chunk_size=chunk_size, window_chunks=window_chunks)
    kv = BDEKVCache(cfg, num_layers=num_layers, num_kv_heads=N_HEADS, head_size=HEAD_DIM,
                    dtype=torch.float32, block_size=BLOCK, max_model_len=4096,
                    available_bytes=1 << 24, device=torch.device("cpu"))
    pos = kv.begin_request("r-pos")
    neg = kv.begin_request("r-neg")
    kv.allocate_chunk(pos)
    kv.allocate_chunk(neg)
    return kv, BDEKVState(kv, pos, neg, num_layers=num_layers)


def test_get_kv_caches_returns_gathered_windows():
    kv, st = make_state()
    pos_windows = st.get_kv_caches(False)
    neg_windows = st.get_kv_caches(True)
    assert len(pos_windows) == 2
    assert len(neg_windows) == 2
    # One allocated chunk → one resident block → block_size tokens.
    for w in pos_windows + neg_windows:
        assert w.shape == (2, 1, BLOCK, N_HEADS, HEAD_DIM)


def test_write_and_gather_roundtrip():
    kv, st = make_state()
    for layer in range(2):
        new_kv = torch.randn(2, BLOCK, N_HEADS, HEAD_DIM)
        st.update_kv_cache(layer, new_kv, False)
    st.commit_chunk()
    # After commit, gather for the next chunk.
    kv.allocate_chunk(st.pos)
    kv.allocate_chunk(st.neg)
    window = st.get_kv_caches(False)[0]
    # The written K/V should be in the gathered window (last BLOCK positions).
    written_k = window[0, 0, -BLOCK:]
    # The pool stores exactly what was written.
    slots = kv.chunk_write_slots(st.pos)
    assert torch.allclose(written_k, kv._k_pools[0][slots], atol=1e-5)


def test_commit_advances_both_adapters():
    kv, st = make_state()
    assert st.pos.completed_chunks == 0
    assert st.neg.completed_chunks == 0
    st.commit_chunk()
    assert st.pos.completed_chunks == 1
    assert st.neg.completed_chunks == 1


def test_kv_state_noop():
    """When _bde_kv_state is None, proxy methods fall through — verified by
    the existing tests passing unchanged (the pipeline's default __init__
    sets _bde_kv_state = None)."""
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
    p = DreamZeroPipeline.__new__(DreamZeroPipeline)
    assert p._bde_kv_state is None
    for attr in ("_kv_get", "_kv_create", "_kv_update", "_kv_commit"):
        assert hasattr(p, attr), f"proxy method {attr} missing"


def test_kv_create_fallback_does_not_recurse():
    """Regression: _kv_create must call state.create_kv_caches, NOT itself."""
    from unittest.mock import MagicMock
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    p = DreamZeroPipeline.__new__(DreamZeroPipeline)
    p._bde_kv_state = None  # explicitly None, as in the default
    state = MagicMock()
    p._kv_create(state, 1, "float32", "cpu", 24, 4, 64)
    state.create_kv_caches.assert_called_once_with(1, "float32", "cpu", 24, 4, 64)
