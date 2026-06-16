# SPDX-License-Identifier: Apache-2.0
"""Tests for the BDEKVState pipeline bridge (Step 5)."""

import torch
from vllm_omni.bde.kv_cache import BDEKVCache, BDEKVConfig
from vllm_omni.bde.kv_cache.gather import BDEKVState

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def make_state(num_layers=2):
    cfg = BDEKVConfig(enable=True, chunk_size=BLOCK, window_chunks=2)
    kv = BDEKVCache(cfg, num_layers=num_layers, num_kv_heads=N_HEADS, head_size=HEAD_DIM,
                    dtype=torch.float32, block_size=BLOCK, max_model_len=4096,
                    available_bytes=1 << 24, device=torch.device("cpu"))
    pos = kv.begin_request("r-pos")
    neg = kv.begin_request("r-neg")
    return kv, BDEKVState(kv, pos, neg, num_layers=num_layers)


# Approach 1: BDEKVState holds the model's window tensors opaquely (shape-agnostic),
# falling back to the model-local empty cache until the first write.


def test_get_returns_fallback_when_empty():
    _, st = make_state()
    sentinel = ["FALLBACK"]
    assert st.get_kv_caches(False, lambda: sentinel) is sentinel
    assert st.get_kv_caches(True, lambda: sentinel) is sentinel


def test_update_then_get_roundtrips_exactly():
    _, st = make_state(num_layers=2)
    # Variable-length / arbitrary-shape window tensors round-trip unchanged.
    tensors = [torch.randn(2, 5, N_HEADS, HEAD_DIM), torch.randn(2, 9, N_HEADS, HEAD_DIM)]
    for i, t in enumerate(tensors):
        st.update_kv_cache(i, t, False)
    got = st.get_kv_caches(False, lambda: ["UNUSED"])
    assert [g is t for g, t in zip(got, tensors)] == [True, True]


def test_branches_are_independent():
    _, st = make_state()
    kpos = torch.randn(2, 3, N_HEADS, HEAD_DIM)
    st.update_kv_cache(0, kpos, False)
    st.update_kv_cache(1, kpos, False)
    # Negative branch still empty -> fallback; positive returns the stored tensor.
    assert st.get_kv_caches(True, lambda: ["NEG_EMPTY"]) == ["NEG_EMPTY"]
    assert st.get_kv_caches(False, lambda: ["X"])[0] is kpos


def test_commit_is_noop():
    _, st = make_state()
    st.commit_chunk()  # window-store mode: no per-chunk bookkeeping, no error


def test_kv_state_noop():
    """When _bde_kv_state is None, proxy methods fall through — verified by
    the existing tests passing unchanged (the pipeline's default __init__
    sets _bde_kv_state = None)."""
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
    p = DreamZeroPipeline.__new__(DreamZeroPipeline)
    assert p._bde_kv_state is None
    for attr in ("_kv_get", "_kv_create", "_kv_update", "_kv_commit"):
        assert hasattr(p, attr), f"proxy method {attr} missing"


def test_kv_create_always_initializes_model_local():
    """_kv_create must call state.create_kv_caches (no recursion), and must do so
    even under BDE so cross-attention (not managed by BDE) is initialized."""
    from unittest.mock import MagicMock
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    p = DreamZeroPipeline.__new__(DreamZeroPipeline)
    for bde_state in (None, object()):  # disabled and enabled both init model-local
        p._bde_kv_state = bde_state
        state = MagicMock()
        p._kv_create(state, 1, "float32", "cpu", 24, 4, 64)
        state.create_kv_caches.assert_called_once_with(1, "float32", "cpu", 24, 4, 64)
