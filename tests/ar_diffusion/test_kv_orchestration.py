# SPDX-License-Identifier: Apache-2.0
"""Tests for ARDiffusionKVCache — the engine-level KV orchestration body (Phase 1)."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig

DIMS = dict(num_layers=2, num_kv_heads=4, head_size=64, dtype=torch.float16, block_size=16)


def make_cache(*, chunk_size=16, window_chunks=2, available_bytes=1 << 24):
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=chunk_size, window_chunks=window_chunks)
    return ARDiffusionKVCache(cfg, max_model_len=4096, available_bytes=available_bytes, **DIMS)


def test_requires_enabled_config():
    with pytest.raises(ValueError):
        ARDiffusionKVCache(ARDiffusionKVConfig(enable=False), max_model_len=256, available_bytes=1 << 20, **DIMS)


def test_requires_bounded_window():
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=16, window_chunks=None)
    with pytest.raises(ValueError):
        ARDiffusionKVCache(cfg, max_model_len=256, available_bytes=1 << 20, **DIMS)


def test_full_request_lifecycle_and_eviction():
    """begin -> per-chunk allocate/slots/commit over a long rollout -> free.

    Exercises the orchestrator end-to-end and asserts the chunk window bounds
    memory (pool plateaus) and frees cleanly.
    """
    kv = make_cache(chunk_size=16, window_chunks=2)
    free_total = kv.manager.block_pool.get_num_free_blocks()

    adapter = kv.begin_request("req-0")
    free_after = []
    for k in range(10):
        block_table = kv.allocate_chunk(adapter)
        slots = kv.chunk_write_slots(adapter)
        # Slots target real blocks in this chunk's table; length == chunk_size.
        assert len(slots) == kv.spec.chunk_size
        used = {int(s) // kv.block_size for s in slots}
        assert kv.null_block_id not in used
        assert used <= set(block_table)
        # Resident window stays bounded by window + the in-flight chunk.
        assert len(kv.window_block_ids(adapter)) <= kv.spec.window_chunks + 1
        free_after.append(kv.manager.block_pool.get_num_free_blocks())
        kv.commit_chunk(adapter)

    # Pool memory plateaus once the window is full (eviction recycles blocks).
    assert free_after[-1] == free_after[kv.spec.window_chunks]

    kv.end_request(adapter)
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_num_computed_advances_per_chunk():
    kv = make_cache(chunk_size=16, window_chunks=2)
    a = kv.begin_request("r")
    assert a.num_computed_tokens == 0
    kv.allocate_chunk(a)
    kv.commit_chunk(a)
    assert a.num_computed_tokens == 16
    kv.end_request(a)


def test_state_close_frees_both_branch_blocks():
    """ARDiffusionKVState.close() returns both CFG branches' pool blocks to the pool.

    This is the primitive the runner's LRU eviction relies on: when a session is
    evicted, close() must free the blocks both adapters hold, or session churn
    leaks pool ownership (review P1).
    """
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

    kv = make_cache(chunk_size=16, window_chunks=2)
    free_total = kv.manager.block_pool.get_num_free_blocks()

    pos = kv.begin_request("bde__s")
    neg = kv.begin_request("bde__s__neg")
    state = ARDiffusionKVState(kv, pos, neg, num_layers=kv.num_layers)
    for _ in range(3):
        for adapter in (pos, neg):
            kv.allocate_chunk(adapter)
            kv.commit_chunk(adapter)
    # Both branches hold resident blocks now.
    assert kv.manager.block_pool.get_num_free_blocks() < free_total

    state.close()
    # All blocks returned — no leak across an evicted session.
    assert kv.manager.block_pool.get_num_free_blocks() == free_total
