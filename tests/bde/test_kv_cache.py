# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the BDE KV cache helpers (Phase 1, PR-2).

Covers the request adapter, the chunk-window spec/manager (registration + the
eviction policy), and the pool builder — exercised against the installed vLLM
KV stack on CPU (block bookkeeping only, no GPU tensors).
"""

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import RequestStatus

from vllm_omni.experimental.bde.kv_cache import (
    BDEKVConfig,
    BDERequestAdapter,
    ChunkWindowManager,
    ChunkWindowSpec,
    build_kv_manager,
    compute_num_blocks,
)
from vllm_omni.experimental.bde.kv_cache.chunk_window import chunk_window_skipped_tokens

BLOCK = 16


def make_spec(*, chunk_size=BLOCK, window_chunks=2, sink_chunks=0, reset_at_boundary=False):
    return ChunkWindowSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        sliding_window=window_chunks * chunk_size,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        sink_chunks=sink_chunks,
        reset_at_boundary=reset_at_boundary,
    )


# --- ChunkWindowSpec registration -------------------------------------------


def test_spec_registration_resolves_to_chunk_window_manager():
    # Without explicit registration the MRO walk would fall back to the parent
    # SlidingWindowManager; assert the subclass manager wins.
    assert KVCacheSpecRegistry.get_manager_class(make_spec()) is ChunkWindowManager


def test_spec_kind_is_sliding_window():
    assert get_kv_cache_spec_kind(make_spec()) == KVCacheSpecKind.SLIDING_WINDOW


def test_spec_rejects_inconsistent_window():
    with pytest.raises(ValueError):
        ChunkWindowSpec(
            block_size=BLOCK,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float16,
            sliding_window=99,  # != window_chunks * chunk_size
            chunk_size=BLOCK,
            window_chunks=2,
        )


# --- eviction policy (pure) -------------------------------------------------


def test_sliding_replace_keeps_window():
    # window = 2 chunks * 16 = 32. Base sliding formula keeps `window` tokens;
    # the snap is to chunk boundaries.
    def skip(n):
        return chunk_window_skipped_tokens(n, chunk_size=16, sliding_window=32, sink_chunks=0, reset_at_boundary=False)

    assert skip(32) == 0  # nothing past the window yet
    assert skip(48) == 16  # one chunk fell out of the window
    assert skip(64) == 32


def test_sliding_replace_snaps_to_chunk_boundary():
    # A non-chunk-aligned overflow must snap down so a chunk is never half-evicted.
    skip = chunk_window_skipped_tokens(50, chunk_size=16, sliding_window=32, sink_chunks=0, reset_at_boundary=False)
    assert skip % 16 == 0 and skip == 16


def test_sink_chunks_protected():
    # sink = 1 chunk (16 tokens) is never skipped.
    skip = chunk_window_skipped_tokens(80, chunk_size=16, sliding_window=32, sink_chunks=1, reset_at_boundary=False)
    # raw overflow = 80 - 32 - 16 = 32 -> snapped 32
    assert skip == 32


def test_reset_at_boundary_drops_completed_past_sink():
    skip = chunk_window_skipped_tokens(48, chunk_size=16, sliding_window=32, sink_chunks=1, reset_at_boundary=True)
    # completed = 48; sink = 16 -> drop 32
    assert skip == 32


# --- BDEKVConfig ------------------------------------------------------------


def test_kv_config_sliding_window_property():
    assert BDEKVConfig(chunk_size=16, window_chunks=3).sliding_window == 48
    assert BDEKVConfig(chunk_size=16, window_chunks=None).sliding_window is None


def test_kv_config_disabled_by_default():
    assert BDEKVConfig().enable is False


# --- BDERequestAdapter ------------------------------------------------------


def test_adapter_advances_per_chunk_not_per_step():
    a = BDERequestAdapter("r0", chunk_size=16)
    assert a.num_computed_tokens == 0
    assert a.num_tokens == 16  # in-flight chunk
    a.on_chunk_committed()
    assert a.num_computed_tokens == 16
    assert a.num_tokens == 32


def test_adapter_accounts_for_prefill_prefix():
    a = BDERequestAdapter("r0", chunk_size=16, prefill_prefix_tokens=4)
    assert a.num_computed_tokens == 4
    assert a.num_prompt_tokens == 4
    assert a.num_tokens == 20


def test_adapter_status_is_vllm_enum():
    assert isinstance(BDERequestAdapter("r0", chunk_size=16).status, RequestStatus)


# --- pool / manager ---------------------------------------------------------


def test_compute_num_blocks():
    # 1 MiB budget at fraction 0.5 with 16 KiB pages -> 32 blocks.
    assert compute_num_blocks(1 << 20, 0.5, 16 << 10) == 32


def test_compute_num_blocks_validates():
    with pytest.raises(ValueError):
        compute_num_blocks(1 << 20, 0.5, 0)
    with pytest.raises(ValueError):
        compute_num_blocks(1 << 20, 1.5, 16 << 10)


def test_build_manager_allocate_free_roundtrip():
    """End-to-end: a BDERequestAdapter drives a real KVCacheManager.

    This is the adapter conformance check — if the adapter were missing an
    attribute the manager reads, allocate_slots/free would raise here.
    """
    spec = make_spec()
    mgr = build_kv_manager(spec, ["layer0"], num_blocks=16, max_model_len=1024)
    free_before = mgr.block_pool.get_num_free_blocks()

    adapter = BDERequestAdapter("req-0", chunk_size=BLOCK)
    blocks = mgr.allocate_slots(adapter, num_new_tokens=BLOCK, full_sequence_must_fit=True)
    assert blocks is not None
    assert mgr.block_pool.get_num_free_blocks() < free_before

    mgr.free(adapter)
    assert mgr.block_pool.get_num_free_blocks() == free_before


def test_window_eviction_plateaus_pool_memory():
    """Multi-chunk rollout: out-of-window blocks are recycled, so pool memory
    plateaus at ~window size instead of growing with rollout length.

    This is the core promise of the chunk window — drive a real KVCacheManager
    over many chunks and assert the free-block count stops decreasing once the
    window is full, and the resident (non-null) blocks stay bounded.
    """
    spec = make_spec(chunk_size=BLOCK, window_chunks=2)  # window = 2 chunks
    mgr = build_kv_manager(spec, ["l0"], num_blocks=64, max_model_len=4096)
    null_id = mgr.block_pool.null_block.block_id
    adapter = BDERequestAdapter("req", chunk_size=BLOCK)

    free_after = []
    distinct_real = []
    for _ in range(10):
        mgr.allocate_slots(adapter, num_new_tokens=BLOCK)
        ids = mgr.get_block_ids(adapter.request_id)[0]
        distinct_real.append(len({b for b in ids if b != null_id}))
        free_after.append(mgr.block_pool.get_num_free_blocks())
        adapter.on_chunk_committed()

    # Free-block count stops decreasing once the window is full (warmup = window).
    assert free_after[-1] == free_after[spec.window_chunks]
    assert free_after[-1] == free_after[-5]  # flat across the tail
    # Resident non-null blocks bounded by window + the in-flight chunk.
    assert max(distinct_real) <= spec.window_chunks + 1
