# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AR-Diffusion KV cache helpers (Phase 1, PR-2).

Covers the request adapter, the chunk-window spec/manager (registration + the
eviction policy), and the pool builder — exercised against the installed vLLM
KV stack on CPU (block bookkeeping only, no GPU tensors).
"""

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import RequestStatus

from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionRequestAdapter,
    ChunkWindowManager,
    ChunkWindowSpec,
    allocate_kv_pool_with_views,
    build_kv_manager,
    compute_num_blocks,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import chunk_window_skipped_tokens

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
    spec = make_spec()
    assert KVCacheSpecRegistry.get_manager_class(spec) is ChunkWindowManager


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


# --- ARDiffusionKVConfig ------------------------------------------------------------


def test_kv_config_sliding_window_property():
    assert ARDiffusionKVConfig(chunk_size=16, window_chunks=3).sliding_window == 48
    assert ARDiffusionKVConfig(chunk_size=16, window_chunks=None).sliding_window is None


# --- ARDiffusionRequestAdapter ------------------------------------------------------


def test_adapter_advances_per_chunk_not_per_step():
    a = ARDiffusionRequestAdapter("r0", chunk_size=16)
    assert a.num_computed_tokens == 0
    assert a.num_tokens == 16  # in-flight chunk
    a.on_chunk_committed()
    assert a.num_computed_tokens == 16
    assert a.num_tokens == 32


def test_adapter_accounts_for_prefill_prefix():
    a = ARDiffusionRequestAdapter("r0", chunk_size=16, prefill_prefix_tokens=4)
    assert a.num_computed_tokens == 4
    assert a.num_prompt_tokens == 4
    assert a.num_tokens == 20


def test_adapter_status_is_vllm_enum():
    assert isinstance(ARDiffusionRequestAdapter("r0", chunk_size=16).status, RequestStatus)


# --- pool / manager ---------------------------------------------------------


def test_compute_num_blocks():
    # 1 MiB budget at fraction 0.5 with 16 KiB pages -> 32 blocks.
    assert compute_num_blocks(1 << 20, 0.5, 16 << 10) == 32


def test_paged_pool_layout_exposes_flat_slot_views():
    kv_pools, k_pools, v_pools = allocate_kv_pool_with_views(
        num_blocks=4,
        block_size=BLOCK,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert kv_pools[0].shape == (2, 4, BLOCK, 4, 64)
    assert k_pools[0].shape == (4 * BLOCK, 4, 64)
    assert v_pools[0].shape == (4 * BLOCK, 4, 64)

    k_pools[0][BLOCK + 3].fill_(7)
    v_pools[0][2 * BLOCK + 5].fill_(11)
    assert torch.equal(kv_pools[0][0, 1, 3], k_pools[0][BLOCK + 3])
    assert torch.equal(kv_pools[0][1, 2, 5], v_pools[0][2 * BLOCK + 5])


def test_build_manager_allocate_free_roundtrip():
    """End-to-end: a ARDiffusionRequestAdapter drives a real KVCacheManager.

    This is the adapter conformance check — if the adapter were missing an
    attribute the manager reads, allocate_slots/free would raise here.
    """
    spec = make_spec()
    mgr = build_kv_manager(spec, ["layer0"], num_blocks=16, max_model_len=1024)
    free_before = mgr.block_pool.get_num_free_blocks()

    adapter = ARDiffusionRequestAdapter("req-0", chunk_size=BLOCK)
    blocks = mgr.allocate_slots(adapter, num_new_tokens=BLOCK, full_sequence_must_fit=True)
    assert blocks is not None
    assert mgr.block_pool.get_num_free_blocks() < free_before

    mgr.free(adapter)
    assert mgr.block_pool.get_num_free_blocks() == free_before


def test_cross_attn_pool_deducted_from_self_attn_budget():
    """The cross-attn pool is allocated directly; its bytes are subtracted from
    the self-attn paged-pool budget so the two together stay within the free
    memory budget (review: zwhzzz0821)."""
    L = 512
    avail = 1 << 30  # 1 GiB
    kv = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=2, gpu_memory_fraction=0.5),
        num_layers=2,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=avail,
        cross_attn_length=L,
        device=torch.device("cpu"),
    )
    cross_bytes = 2 * 2 * L * 4 * 64 * torch.float16.itemsize * 2  # K+V, pos+neg, layers
    expected = compute_num_blocks(avail - cross_bytes, 0.5, kv.spec.page_size_bytes * 2)
    assert kv.num_blocks == expected
    assert expected > 2 * (2 + 1) + 2  # above the min-blocks floor, so the deduction is what's tested


def _make_kv(*, local_branches, num_frame_per_block=2, window_chunks=9):
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window_chunks),
        num_layers=1,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 16,  # tiny -> the floor binds
        device=torch.device("cpu"),
        local_branches=local_branches,
        num_frame_per_block=num_frame_per_block,
    )


def test_pool_floor_is_branch_aware():
    """CFG-parallel rank (one local branch) sizes for one window + in-flight chunk;
    a single-process run (both branches) sizes for two. Scratch scales the same way."""
    one = _make_kv(local_branches=1)
    two = _make_kv(local_branches=2)

    assert one.managed_num_blocks == 1 * (9 + 2) + 2  # 13
    assert two.managed_num_blocks == 2 * (9 + 2) + 2  # 24
    assert one.scratch_num_blocks == one.scratch_blocks_per_branch
    assert two.scratch_num_blocks == 2 * two.scratch_blocks_per_branch
    assert one.num_blocks_total == 13 + one.scratch_blocks_per_branch


def test_scratch_maps_to_slot_zero_with_one_local_branch():
    """A CFG-parallel rank runs exactly one branch: whichever CFG side it is,
    its scratch lands in the rank's single slot (no dead second slot)."""
    one = _make_kv(local_branches=1)
    assert one.scratch_block_ids(True, 0, 2) == one.scratch_block_ids(False, 0, 2)

    two = _make_kv(local_branches=2)
    assert two.scratch_block_ids(True, 0, 2) != two.scratch_block_ids(False, 0, 2)


def test_scratch_exhaustion_still_raises():
    one = _make_kv(local_branches=1)
    cap = one.scratch_blocks_per_branch
    with pytest.raises(RuntimeError, match="scratch blocks exhausted"):
        one.scratch_block_ids(False, 0, cap + 1)


def test_invalid_local_branches_rejected():
    with pytest.raises(ValueError, match="local_branches"):
        _make_kv(local_branches=3)
