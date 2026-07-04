# SPDX-License-Identifier: Apache-2.0
"""Slot-mapping / block-table tests for the AR-Diffusion engine (Phase 1, Step 3)."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionRequestAdapter,
    ChunkWindowSpec,
    build_kv_manager,
    chunk_slot_mapping,
    compute_slot_mapping,
    resident_block_ids,
)

BLOCK = 16


def make_spec(*, window_chunks=2):
    return ChunkWindowSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        sliding_window=window_chunks * BLOCK,
        chunk_size=BLOCK,
        window_chunks=window_chunks,
    )


def test_slot_mapping_matches_block_offsets():
    # block table [5, 2, 9], block_size 4. slot = block*4 + (pos % 4).
    slots = compute_slot_mapping([5, 2, 9], [0, 1, 4, 5, 8, 9], block_size=4)
    assert slots.tolist() == [20, 21, 8, 9, 36, 37]


def test_slot_mapping_rejects_bad_block_size():
    with pytest.raises(ValueError):
        compute_slot_mapping([1], [0], block_size=0)


def test_chunk_slot_mapping_targets_current_chunk():
    # After 2 committed chunks (num_computed=32), the in-flight chunk maps to the
    # block covering positions 32..47 — i.e. block table index 2.
    block_ids = [7, 4, 3]  # block index 2 -> physical block 3
    slots = chunk_slot_mapping(block_ids, num_computed_tokens=32, chunk_size=4, block_size=16)
    # positions 32..35 -> block_index 2 -> physical 3 -> slots 48..51
    assert slots.tolist() == [48, 49, 50, 51]


def test_resident_block_ids_excludes_null():
    assert resident_block_ids([0, 1, 0, 2, 3], null_block_id=0) == [1, 2, 3]


def test_blocktable_build_single_request():
    """Integration: a real KVCacheManager block table -> a valid slot mapping.

    Every slot for the in-flight chunk must land inside a real (non-null) block
    that the manager actually allocated for the request.
    """
    spec = make_spec()
    mgr = build_kv_manager(spec, ["l0"], num_blocks=16, max_model_len=1024)
    null_id = mgr.block_pool.null_block.block_id
    adapter = ARDiffusionRequestAdapter("req", chunk_size=BLOCK)

    mgr.allocate_slots(adapter, num_new_tokens=BLOCK)
    block_ids = mgr.get_block_ids(adapter.request_id)[0]
    real = set(resident_block_ids(block_ids, null_id))
    assert real, "expected at least one real block after allocation"

    slots = chunk_slot_mapping(block_ids, adapter.num_computed_tokens, spec.chunk_size, spec.block_size)
    # Each slot resolves to a real allocated block, none to null_block.
    blocks_used = {int(s) // spec.block_size for s in slots}
    assert blocks_used <= real
    assert null_id not in blocks_used
    assert len(slots) == spec.chunk_size
