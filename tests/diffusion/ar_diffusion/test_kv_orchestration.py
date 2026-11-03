# SPDX-License-Identifier: Apache-2.0
"""Tests for ARDiffusionKVCache — the engine-level KV orchestration body (Phase 1)."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
DIMS = dict(
    num_layers=2,
    num_kv_heads=4,
    head_size=64,
    dtype=torch.float16,
    block_size=16,
    kv_branches=(ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 1)),
    session_capacity=2,
)


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
        # Allocation may transiently add one in-flight chunk.
        assert len(kv.window_block_ids(adapter)) <= kv.spec.window_chunks + 1
        kv.commit_chunk(adapter)
        # A successful commit immediately prunes physical ownership.
        assert len(kv.window_block_ids(adapter)) <= kv.spec.window_chunks
        free_after.append(kv.manager.block_pool.get_num_free_blocks())

    # Pool memory plateaus once the window is full (eviction recycles blocks).
    assert free_after[-1] == free_after[kv.spec.window_chunks - 1]

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
    """ARDiffusionKVState.close() returns both CFG kv_branches' pool blocks to the pool.

    This is the primitive the runner's LRU eviction relies on: when a session is
    evicted, close() must free the blocks both adapters hold, or session churn
    leaks pool ownership (review P1).
    """
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

    kv = make_cache(chunk_size=16, window_chunks=2)
    free_total = kv.manager.block_pool.get_num_free_blocks()

    pos = kv.begin_request("bde__s")
    neg = kv.begin_request("bde__s__neg")
    state = ARDiffusionKVState(
        kv,
        "s",
        {"positive": pos, "negative": neg},
        num_layers=kv.num_layers,
    )
    for _ in range(3):
        for adapter in (pos, neg):
            kv.allocate_chunk(adapter)
            kv.commit_chunk(adapter)
    # Both kv_branches hold resident blocks now.
    assert kv.manager.block_pool.get_num_free_blocks() < free_total

    state.close()
    # All blocks returned — no leak across an evicted session.
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


@pytest.mark.parametrize(
    "frames_per_call,block_size,max_model_len,num_calls",
    [(1, 16, 113, 1000), (3, 16, 113, 1000), (3, 1560, 1 << 20, 241)],
    ids=["single-frame", "three-frame", "lingbot-two-minutes"],
)
@pytest.mark.parametrize("sink_chunks,reset_at_boundary", [(0, False), (2, False), (0, True), (2, True)])
def test_rollout_keeps_metadata_bounded(
    frames_per_call, block_size, max_model_len, num_calls, sink_chunks, reset_at_boundary
):
    """Long rollouts retain exact global-position K/V with bounded storage indices."""
    kv = ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=block_size,
            window_chunks=3,
            sink_chunks=sink_chunks,
            reset_at_boundary=reset_at_boundary,
        ),
        num_layers=1,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        block_size=block_size,
        max_model_len=max_model_len,
        available_bytes=1 << 24,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=frames_per_call,
        device=torch.device("cpu"),
    )
    adapter = kv.begin_request("long-rollout")
    state = ARDiffusionKVState(kv, "long-rollout", {"main": adapter}, num_layers=1)
    free_total = kv.manager.block_pool.get_num_free_blocks()
    pool_shape = kv.key_cache(0).shape
    retained_frames = sink_chunks + (0 if reset_at_boundary else 3)
    for call_index in range(num_calls):
        start = call_index * frames_per_call * block_size
        stop = start + frames_per_call * block_size
        if frames_per_call == 1:
            kv.allocate_chunk(adapter)
            slots = kv.chunk_write_slots(adapter)
        else:
            context = state.get_kv_caches("main", seq_len=stop - start, commit_current=True)[0].forward_ctx
            context.ensure_video_slots(torch.device("cpu"))
            slots = context.current_video_slot_mapping
        assert len(kv.block_table(adapter)) <= retained_frames + frames_per_call
        values = torch.arange(start, stop, dtype=torch.float32).view(-1, 1, 1).expand(-1, 1, 4)
        kv.key_cache(0).flatten(0, 1)[slots] = values
        kv.value_cache(0).flatten(0, 1)[slots] = -values
        if frames_per_call == 1:
            kv.commit_chunk(adapter)
        else:
            state.commit_paged_context("main")
            assert state._committed["main"] == stop
        sink_end = min(stop, sink_chunks * block_size)
        tail_start = stop if reset_at_boundary else max(sink_end, stop - 3 * block_size)
        expected = torch.cat((torch.arange(sink_end), torch.arange(tail_start, stop))).float()
        expected = expected.view(-1, 1, 1).expand(-1, 1, 4)
        resident = kv.window_block_ids(adapter)
        torch.testing.assert_close(kv.key_cache(0)[resident].flatten(0, 1), expected)
        torch.testing.assert_close(kv.value_cache(0)[resident].flatten(0, 1), -expected)
        assert adapter.completed_chunks * block_size == stop
        assert adapter.absolute_num_computed_tokens == stop
        assert len(kv.block_table(adapter)) <= retained_frames
        assert adapter.num_computed_tokens <= retained_frames * block_size
        assert kv.key_cache(0).shape == pool_shape
        assert free_total - kv.manager.block_pool.get_num_free_blocks() == min(stop // block_size, retained_frames)
    state.reset()
    fresh = state.adapter("main")
    assert fresh.compacted_tokens == fresh.absolute_num_computed_tokens == fresh.num_computed_tokens == 0
    assert kv.block_table(fresh) == []
    assert kv.manager.block_pool.get_num_free_blocks() == free_total
    kv.allocate_chunk(fresh)
    assert len(kv.chunk_write_slots(fresh)) == block_size
    state.close()
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_compaction_offsets_are_request_local():
    kv = make_cache()
    free_total = kv.manager.block_pool.get_num_free_blocks()
    first, second = kv.begin_request("first"), kv.begin_request("second")
    for index in range(8):
        kv.allocate_chunk(first)
        kv.commit_chunk(first)
        if index in (0, 4):
            kv.allocate_chunk(second)
            kv.commit_chunk(second)
    assert first.absolute_num_computed_tokens == 8 * 16
    assert first.compacted_tokens == 6 * 16
    assert second.absolute_num_computed_tokens == 2 * 16
    assert second.compacted_tokens == 0
    assert first.num_computed_tokens == second.num_computed_tokens == 2 * 16
    assert set(kv.block_table(first)).isdisjoint(kv.block_table(second))
    kv.end_request(first)
    kv.end_request(second)
    group = kv.manager.coordinator.single_type_managers[0]
    assert not group.req_to_blocks and not group.num_cached_block
    assert kv.manager.block_pool.get_num_free_blocks() == free_total
