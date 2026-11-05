# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-token storage and transactions through the real vLLM KV allocator."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionSegmentedKVSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import SegmentHandle
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

SPEC = ARDiffusionSegmentedKVSpec(
    block_size=16,
    max_retained_tokens=64,
    max_retained_segments=5,
    max_pending_tokens=24,
    max_pending_segments=2,
    initial_staging_tokens=96,
)


def make_cache(**kwargs):
    options = dict(
        num_layers=2,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.float32,
        block_size=16,
        max_model_len=128,
        available_bytes=1 << 20,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=2,
        device=torch.device("cpu"),
        segmented=SPEC,
    )
    options.update(kwargs)
    return ARDiffusionKVCache(ARDiffusionKVConfig(enable=True, chunk_size=16, window_chunks=2), **options)


def make_state(cache, name="session"):
    return ARDiffusionKVState(cache, name, {"main": cache.begin_request(name)}, num_layers=cache.num_layers)


def rows(count, offset=0, *, device="cpu", dtype=torch.float32):
    return (torch.arange(count * 8, dtype=torch.float32, device=device).reshape(count, 2, 4) / 19 + offset).to(dtype)


def append(state, lengths, offset=0, retain=None):
    row = partial(rows, device=state.cache.device, dtype=state.cache.dtype)
    count = sum(lengths)
    state.prepare_layer_views(count, 3)
    handles = state.begin_commit(lengths)
    for layer in range(state.cache.num_layers):
        media = row(count, offset + 100 * layer)
        condition = row(3, -20)
        state.assemble(layer, condition, -condition, media, -media)
        state.stage(layer)
    state.commit(retain=handles if retain is None else retain(handles))
    return handles


@pytest.mark.parametrize("lengths", [(1, 1), (15, 3), (17, 5)])
def test_ragged_pages_match_dense_attention_without_padding(lengths):
    cache = make_cache()
    free = cache.manager.block_pool.get_num_free_blocks()
    owner = make_state(cache)
    state = owner.segmented("main")
    # Uninitialized tails must never appear in exact-token history views.
    for pool in cache._k_pools + cache._v_pools:
        pool.fill_(float("nan"))
    handles = append(state, lengths, offset=2)
    assert free - cache.manager.block_pool.get_num_free_blocks() == sum((n + 15) // 16 for n in lengths)
    for handle in handles:
        adapter = state._allocations[handle].adapter
        assert adapter.num_tokens == handle.num_tokens
        assert adapter.num_computed_tokens == 0
    state.prepare_layer_views(4, 2)
    for layer in range(cache.num_layers):
        media, condition = rows(4, 5), rows(2, -3)
        key, value = state.assemble(layer, condition, -condition, media, -media)
        expected = torch.cat((rows(sum(lengths), 2 + layer * 100), media, condition))
        torch.testing.assert_close(key, expected)
        torch.testing.assert_close(value, -expected)
        query = rows(3, 0.1).transpose(0, 1)
        actual = torch.nn.functional.scaled_dot_product_attention(query, key.transpose(0, 1), value.transpose(0, 1))
        reference = torch.nn.functional.scaled_dot_product_attention(
            query, expected.transpose(0, 1), -expected.transpose(0, 1)
        )
        torch.testing.assert_close(actual, reference)
    owner.close()
    assert cache.manager.block_pool.get_num_free_blocks() == free


def test_noisy_steps_reuse_history_and_never_mutate_committed_pages():
    cache = make_cache()
    state = make_state(cache).segmented("main")
    append(state, (3, 5))
    pool_before = [pool.clone() for pool in cache._k_pools + cache._v_pools]
    pointers = [(key.data_ptr(), value.data_ptr()) for key, value in state._staging().layers]
    for step in range(4):
        state.prepare_layer_views(4, 2)
        for layer in range(cache.num_layers):
            state.assemble(layer, rows(2), rows(2), rows(4, step), rows(4, step))
    assert state.history_gathers == 1
    assert pointers == [(key.data_ptr(), value.data_ptr()) for key, value in state._staging().layers]
    for before, after in zip(pool_before, cache._k_pools + cache._v_pools):
        torch.testing.assert_close(before, after, equal_nan=True)


def test_failed_clean_layer_rolls_back_without_publishing_or_leaking():
    cache = make_cache()
    owner = make_state(cache)
    state = owner.segmented("main")
    old = append(state, (3, 5), offset=7)
    free = cache.manager.block_pool.get_num_free_blocks()
    state.prepare_layer_views(6, 2)
    pending = state.begin_commit((2, 4))
    state.assemble(0, rows(2), rows(2), rows(6, 99), rows(6, 99))
    state.stage(0)
    with pytest.raises(RuntimeError, match="missing transformer layers"):
        state.commit(retain=pending)
    assert state.handles == old
    state.rollback()
    state.rollback()
    assert cache.manager.block_pool.get_num_free_blocks() == free
    state.prepare_layer_views(2, 0)
    key, _ = state.assemble(0, rows(0), rows(0), rows(2), rows(2))
    torch.testing.assert_close(key[:8], rows(8, 7))
    assert state.handles == old


def test_mid_allocation_failure_frees_only_pending_segments(monkeypatch):
    cache = make_cache()
    state = make_state(cache).segmented("main")
    old = append(state, (3, 5))
    free = cache.manager.block_pool.get_num_free_blocks()
    allocate = cache.allocate_token_slots
    calls = 0

    def fail_second(adapter, count):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocator failure")
        return allocate(adapter, count)

    monkeypatch.setattr(cache, "allocate_token_slots", fail_second)
    state.prepare_layer_views(7, 0)
    with pytest.raises(RuntimeError, match="injected allocator failure"):
        state.begin_commit((3, 4))
    assert state.handles == old
    assert not state.commit_active
    assert cache.manager.block_pool.get_num_free_blocks() == free
    assert len(cache._adapters) == 3  # Canonical branch and its two old segments.


def test_retention_handles_are_ordered_and_session_local():
    cache = make_cache()
    owner = make_state(cache)
    state = owner.segmented("main")
    handles = append(state, (3, 5), offset=4)
    state.retain(handles[::-1])
    state.prepare_layer_views(1, 0)
    key, _ = state.assemble(0, rows(0), rows(0), rows(1), rows(1))
    reference = rows(8, 4)
    torch.testing.assert_close(key[:8], torch.cat((reference[3:], reference[:3])))
    other = make_state(cache, "other").segmented("main")
    foreign = append(other, (2, 2))
    for invalid in ((handles[0], handles[0]), (foreign[0],), (SegmentHandle(3),)):
        with pytest.raises(ValueError, match="unique segment handles"):
            state.retain(invalid)
    assert state.handles == handles[::-1]


def test_shared_staging_invalidates_when_switching_sessions():
    cache = make_cache()
    a = make_state(cache, "a").segmented("main")
    b = make_state(cache, "b").segmented("main")
    append(a, (3, 5), offset=10)
    append(b, (4, 2), offset=20)
    for state, count, offset in ((a, 8, 10), (b, 6, 20), (a, 8, 10)):
        state.prepare_layer_views(2, 0)
        key, _ = state.assemble(0, rows(0), rows(0), rows(2), rows(2))
        torch.testing.assert_close(key[:count], rows(count, offset))
    assert a.history_gathers == 2
    assert b.history_gathers == 1


def test_thirteen_windows_exact_sink_recent_and_periodic_modality_reset():
    cache = make_cache()
    owner = make_state(cache)
    state = owner.segmented("main")
    free = cache.manager.block_pool.get_num_free_blocks()
    phases: list[tuple[SegmentHandle, ...]] = []
    expected = {}
    for window in range(13):
        if window == 12:
            state.retain(
                [handle for index, pair in enumerate(phases) for handle in pair[1:] if handle in state.handles]
            )
            assert state.history_tokens > 0
        for phase in range(4):
            index = window * 4 + phase
            lengths = (3 + index % 4, 5 + index % 5)
            value = rows(sum(lengths), index * 10)

            def retained(new):
                phases.append(new)
                recents = phases[-2:]
                if len(phases) <= 2:
                    return [handle for pair in recents for handle in pair]
                available = set(state.handles) | set(new)
                return [phases[0][1], *[handle for pair in recents for handle in pair if handle in available]]

            handles = append(state, lengths, index * 10, retained)
            expected[handles[0]], expected[handles[1]] = value[: lengths[0]], value[lengths[0] :]
            state.prepare_layer_views(1, 0)
            key, _ = state.assemble(0, rows(0), rows(0), rows(1), rows(1))
            torch.testing.assert_close(key[: state.history_tokens], torch.cat([expected[h] for h in state.handles]))
            assert len(state.handles) <= 5
            assert len(cache._adapters) <= 6
            assert free - cache.manager.block_pool.get_num_free_blocks() <= SPEC.retained_blocks(cache.block_size)
    owner.close()
    assert cache.manager.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize("operation", ["close", "reset"])
def test_session_lifecycle_releases_pending_and_committed(operation):
    cache = make_cache()
    owner = make_state(cache)
    free = cache.manager.block_pool.get_num_free_blocks()
    state = owner.segmented("main")
    append(state, (3, 5))
    state.prepare_layer_views(7, 0)
    state.begin_commit((2, 5))
    getattr(owner, operation)()
    assert cache.manager.block_pool.get_num_free_blocks() == free
    with pytest.raises(RuntimeError, match="closed"):
        state.prepare_layer_views(1, 0)
    if operation == "reset":
        fresh = owner.segmented("main")
        assert fresh is not state and fresh.history_tokens == 0
    owner.close()
    owner.close()


def test_long_conditioning_grows_outside_forward_and_preserves_history():
    votes = []

    def agree(value):
        votes.append(value)
        return value

    cache = make_cache(agree_staging_allocation=agree)
    votes.clear()
    state = make_state(cache).segmented("main")
    append(state, (3, 5), offset=8)
    old = state._staging()
    budget = cache.memory_budget_bytes
    state.prepare_layer_views(4, 300)
    grown = state._staging()
    assert grown is not old and grown.capacity >= 312
    assert votes == [True, True]
    assert cache.memory_budget_bytes > budget
    key, _ = state.assemble(0, rows(300), rows(300), rows(4), rows(4))
    torch.testing.assert_close(key[:8], rows(8, 8))
    state.prepare_layer_views(4, 300)
    assert state._staging() is grown and state.history_gathers == 1


def test_failed_growth_keeps_previous_buffers_and_history(monkeypatch):
    cache = make_cache()
    state = make_state(cache).segmented("main")
    append(state, (3, 5), offset=8)
    old = state._staging()
    monkeypatch.setattr(cache, "_agree_staging_allocation", lambda _: False)
    with pytest.raises(RuntimeError, match="memory budget"):
        state.prepare_layer_views(4, 300)
    assert state._staging() is old and state.history_tokens == 8
    state.prepare_layer_views(4, 1)
    key, _ = state.assemble(0, rows(1), rows(1), rows(4), rows(4))
    torch.testing.assert_close(key[:8], rows(8, 8))


def test_segment_bounds_and_staging_are_in_admission_budget():
    cache = make_cache()
    assert cache.scratch_num_blocks == 0
    required_pages = (
        cache.num_local_kv_branches * (cache.session_capacity * SPEC.retained_blocks(16) + SPEC.pending_blocks(16)) + 2
    )
    assert cache.managed_num_blocks >= required_pages
    staging_bytes = SPEC.initial_staging_tokens * 2 * cache.num_layers * cache.num_kv_heads * cache.head_size * 4
    assert cache.history_staging_reserved_bytes == staging_bytes
    with pytest.raises(ValueError, match="cannot fit one session"):
        make_cache(available_bytes=staging_bytes)
    with pytest.raises(ValueError, match="staging must fit"):
        replace(SPEC, initial_staging_tokens=8)
    with pytest.raises(ValueError, match="must be positive"):
        replace(SPEC, max_pending_segments=0)


def test_segments_larger_than_the_frame_window_keep_every_row():
    spec = replace(SPEC, max_retained_tokens=15000, max_pending_tokens=6000, initial_staging_tokens=22000)
    cache = make_cache(segmented=spec, available_bytes=1 << 24)
    # The legacy window is only 32 tokens; one ragged segment holds 4860.
    assert cache.spec.sliding_window == 32
    owner = make_state(cache)
    state = owner.segmented("main")
    free = cache.manager.block_pool.get_num_free_blocks()
    first = append(state, (130, 4860), offset=3)
    second = append(state, (117, 4050), offset=12, retain=lambda new: first + new)
    state.prepare_layer_views(2, 0)
    key, value = state.assemble(0, rows(0), rows(0), rows(2), -rows(2))
    expected = torch.cat((rows(4990, 3), rows(4167, 12)))
    torch.testing.assert_close(key[:9157], expected)
    torch.testing.assert_close(value[:9157], -expected)
    assert len(state._allocations[first[1]].slots) == 4860
    state.retain((first[1], *second))
    state.prepare_layer_views(2, 0)
    key, _ = state.assemble(0, rows(0), rows(0), rows(2), rows(2))
    torch.testing.assert_close(key[:9027], torch.cat((rows(4990, 3)[130:], rows(4167, 12))))
    owner.close()
    assert cache.manager.block_pool.get_num_free_blocks() == free


def test_initial_peer_allocation_failure_releases_successful_rank_storage(monkeypatch):
    allocated = []
    real_allocate = ARDiffusionKVCache._allocate_storage

    def record(cache):
        allocated.append(cache)
        real_allocate(cache)

    monkeypatch.setattr(ARDiffusionKVCache, "_allocate_storage", record)
    with pytest.raises(RuntimeError, match="pool allocation failed"):
        make_cache(agree_staging_allocation=lambda _: False)
    cache = allocated[0]
    assert not cache._k_pools and not cache._v_pools and not cache._kv_pools
    assert not cache.segmented_staging


def test_prepare_and_growth_reject_cuda_graph_capture_before_work(monkeypatch):
    cache = make_cache()
    state = make_state(cache).segmented("main")
    old = state._staging()
    monkeypatch.setattr(cache, "device", torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        state.prepare_layer_views(4, 2)
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        cache.ensure_segmented_staging("main", 1000)
    assert state._staging() is old and state.history_gathers == 0
