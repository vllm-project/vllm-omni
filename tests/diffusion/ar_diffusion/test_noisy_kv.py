# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for noisy KV last-use eviction (k=1 overlap)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.cell_schedule import (
    CellPlan,
    CellSchedule,
    Inflight,
    Ordering,
    build_cell_plan,
    union_wait_ready,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cache(**kwargs) -> NoisyKVCache:
    spec = ARDiffusionNoisyKVSpec(
        num_layers=2,
        num_kv_heads=2,
        head_size=4,
        block_size=4,
        max_chunk_tokens=kwargs.get("max_chunk_tokens", 4),
        max_history_chunks=6,
    )
    return NoisyKVCache(
        spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=kwargs.get("layer_groups", 2),
        max_batch_size=kwargs.get("max_batch_size", 1),
        step_replicas=kwargs.get("step_replicas", 1),
    )


def test_capacity_uses_k1():
    cache = _cache(layer_groups=2, max_batch_size=2)
    # R * (H + 2 + G) = 2 * (6 + 2 + 2) = 20
    assert cache.capacity == 20


def test_capacity_covers_latest_stage_frontier():
    cache = _cache(layer_groups=1, step_replicas=5)
    # A Latest-KV plan with H=6 and S=5 needs up to 10 live versions on one rank.
    assert cache.capacity == 11


def test_capacity_rejects_undersized_memory_budget():
    cache = _cache(layer_groups=2, max_batch_size=2)
    bytes_per_slot = cache.bytes_per_version * cache.spec.num_layers
    with pytest.raises(RuntimeError, match="budget fits 1 versions.*requires 20"):
        NoisyKVCache(
            cache.spec,
            dtype=torch.float32,
            device=torch.device("cpu"),
            layer_groups=2,
            max_batch_size=2,
            gpu_memory_fraction=1.0,
            available_bytes=bytes_per_slot + 8,
        )


@pytest.mark.parametrize("extra_bytes", [0, 8])
def test_capacity_accepts_complete_schedule_budget(extra_bytes):
    cache = _cache(layer_groups=1, step_replicas=5)
    bounded = NoisyKVCache(
        cache.spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=5,
        available_bytes=cache.reserved_bytes + extra_bytes,
    )
    assert bounded.capacity == cache.capacity == 11
    assert bounded.reserved_bytes == cache.reserved_bytes


def test_reset_all_releases_versions():
    cache = _cache()
    state = NoisyKVState(cache)
    key = ("A", 0, 0, 0)
    cache.pool.alloc(key)
    assert cache.pool.has(key)
    state.reset_all()
    assert not cache.pool.has(key)
    assert cache.pool.has_free
    assert all(len(stack) == cache.capacity for stack in cache.pool.free)


def test_per_layer_free_stacks_do_not_share_slots():
    """Each local block has its own capacity; filling layer 0 must not starve layer 1."""
    cache = _cache()
    cap = cache.capacity
    # Fill every per-layer slot on block/layer 0.
    for i in range(cap):
        cache.pool.alloc(("A", i, 0, 0))
    assert len(cache.pool.free[0]) == 0
    with pytest.raises(RuntimeError, match="layer=0"):
        cache.pool.alloc(("A", cap, 0, 0))
    # Layer 1 still has a full free stack.
    assert len(cache.pool.free[1]) == cap
    for i in range(cap):
        cache.pool.alloc(("A", i, 0, 1))
    assert len(cache.pool.keys) == 2 * cap
    assert len(cache.pool.free[1]) == 0


@pytest.mark.parametrize("chunk_tokens", [4, 8, 12])
@pytest.mark.parametrize("write_slot", [0, 1, 3])
def test_cached_write_slots_match_compute_slot_mapping(chunk_tokens, write_slot):
    cache = _cache(max_chunk_tokens=12)
    state = NoisyKVState(cache)
    device = torch.device("cpu")
    got = state._cached_write_slots(write_slot, chunk_tokens, device)
    write_blocks = cache.pool.block_ids(write_slot)[: chunk_tokens // cache.pool.block_size]
    expect = compute_slot_mapping(write_blocks, torch.arange(chunk_tokens, dtype=torch.long), cache.pool.block_size)
    torch.testing.assert_close(got, expect)
    # Second call hits cache (same tensor object).
    assert state._cached_write_slots(write_slot, chunk_tokens, device) is got


def test_prepare_evict_releases_last_use():
    plan = build_cell_plan(
        CellSchedule(
            chunks=4,
            num_denoise_steps=2,
            step_replicas=3,
            layer_groups=2,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
        )
    )
    cache = _cache()
    state = NoisyKVState(cache)
    state.bind_rank(0, None)
    state.begin_request("A", plan, chunk_tokens=4, t0=0)
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    state.set_inflight(inflight)
    # Slot 0 writes (0,0)
    state.prepare((("A", (0, 0, 0)),))
    assert state.resident_versions == 1
    state.evict(0)
    # last-use of (0,0) on r0 is later than slot 0
    assert state.resident_versions == 1
    state.prepare((("A", (1, 0, 0)),))
    state.prepare((("A", (2, 0, 0)),))
    # after slot 2, (0,0) should still be resident until evict(2)
    before = state.resident_versions
    state.evict(2)
    assert state.resident_versions < before or (0, 0, 0) not in [
        (k[1], k[2], k[3]) for k in cache.pool.keys if k[0] == "A"
    ]


def _vertical_plan() -> CellPlan:
    return build_cell_plan(
        CellSchedule(
            chunks=4,
            num_denoise_steps=2,
            step_replicas=3,
            layer_groups=2,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
        )
    )


def test_wait_ready_covers_first_remote_consume():
    """Await remotes at the tick before first consume (not only same-slot inbound)."""
    plan = _vertical_plan()
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    first_consume: dict[tuple[int, int, int], int] = {}
    for slot in range(plan.num_slots):
        task = plan.task(slot, 0)
        if task is None:
            continue
        for src in plan.sources(task, 0):
            if src.owner != 0:
                first_consume.setdefault(src.version, slot)
    # Every first remote consume at C has wait_ready at C-1 (or 0 if C==0).
    for version, consume_slot in first_consume.items():
        wait_slot = 0 if consume_slot == 0 else consume_slot - 1
        ready = {v for _req, v in union_wait_ready(inflight, wait_slot, 0)}
        assert version in ready, f"{version} first used at {consume_slot} missing wait at {wait_slot}"
    # wait_ready must be non-empty when there is any remote history (cell expand).
    assert first_consume
    assert any(union_wait_ready(inflight, slot, 0) for slot in range(plan.num_slots))


def test_wait_ready_is_empty_without_kv_history():
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=2,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=0,
        )
    )
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    for slot in range(plan.num_slots):
        assert union_wait_ready(inflight, slot, 0) == frozenset()


@pytest.mark.parametrize(
    ("stages", "ordering"),
    [(1, Ordering.SELF_FORCING), (3, Ordering.LATEST_KV)],
)
@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize("chunk_tokens", [4, 8, 12])
def test_attention_reads_only_valid_version_pages(stages, ordering, chunk, chunk_tokens):
    plan = build_cell_plan(
        CellSchedule(
            chunks=4,
            num_denoise_steps=2,
            step_replicas=stages,
            layer_groups=2,
            ordering=ordering,
            kv_history_chunks=6,
        )
    )
    cache = _cache(max_chunk_tokens=12)
    state = NoisyKVState(cache)
    state.bind_rank(0, None)
    state.begin_request("A", plan, chunk_tokens=chunk_tokens)
    task = (chunk, 0, 0)
    sources = plan.sources(task, 0)
    assert len(sources) == chunk
    source_slots = [cache.pool.alloc(("A", *src.version)) for src in sources]
    # Keep an unselected version of the same chunk resident; reads must still follow the plan.
    first_chunk, first_step, first_block = sources[0].version
    cache.pool.alloc(("A", first_chunk, (first_step + 1) % 3, first_block))
    contexts = state.prepare((("A", task),))[0]
    generator = torch.Generator().manual_seed(42)
    shape = (chunk_tokens, cache.spec.num_kv_heads, cache.spec.head_size)
    for context in contexts:  # one layer per cell
        context.key_pool.zero_()
        context.value_pool.zero_()
        unused = torch.ones(context.key_pool.shape[0], dtype=torch.bool)
        keys, values = [], []
        for slot in source_slots:
            start = slot * cache.spec.max_chunk_tokens
            key = torch.randn(shape, generator=generator)
            value = torch.randn(shape, generator=generator)
            context.key_pool[start : start + chunk_tokens] = key
            context.value_pool[start : start + chunk_tokens] = value
            unused[start : start + chunk_tokens] = False
            keys.append(key)
            values.append(value)
        query = torch.randn(shape, generator=generator)
        current_key = torch.randn(shape, generator=generator)
        current_value = torch.randn(shape, generator=generator)
        keys.append(current_key)
        values.append(current_value)
        unused[context.video_slots] = False
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1),
            torch.cat(keys).transpose(0, 1),
            torch.cat(values).transpose(0, 1),
        ).transpose(0, 1)
        inputs = context.to_layer_inputs()
        actual = paged_write_attn(inputs, query, current_key, current_value, None, None, cache.spec.head_size**-0.5)
        torch.testing.assert_close(actual, expected)

        # Changing a reserved page or an unselected version must not change the attention output.
        context.key_pool[unused] = 50
        context.value_pool[unused] = -100
        after = paged_write_attn(inputs, query, current_key, current_value, None, None, cache.spec.head_size**-0.5)
        torch.testing.assert_close(after, actual, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_tokens", [-4, 0, 3, 5, 16])
def test_begin_request_rejects_invalid_chunk_length(chunk_tokens):
    state = NoisyKVState(_cache(max_chunk_tokens=12))
    with pytest.raises(ValueError, match="chunk_tokens"):
        state.begin_request("A", _vertical_plan(), chunk_tokens=chunk_tokens)
    assert state.resident_versions == 0


@pytest.mark.parametrize("history", [0, 1, 3])
def test_same_step_sink_cache_lifetime_and_attention_window(history):
    # Simulate all ranks and explicit P2P payload copies. This checks actual
    # pool allocation, table width, last-use eviction and final cleanup.
    steps, blocks, chunks = 2, 2, 12
    plan = build_cell_plan(
        CellSchedule(
            chunks=chunks,
            num_denoise_steps=steps,
            step_replicas=steps,
            layer_groups=1,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=history,
            blocks=blocks,
        )
    )
    states = []
    for rank in range(plan.world):
        spec = ARDiffusionNoisyKVSpec(
            num_layers=blocks,
            num_kv_heads=2,
            head_size=4,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=max(1, history),
        )
        cache = NoisyKVCache(
            spec, dtype=torch.float32, device=torch.device("cpu"), layer_groups=1, max_batch_size=1, step_replicas=steps
        )
        state = NoisyKVState(cache)
        state.bind_rank(rank, None)
        state.begin_request("A", plan, chunk_tokens=4)
        states.append(state)
    for tick in range(plan.num_slots):
        for rank, state in enumerate(states):
            task = plan.task(tick, rank)
            if task is None:
                continue
            context = state.prepare((("A", task),))[0][0]
            pool = state.cache.pool
            for src in plan.sources(task, rank):
                key = ("A", *src.version)
                k, v = pool.kv_block_views(key, 1)
                expected = 1000 + 100 * src.version[0] + 10 * src.version[1] + src.version[2]
                torch.testing.assert_close(k, torch.full_like(k, expected))
                torch.testing.assert_close(v, torch.full_like(v, -expected))
            assert context.seq_len == (len(plan.sources(task, rank)) + 1) * 4
            key = ("A", *task)
            k, v = pool.kv_block_views(key, 1)
            value = 1000 + 100 * task[0] + 10 * task[1] + task[2]
            k.fill_(value)
            v.fill_(-value)
        for xfer in plan.transfers(tick):
            key = ("A", *xfer.version)
            source = states[xfer.src].cache.pool
            target = states[xfer.dst].cache.pool
            target.alloc(key)
            target.copy_into(key, source.tensor_dict(key, 1))
        for state in states:
            state.evict(tick)
    for state in states:
        state.end_request("A")
        assert state.resident_versions == 0
