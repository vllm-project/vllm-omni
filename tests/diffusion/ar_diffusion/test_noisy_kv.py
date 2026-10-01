# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for noisy KV last-use eviction (k=1 overlap)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    Inflight,
    Ordering,
    ChunkPlan,
    ChunkSchedule,
    build_chunk_plan,
    incoming_transfers,
    union_wait_ready,
)

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
    )


def test_capacity_uses_k1():
    cache = _cache(layer_groups=2, max_batch_size=2)
    # R * (H + 2 + G) = 2 * (6 + 2 + 2) = 20
    assert cache.capacity == 20


def test_capacity_respects_memory_budget():
    cache = _cache(layer_groups=2, max_batch_size=2)
    # Force a tiny budget: only one version slot fits.
    tiny = NoisyKVCache(
        cache.spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=2,
        max_batch_size=2,
        gpu_memory_fraction=1.0,
        available_bytes=cache.bytes_per_version + 8,
    )
    assert tiny.capacity == 1
    assert tiny.reserved_bytes == tiny.bytes_per_version


def test_reset_all_releases_versions():
    cache = _cache()
    state = NoisyKVState(cache)
    key = ("A", 0, 0)
    cache.pool.alloc(key)
    assert cache.pool.has(key)
    state.reset_all()
    assert not cache.pool.has(key)
    assert cache.pool.free


def test_prepare_evict_releases_last_use():
    plan = build_chunk_plan(
        ChunkSchedule(
            chunks=4,
            num_denoise_steps=2,
            stages=3,
            layer_groups=2,
            ordering=Ordering.INTERLEAVED,
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
    state.prepare((("A", (0, 0)),))
    assert state.resident_versions == 1
    state.evict(0)
    # last-use of (0,0) on r0 is later than slot 0
    assert state.resident_versions == 1
    state.prepare((("A", (1, 0)),))
    state.prepare((("A", (2, 0)),))
    # after slot 2, (0,0) should still be resident until evict(2)
    before = state.resident_versions
    state.evict(2)
    assert state.resident_versions < before or (0, 0) not in [(k[1], k[2]) for k in cache.pool.keys if k[0] == "A"]


def _vertical_plan() -> ChunkPlan:
    return build_chunk_plan(
        ChunkSchedule(
            chunks=4,
            num_denoise_steps=2,
            stages=3,
            layer_groups=2,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=6,
        )
    )


def test_wait_ready_matches_incoming_transfers():
    """I9②: the awaited set is exactly slot T's inbound, read at slot T+1."""
    plan = _vertical_plan()
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    for slot in range(plan.num_slots):
        incoming = sorted(incoming_transfers(plan._transfers, slot, 0), key=lambda x: x.version)
        ready = sorted(union_wait_ready(inflight, slot, 0), key=lambda item: item[1])
        assert [x.version for x in incoming] == [version for _req, version in ready]


def test_wait_ready_is_empty_without_kv_history():
    plan = build_chunk_plan(
        ChunkSchedule(
            chunks=2,
            num_denoise_steps=1,
            stages=1,
            layer_groups=2,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=0,
        )
    )
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    for slot in range(plan.num_slots):
        assert union_wait_ready(inflight, slot, 0) == frozenset()


@pytest.mark.parametrize("stages", [1, 3])
@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize("chunk_tokens", [4, 8, 12])
def test_attention_reads_only_valid_version_pages(stages, chunk, chunk_tokens):
    plan = build_chunk_plan(
        ChunkSchedule(
            chunks=4,
            num_denoise_steps=2,
            stages=stages,
            layer_groups=2,
            ordering=Ordering.SERIAL,
            kv_history_chunks=6,
        )
    )
    cache = _cache(max_chunk_tokens=12)
    state = NoisyKVState(cache)
    state.bind_rank(0, None)
    state.begin_request("A", plan, chunk_tokens=chunk_tokens)
    task = (chunk, 0)
    sources = plan.sources(task, 0)
    assert len(sources) == chunk
    source_slots = [cache.pool.alloc(("A", *src.version)) for src in sources]
    # 同一 chunk 的未选版本也驻留，读取仍只能遵守计划。
    first_chunk, first_step = sources[0].version
    cache.pool.alloc(("A", first_chunk, (first_step + 1) % 3))
    contexts = state.prepare((("A", task),))[0]
    generator = torch.Generator().manual_seed(42)
    shape = (chunk_tokens, cache.spec.num_kv_heads, cache.spec.head_size)
    for context in contexts:
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

        # 修改预留页和未选版本后，attention 输出不应变化。
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
