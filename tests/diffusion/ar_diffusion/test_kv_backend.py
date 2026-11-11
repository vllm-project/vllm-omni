# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CPU pools exercising the common execution and history lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import pytest
import torch

from tests.diffusion.ar_diffusion.test_noisy_kv import _cache, _history_plan
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    CellSchedule,
    Ordering,
    build_cell_plan,
    expected_prior_cells,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.lifecycle import KVBackend, KVCacheLifecycle
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVState, NoisyLayerContext, prior_history_in_pool
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState, PagedKVTask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
TaskT = TypeVar("TaskT")
ViewT = TypeVar("ViewT", bound=list[ARDiffusionPagedLayerContext] | list[NoisyLayerContext])


def _forward(backend: KVBackend[TaskT, ViewT], tasks: Sequence[TaskT]) -> Sequence[ViewT]:
    """Same executor boundary for both real backends, without converting tasks."""
    views = backend.prepare(tasks)
    assert len(views) == len(tasks)
    for layers in views:
        for layer in layers:
            if isinstance(layer, ARDiffusionPagedLayerContext):
                layer.forward_ctx.prepare(torch.device("cpu"), action_len=0, query_len=4)
            inputs = layer.to_layer_inputs()
            assert isinstance(inputs, ARDiffusionPagedLayerInputs)
            if isinstance(layer, NoisyLayerContext) and inputs.seq_len > 4:
                blocks = inputs.block_table[0, : (inputs.seq_len - 4) // inputs.block_size].long()
                history_slots = (blocks[:, None] * inputs.block_size + torch.arange(inputs.block_size)).flatten()
                actual = inputs.key_pool[history_slots]
                torch.testing.assert_close(actual, torch.full_like(actual, 3))
            inputs.key_pool[inputs.video_slots] = 3
            inputs.value_pool[inputs.video_slots] = 7
    backend.publish(tasks)
    return views


def _paged():
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=4, window_chunks=2),
        num_layers=1,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        block_size=4,
        max_model_len=64,
        available_bytes=8192,
        kv_branches=(ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 1)),
        session_capacity=2,
        device=torch.device("cpu"),
    )
    adapters = {branch.name: cache.begin_request(branch.name) for branch in cache.kv_branches}
    return cache, ARDiffusionKVState(cache, "session", adapters, num_layers=1)


@pytest.mark.parametrize("kind", ["paged", "cell"])
def test_real_backends_prepare_publish_finish_retained_release(kind):
    if kind == "paged":
        cache, state = _paged()
        task = PagedKVTask("positive", 4, True)
        owner = state.adapter("positive")
        lifecycle: KVCacheLifecycle = cache
        layers: list[ARDiffusionPagedLayerContext] = _forward(state, (task,))[0]
        slots = layers[0].forward_ctx.current_video_slot_mapping
        keys = cache._k_pools[0]
        history = cache.window_block_ids(owner)
        assert history
        assert cache.window_block_ids(state.adapter("negative")) == []
    else:
        cache = _cache()
        state = NoisyKVState(cache)
        plan = _history_plan()
        owner = "session"
        state.begin_request(owner, plan, chunk_tokens=4)
        cell = next(plan.task(slot, 0) for slot in range(plan.num_slots) if plan.task(slot, 0) is not None)
        task = (owner, cell)
        noisy_layers: list[NoisyLayerContext] = _forward(state, (task,))[0]
        slots = noisy_layers[0].video_slots
        keys = noisy_layers[0].key_pool
        state.retain_history(owner, [cell])
        lifecycle = state
        history = (owner, *cell)
    lifecycle.finish_request(owner)
    torch.testing.assert_close(keys[slots], torch.full_like(keys[slots], 3))
    if kind == "paged":
        assert cache.window_block_ids(owner) == history
        lifecycle.release_history(owner)
        with pytest.raises(RuntimeError, match="matching active adapter"):
            state.prepare((task,))
        cache.release_history(state.adapter("negative"))
    else:
        assert cache.pool.has(history)
        lifecycle.release_history(owner)
        assert not cache.pool.has(history)
        with pytest.raises(ValueError, match="no active request"):
            state.prepare((task,))


def test_paged_backend_stale_boundary_and_closed_rules():
    cache, state = _paged()
    task = PagedKVTask("positive", 4, True)
    with pytest.raises(RuntimeError, match="prepared task"):
        state.publish((task,))
    state.prepare((task,))
    with pytest.raises(RuntimeError, match="prepared task"):
        state.publish((PagedKVTask("positive", 4, True),))
    state.prepare((task,))
    with pytest.raises(ValueError, match="positive"):
        state.prepare((PagedKVTask("positive", 0, False),))
    with pytest.raises(AssertionError, match="frame-aligned"):
        state.prepare((PagedKVTask("positive", 3, False),))
    with pytest.raises(ValueError, match="same branch"):
        state.prepare((task, task))
    state.close()
    state.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.prepare((task,))
    assert not cache._adapters


def test_model_facing_wrappers_call_real_backend(monkeypatch):
    _, state = _paged()
    prepared, published = [], []
    prepare, publish = state.prepare, state.publish

    def record_prepare(tasks):
        prepared.extend(tasks)
        return prepare(tasks)

    def record_publish(tasks):
        published.extend(tasks)
        return publish(tasks)

    monkeypatch.setattr(state, "prepare", record_prepare)
    monkeypatch.setattr(state, "publish", record_publish)
    state.get_kv_caches("positive", seq_len=4, commit_current=True)
    state.commit_paged_context("positive")
    assert prepared == published
    assert isinstance(prepared[0], PagedKVTask)
    state.close()


def test_noisy_backend_publication_and_rank_ownership():
    cache = _cache()
    state = NoisyKVState(cache)
    plan = _history_plan()
    state.begin_request("owner", plan, chunk_tokens=4)
    cell = next(plan.task(slot, 0) for slot in range(plan.num_slots) if plan.task(slot, 0) is not None)
    task = ("owner", cell)
    with pytest.raises(RuntimeError, match="not prepared"):
        state.publish((task,))
    with pytest.raises(ValueError, match="unique"):
        state.prepare((task, task))
    remote = next(plan.task(slot, 1) for slot in range(plan.num_slots) if plan.task(slot, 1) is not None)
    with pytest.raises(ValueError, match="not scheduled locally"):
        state.prepare((("owner", remote),))
    _forward(state, (task,))
    with pytest.raises(RuntimeError, match="not prepared"):
        state.publish((task,))
    state.finish_request("owner")
    assert not cache.pool.keys


def test_shared_cache_new_state_continuation_reads_and_replaces_history():
    cache = _cache()
    first = NoisyKVState(cache)
    initial = CellSchedule(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
    )
    plan = build_cell_plan(initial)
    history = (0, 1, 0)
    first.begin_request("session", plan, chunk_tokens=4, retain=[history])
    for slot in range(plan.num_slots):
        cell = plan.task(slot, 0)
        if cell is not None:
            _forward(first, (("session", cell),))
        first.evict(slot)
    first.finish_request("session")
    continuation = NoisyKVState(cache)
    assert continuation._retain is first._retain is cache._retained_history
    following = CellSchedule(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
        chunk_base=1,
    )
    prior = expected_prior_cells(following)
    assert prior == prior_history_in_pool(cache, "session", chunk_base=1)
    next_plan = build_cell_plan(following, prior_history=prior)
    continuation.begin_request("session", next_plan, chunk_tokens=4, retain=[(1, 1, 0)])
    with pytest.raises(RuntimeError, match="another state"):
        first.end_request("session")
    with pytest.raises(RuntimeError, match="Finish"):
        first.release_history("session")
    with pytest.raises(RuntimeError, match="already has an active"):
        first.begin_request("session", next_plan, chunk_tokens=4)
    for slot in range(next_plan.num_slots):
        cell = next_plan.task(slot, 0)
        if cell is not None:
            _forward(continuation, (("session", cell),))
        continuation.evict(slot)
    continuation.finish_request("session")
    assert not cache.pool.has(("session", *history))
    assert cache.pool.has(("session", 1, 1, 0))
    NoisyKVState(cache).release_history("session")
    assert not cache.pool.keys and not cache._retained_history


def test_noisy_release_preserves_other_session_and_retryable_receive():
    cache = _cache()
    state = NoisyKVState(cache)
    state.begin_request("A", _history_plan(), chunk_tokens=4)
    own, other = ("A", 0, 0, 0), ("B", 0, 0, 0)
    cache.pool.alloc(own)
    cache.pool.alloc(other)
    state.retain_history("A", [(0, 0, 0)])
    state.finish_request("A")

    class Handle:
        failed = False

        def wait(self):
            assert cache.pool.has(own)
            if not self.failed:
                self.failed = True
                raise RuntimeError("receive incomplete")

    cache.transport._pending_recv[own] = ([Handle()], None, 0)
    with pytest.raises(RuntimeError, match="receive incomplete"):
        NoisyKVState(cache).release_history("A")
    assert own in cache.transport._pending_recv
    assert cache.pool.has(own) and "A" in cache._retained_history
    NoisyKVState(cache).release_history("A")
    assert not cache.pool.has(own) and cache.pool.has(other)


@pytest.mark.parametrize("length", [-4, 0, 3, 16])
def test_noisy_invalid_lengths_do_not_admit_owner(length):
    cache = _cache(max_chunk_tokens=12)
    state = NoisyKVState(cache)
    with pytest.raises(ValueError, match="chunk_tokens"):
        state.begin_request("owner", _history_plan(), chunk_tokens=length)
    assert not cache._active_states and not cache.pool.keys


def test_noisy_eviction_obeys_computation_consumer_and_history_gates():
    cache = _cache()
    state = NoisyKVState(cache)
    plan = _history_plan()
    state.begin_request("owner", plan, chunk_tokens=4, t0=10)
    cell = next(plan.task(slot, 0) for slot in range(plan.num_slots) if plan.task(slot, 0) is not None)
    key = ("owner", *cell)
    _forward(state, (("owner", cell),))
    completed = state._completion_global["owner"][cell]
    last = state._last_use_global["owner"][cell]
    state.evict(completed - 1)
    assert cache.pool.has(key)
    state.evict(last - 1)
    assert cache.pool.has(key)
    state.retain_history("owner", [cell])
    state.evict(last)
    assert cache.pool.has(key)
    state.finish_request("owner")
    assert cache.pool.has(key)
    NoisyKVState(cache).release_history("owner")
    assert not cache.pool.has(key)
