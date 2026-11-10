# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native KV visibility, ring addresses and exported-buffer lifetime contracts."""

from collections import defaultdict

import pytest
import torch
from torch import nn

from benchmarks.wan import hybrid_transport
from benchmarks.wan.block_plan import BlockPlan, KVKey
from benchmarks.wan.hybrid_kv import HybridNoisyKVState
from benchmarks.wan.hybrid_transport import LayerMajorPages
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    CellSchedule,
    Ordering,
    build_cell_plan,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_noisy_kv_cache_accepts_benchmark_kwargs():
    """Call-site guard for the GPU-only rename blind spot in the benchmark.

    ``NoisyKVCache`` is constructed only inside the GPU benchmark job, so a
    kwarg rename on main would otherwise surface only there. Build it with
    the exact kwargs ``benchmark_noisy_pp`` passes, on CPU with a tiny spec,
    so the next rename fails here first. The schedule-side ``step_replicas``
    read is exercised on the real admit path by
    ``test_hybrid_adapter_satisfies_executor_protocol``.
    """
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVCache

    spec = ARDiffusionNoisyKVSpec(15, 12, 128, 8, 8, 6)
    kv = NoisyKVCache(
        spec,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layer_groups=2,
        max_batch_size=1,
        step_replicas=4,
        layer_offset=0,
        blocks_per_rank=15,
    )
    assert kv.blocks_per_rank == 15 and kv.pool.capacity > 0


def test_hybrid_adapter_satisfies_executor_protocol(monkeypatch):
    """The merged executor passes retain= on admit/drop and calls the
    prior-handoff hook after admission; the adapter must accept all three and
    fail fast on non-empty retain (the benchmark owns no retained history).
    Allocation is stubbed so this runs on CPU ( Bounty-hunter's P1)."""
    from types import SimpleNamespace

    import benchmarks.wan.hybrid_kv as hybrid_kv_module

    spec = ARDiffusionNoisyKVSpec(15, 12, 128, 8, 8, 6)
    model = SimpleNamespace(local_num_layers=2, start_layer=0)
    state = HybridNoisyKVState(spec, model, None, torch.device("cpu"), torch.bfloat16)
    # begin_request wraps this CellPlan in BlockPlan and reads plan.schedule.
    native = build_cell_plan(CellSchedule(7, 3, 4, 2, Ordering.LATEST_KV, 6, blocks=30, blocks_per_rank=15))
    # Must be a multiple of spec.block_size and <= spec.max_chunk_tokens.
    chunk_tokens = spec.block_size

    class FakePages:
        def __init__(self, plan, spec, chunk_tokens, device, dtype, group):
            del plan, chunk_tokens, group
            self.capacity = 4
            self.device = torch.device(device)
            self.sent_bytes = 0
            self.received_bytes = 0
            # begin_request views buffers[local, field] as (-1, num_kv_heads, head_size).
            self.buffers = torch.zeros(
                model.local_num_layers,
                2,
                spec.num_kv_heads * spec.head_size,
                dtype=dtype,
                device=device,
            )
            self.flags = torch.zeros(2, device=device)

        def close(self, abort=False):
            pass

    monkeypatch.setattr(hybrid_kv_module, "LayerMajorPages", FakePages)
    # Allocation stubbed: the CPU CI host cannot pin memory.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

    # Executor admission (cell_executor.admit) passes retain=, empty for
    # ordinary requests.
    state.begin_request("req", native, chunk_tokens=chunk_tokens, t0=0, retain=frozenset())
    assert state._chunk_tokens["req"] == chunk_tokens
    # The pre-forward prior-handoff hook in the noisy PP executor must
    # exist and be callable after admission.
    state.exchange_and_await_prior_handoffs(0)
    # Executor drop (cell_executor.drop) passes retain=() and expects the
    # per-request state gone.
    state.end_request("req", retain=())
    assert state.pages is None and "req" not in state._chunk_tokens

    # The benchmark owns no retained history: non-empty retain fails fast.
    with pytest.raises(ValueError, match="retained history"):
        state.begin_request("req2", native, chunk_tokens=chunk_tokens, t0=0, retain=frozenset({1}))
    with pytest.raises(ValueError, match="retained history"):
        state.end_request("req2", retain=(1,))


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("steps", [3, 4])
@pytest.mark.parametrize("chunks", [1, 7, 128])
def test_block_rounds_preserve_native_sources_and_last_reader(groups, steps, chunks):
    native = build_cell_plan(
        CellSchedule(chunks, steps, steps + 1, groups, Ordering.LATEST_KV, 6, blocks=30, blocks_per_rank=30 // groups)
    )
    plan = BlockPlan(native)
    assert len(plan.by_key) == chunks * (steps + 1) * 30
    consumers = defaultdict(list)
    for key, task in plan.by_key.items():
        cell = (key.chunk, key.step, key.block)
        assert native.task(task.tick, task.rank) == cell
        assert native.completion_tick(cell) == task.tick
        sources = native.sources(cell, task.rank)
        assert plan.reads[task] == tuple(KVKey(*source.version) for source in sources)
        for source, selected in zip(sources, plan.reads[task]):
            assert plan.owner(selected) == source.owner
            assert plan.by_key[selected].tick < task.tick
            consumers[selected, task.rank].append(task)
    for key, destinations in plan.destinations.items():
        producer_slot = plan.by_key[key].tick
        for rank, last in destinations.items():
            latest = max(consumers[key, rank], key=lambda task: task.tick).key
            assert last == (latest.chunk, latest.step, latest.block)
            assert native.task(producer_slot, rank) is not None


def test_ipc_addresses_point_to_the_contiguous_attention_pools():
    pages = object.__new__(LayerMajorPages)
    pages.capacity, pages.stages, pages.storage_blocks = 7, 5, 15
    pages.buffers = torch.empty((15, 2, 7, 5, 1, 8, 2, 4), dtype=torch.bfloat16)
    pages.bytes_per_tensor = 8 * 2 * 4 * pages.buffers.element_size()
    for chunk in (0, 6, 7, 14, 127):
        for step in range(5):
            for block in (15, 16, 29):
                key = KVKey(chunk, step, block)
                for field, value in enumerate(pages.page(key)):
                    address = pages.address(pages.buffers.data_ptr(), key, field)
                    assert value.data_ptr() == address
                    pool = pages.buffers[block - 15, field].view(-1, 2, 4)
                    slot = chunk % 7 * 5 + step
                    assert pool[slot * 8 : (slot + 1) * 8].data_ptr() == address
                    assert value.is_contiguous() and pool.is_contiguous()


def test_hybrid_state_constructor_does_not_allocate_a_second_kv_pool(monkeypatch):
    spec = ARDiffusionNoisyKVSpec(15, 12, 128, 8, 8, 6)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("hybrid state allocated storage before admission")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    state = HybridNoisyKVState(spec, nn.Module(), None, torch.device("cpu"), torch.bfloat16)
    assert state.pages is None and state.reserved_bytes == 0
    native = build_cell_plan(CellSchedule(7, 3, 4, 2, Ordering.LATEST_KV, 6, blocks=30, blocks_per_rank=15))
    with pytest.raises(ValueError, match="chunk tokens"):
        state.begin_request("invalid", native, chunk_tokens=0)
    assert not state._chunk_tokens


def test_abort_retains_exported_storage_and_close_is_idempotent(monkeypatch):
    retained: list[LayerMajorPages] = []
    monkeypatch.setattr(hybrid_transport, "_ABORTED", retained)
    pages = object.__new__(LayerMajorPages)
    pages.closed = False
    pages.buffers = torch.empty(1)
    exported = pages.buffers
    pages.close(abort=True)
    pages.close()
    assert retained == [pages]
    assert pages.closed and pages.buffers is exported
