# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native KV visibility, ring addresses and exported-buffer lifetime contracts."""

from collections import defaultdict

import pytest
import torch
from torch import nn

from benchmarks.ar_diffusion import hybrid_transport
from benchmarks.ar_diffusion.block_plan import BlockPlan, KVKey
from benchmarks.ar_diffusion.hybrid_kv import HybridNoisyKVState
from benchmarks.ar_diffusion.hybrid_transport import LayerMajorPages
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
