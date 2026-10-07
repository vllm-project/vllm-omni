# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.experimental.ar_diffusion.chunk_schedule import ChunkSchedule, Inflight, Ordering, build_chunk_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _transfer_worker(rank: int, chunk_tokens: int, init_method: str, device_kind: str, mode: str) -> None:
    backend = "nccl" if device_kind == "cuda" else "gloo"
    if device_kind == "cuda":
        current_omni_platform.set_device(torch.device("cuda", rank))
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    pp_group = GroupCoordinator([[0, 1]], local_rank=rank, torch_distributed_backend=backend)
    device = torch.device("cuda", rank) if device_kind == "cuda" else torch.device("cpu")
    spec = ARDiffusionNoisyKVSpec(
        num_layers=2, num_kv_heads=2, head_size=4, block_size=4, max_chunk_tokens=12, max_history_chunks=2
    )
    cache = NoisyKVCache(spec, dtype=torch.float32, device=device, layer_groups=1, max_batch_size=1)
    state = NoisyKVState(cache)
    state.bind_rank(rank, pp_group)
    plan = build_chunk_plan(
        ChunkSchedule(
            chunks=3,
            num_denoise_steps=1,
            stages=2,
            layer_groups=1,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=2,
        )
    )
    state.begin_request("A", plan, chunk_tokens=chunk_tokens)
    state.set_inflight((Inflight(req="A", t0=0, plan=plan),))
    transfers = [(slot, xfer) for slot in range(plan.num_slots) for xfer in plan.transfers(slot)]
    assert len(transfers) == 1
    slot, xfer = transfers[0]
    key = ("A", *xfer.version)
    pool = cache.pool
    shape = (chunk_tokens, spec.num_kv_heads, spec.head_size)
    if rank == xfer.src:
        source_slot = pool.alloc(key)
        for layer in range(spec.num_layers):
            pool.k_pools[layer].fill_(77)
            pool.v_pools[layer].fill_(77)
            start = source_slot * spec.max_chunk_tokens
            expected_key = torch.arange(
                chunk_tokens * spec.num_kv_heads * spec.head_size, dtype=torch.float32, device=device
            ).reshape(shape)
            pool.k_pools[layer][start : start + chunk_tokens] = expected_key + layer
            pool.v_pools[layer][start : start + chunk_tokens] = expected_key + layer + 0.5
    else:
        for layer in range(spec.num_layers):
            pool.k_pools[layer].fill_(-1)
            pool.v_pools[layer].fill_(-1)

    handles = state.exchange(slot)
    for handle in handles:
        handle.wait()
    expected_bytes = 2 * spec.num_layers * chunk_tokens * spec.num_kv_heads * spec.head_size * 4
    if mode != "immediate":
        assert cache.transport.await_ready(frozenset()) == (1 if rank == xfer.dst else 0)
        torch.distributed.barrier()
        cache.transport.exchange((), rank=rank, pp_group=pp_group, chunk_tokens_by_request={"A": chunk_tokens})
        if mode == "discard":
            if rank == xfer.dst:
                state.evict(plan.last_use(rank)[xfer.version])
                assert not pool.has(key)
                assert key not in cache.transport._pending_recv
                assert cache.transport.bytes_received == expected_bytes
            torch.distributed.barrier()
            state.end_request("A")
            pp_group.destroy()
            torch.distributed.destroy_process_group()
            return
        assert mode == "delayed"
        assert cache.transport.await_ready(frozenset({key})) == 0
    else:
        state.await_ready(slot)

    if rank == xfer.src:
        assert state.bytes_sent == expected_bytes
    else:
        assert cache.transport.bytes_received == expected_bytes
        received_slot = pool.slot_of(key)
        start = received_slot * spec.max_chunk_tokens
        for layer in range(spec.num_layers):
            expected_key = torch.arange(
                chunk_tokens * spec.num_kv_heads * spec.head_size, dtype=torch.float32, device=device
            ).reshape(shape)
            torch.testing.assert_close(pool.k_pools[layer][start : start + chunk_tokens], expected_key + layer)
            torch.testing.assert_close(pool.v_pools[layer][start : start + chunk_tokens], expected_key + layer + 0.5)
            assert torch.all(pool.k_pools[layer][start + chunk_tokens : start + spec.max_chunk_tokens] == -1)
            assert torch.all(pool.v_pools[layer][start + chunk_tokens : start + spec.max_chunk_tokens] == -1)
    print(
        f"rank={rank} device={device_kind} chunk_tokens={chunk_tokens} "
        f"sent={state.bytes_sent} received={state.bytes_received}",
        flush=True,
    )
    torch.distributed.barrier()
    state.end_request("A")
    pp_group.destroy()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("chunk_tokens", [4, 8, 12])
@pytest.mark.parametrize("mode", ["immediate", "delayed"])
def test_noisy_kv_transfer_uses_valid_blocks(chunk_tokens: int, mode: str) -> None:
    torch.multiprocessing.spawn(
        _transfer_worker,
        args=(chunk_tokens, f"tcp://127.0.0.1:{get_open_port()}", "cpu", mode),
        nprocs=2,
    )


def test_unused_inbound_kv_finishes_before_page_reuse() -> None:
    torch.multiprocessing.spawn(
        _transfer_worker,
        args=(4, f"tcp://127.0.0.1:{get_open_port()}", "cpu", "discard"),
        nprocs=2,
    )
