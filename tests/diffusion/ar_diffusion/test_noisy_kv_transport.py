# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.experimental.ar_diffusion.cell_schedule import Cell, CellSchedule, Inflight, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    plan = build_cell_plan(
        CellSchedule(
            chunks=3,
            num_denoise_steps=1,
            step_replicas=2,
            layer_groups=1,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=2,
        )
    )
    state.begin_request("A", plan, chunk_tokens=chunk_tokens)
    state.set_inflight((Inflight(req="A", t0=0, plan=plan),))
    transfers = [(slot, xfer) for slot in range(plan.num_slots) for xfer in plan.transfers(slot)]
    assert len(transfers) == 1
    slot, xfer = transfers[0]
    key = ("A", *xfer.version)  # (req,c,s,b)
    pool = cache.pool
    shape = (chunk_tokens, spec.num_kv_heads, spec.head_size)
    layer = key[3] % spec.num_layers
    if rank == xfer.src:
        source_slot = pool.alloc(key)
        for li in range(spec.num_layers):
            pool.k_pools[li].fill_(77)
            pool.v_pools[li].fill_(77)
        start = source_slot * spec.max_chunk_tokens
        expected_key = torch.arange(
            chunk_tokens * spec.num_kv_heads * spec.head_size, dtype=torch.float32, device=device
        ).reshape(shape)
        pool.k_pools[layer][start : start + chunk_tokens] = expected_key + layer
        pool.v_pools[layer][start : start + chunk_tokens] = expected_key + layer + 0.5
    else:
        for li in range(spec.num_layers):
            pool.k_pools[li].fill_(-1)
            pool.v_pools[li].fill_(-1)

    handles = state.exchange(slot)
    for handle in handles:
        handle.wait()
    expected_bytes = 2 * chunk_tokens * spec.num_kv_heads * spec.head_size * 4
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


def _sink_pipeline_worker(
    rank: int,
    steps: int,
    groups: int,
    chunks: int,
    init_method: str,
    history: int = 1,
    fixed_shape: bool = False,
    device_kind: str = "cpu",
) -> None:
    from vllm_omni.experimental.ar_diffusion import cell_executor
    from vllm_omni.experimental.ar_diffusion.cell_executor import (
        ARDiffusionCellContext,
        CellAdapter,
        CellRunSpec,
        CellTopology,
        run_cell_pipeline,
    )

    world, blocks = steps * groups, 3
    backend = "nccl" if device_kind == "cuda" else "gloo"
    device = torch.device("cuda", rank) if device_kind == "cuda" else torch.device("cpu")
    if device_kind == "cuda":
        current_omni_platform.set_device(device)
    fixed_calls = 0
    original_post = cell_executor._post_activations_ws

    def record_fixed_post(*args, **kwargs):
        nonlocal fixed_calls
        fixed_calls += 1
        return original_post(*args, **kwargs)

    cell_executor._post_activations_ws = record_fixed_post
    try:
        torch.distributed.init_process_group(
            backend, init_method=init_method, rank=rank, world_size=world, timeout=timedelta(seconds=90)
        )
        pp = GroupCoordinator([list(range(world))], local_rank=rank, torch_distributed_backend=backend)
        schedule = CellSchedule(
            chunks=chunks,
            num_denoise_steps=steps,
            step_replicas=steps,
            layer_groups=groups,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=history,
            blocks=blocks,
        )
        plan = build_cell_plan(schedule)
        spec = ARDiffusionNoisyKVSpec(
            num_layers=blocks,
            num_kv_heads=1,
            head_size=1,
            block_size=1,
            max_chunk_tokens=1,
            max_history_chunks=max(1, history),
        )
        cache = NoisyKVCache(
            spec,
            dtype=torch.float32,
            device=device,
            layer_groups=groups,
            max_batch_size=1,
            step_replicas=steps,
        )
        state = NoisyKVState(cache)

        # Independent scalar forward reference: serial rollout, explicit clean
        # anchor, then same-step recent history. Communication never enters it.
        pages: dict[Cell, float] = {}
        inputs: dict[Cell, float] = {}
        outputs: dict[Cell, float] = {}
        latent: dict[int, float] = {}
        order = [(0, step, block) for step in range(steps + int(chunks > 1)) for block in range(blocks)]
        order += [
            (chunk, step, block) for chunk in range(1, chunks) for step in range(steps) for block in range(blocks)
        ]
        for chunk, step, block in order:
            if block == 0:
                value = float(chunk + 1) if step == 0 else latent[chunk]
            else:
                value = outputs[chunk, step, block - 1]
            inputs[chunk, step, block] = value
            sources = [(0, steps, block)] if chunk else []
            sources += [(c, step, block) for c in range(max(1, chunk - history), chunk)]
            pages[chunk, step, block] = value
            outputs[chunk, step, block] = value + sum(pages[x] for x in sources) * 0.01 + 0.1 * (block + 1)
            if block == blocks - 1 and step < steps:
                latent[chunk] = outputs[chunk, step, block] + 0.5

        class Adapter(CellAdapter):
            def __init__(self):
                self.device = device
                self.clean_calls = 0
                self.stash = {}
                self.completed = {}

            def activation_spec(self, *, batch, include_hidden):
                if not fixed_shape:
                    return None
                schema = [("latent", (batch,), torch.float32)]
                if include_hidden:
                    schema.append(("hidden_states", (batch,), torch.float32))
                return schema

            def stash_activation(self, payload, *, handles=None, meta=None):
                self.stash[meta[1:]] = payload

            def forward(self, tasks, kv_contexts, *, hidden):
                _req, cell = tasks[0]
                chunk, step, block = cell
                self.clean_calls += int(step == steps)
                payload = hidden if isinstance(hidden, dict) else self.stash.pop((chunk, step), None)
                if block == 0:
                    if step > 0:
                        assert payload is not None, "step input must come from the executor or transport"
                    value = (
                        payload["latent"] if payload is not None else torch.tensor([float(chunk + 1)], device=device)
                    )
                else:
                    assert payload is not None
                    value = payload["hidden_states"]
                torch.testing.assert_close(value, torch.tensor([inputs[cell]], device=device), rtol=1e-5, atol=1e-5)
                context = kv_contexts[0][0]
                n = context.seq_len - 1
                table = context.block_table[0, :n].long()
                past = context.key_pool[table].sum() if n else torch.tensor(0.0, device=device)
                slot = context.video_slots.long()
                context.key_pool[slot] = value.reshape(1, 1, 1)
                context.value_pool[slot] = value.reshape(1, 1, 1)
                result = value + past * 0.01 + 0.1 * (block + 1)
                torch.testing.assert_close(result, torch.tensor([outputs[cell]], device=device), rtol=1e-5, atol=1e-5)
                current = payload["latent"] if payload is not None else torch.tensor([float(chunk + 1)], device=device)
                if block == blocks - 1:
                    if step < steps:
                        current = result + 0.5
                    if step == steps - 1:
                        self.completed[chunk] = current
                    return {"latent": current}
                return {"latent": current, "hidden_states": result}

            def pack_activation(self, output):
                return output

        adapter = Adapter()
        ctx = ARDiffusionCellContext(
            spec=CellRunSpec(topology=CellTopology(steps, groups), rank=rank, pp_group=pp), kv=state
        )
        ctx.enqueue("A", plan, chunk_tokens=1)
        run_cell_pipeline(ctx=ctx, adapter=adapter)
        assert state.resident_versions == 0
        assert bool(fixed_calls) == fixed_shape
        expected_clean = sum(
            cell is not None and cell[1] == steps for cell in (plan.task(t, rank) for t in range(plan.num_slots))
        )
        assert adapter.clean_calls == expected_clean
        totals = torch.tensor([state.bytes_sent, state.bytes_received], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(totals)
        expected_bytes = sum(len(plan.transfers(t)) for t in range(plan.num_slots)) * 8
        assert totals.tolist() == [expected_bytes, expected_bytes]
        print(
            f"sink_pipeline rank={rank} device={device_kind} S={steps} G={groups} H={history} chunks={chunks} "
            f"fixed_calls={fixed_calls} clean_calls={adapter.clean_calls} kv_bytes={expected_bytes}",
            flush=True,
        )
        if rank == world - 1:
            assert len(adapter.completed) == chunks
            for chunk, value in adapter.completed.items():
                torch.testing.assert_close(value, torch.tensor([latent[chunk]], device=device), rtol=1e-5, atol=1e-5)
        pp.destroy()
        torch.distributed.destroy_process_group()
    finally:
        cell_executor._post_activations_ws = original_post


# Ready-lane budget: Simple · Engine&Entrypoints&Diffusion is 20 minutes and
# each Gloo spawn re-imports the stack (~30–60s wall). One scalar + one
# fixed-shape case here; the CUDA harness only varies chunks×history at
# fixed S=2,G=2 fixed-shape (no scalar NCCL twin).
@pytest.mark.parametrize("steps,groups,chunks", [(2, 1, 6)])
def test_same_step_sink_distributed_pipeline_matches_serial(steps, groups, chunks):
    torch.multiprocessing.spawn(
        _sink_pipeline_worker,
        args=(steps, groups, chunks, f"tcp://127.0.0.1:{get_open_port()}"),
        nprocs=steps * groups,
        join=True,
    )


@pytest.mark.parametrize("steps,groups,chunks,history", [(2, 2, 6, 1)])
def test_same_step_fixed_shape_activation_pipeline_matches_serial(steps, groups, chunks, history):
    torch.multiprocessing.spawn(
        _sink_pipeline_worker,
        args=(steps, groups, chunks, f"tcp://127.0.0.1:{get_open_port()}", history, True),
        nprocs=steps * groups,
        join=True,
    )
