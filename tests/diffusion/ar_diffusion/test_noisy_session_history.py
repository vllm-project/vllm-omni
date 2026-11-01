# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session history across ranks: the continuation must read what it kept.

A vertical topology produces a chunk's KV on one rank and reads it from
another, so a session that only kept local versions would silently continue
cold. These tests run the real cell pipeline over gloo and assert the values a
continuation reads, not just that it finished.
"""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.session import NoisyKVSession

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BLOCKS = 2
HISTORY = 1


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def marker(chunk: int, step: int, block: int) -> float:
    """Value a cell writes into its KV slot; unique per (chunk, step, block)."""
    return float((chunk + 1) * 100 + (step + 1) * 10 + block)


class _MarkerAdapter(CellAdapter):
    """Writes a known marker per cell and checks the history it reads back."""

    def __init__(self, device: torch.device, expected=None) -> None:
        self.device = device
        self.expected = expected
        self.checked = 0

    def forward(self, tasks, kv_contexts, *, hidden):
        del hidden
        for (_req, cell), ctx_list in zip(tasks, kv_contexts):
            ctx = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            chunk, step, block = cell
            slot = ctx.video_slots.long()
            value = torch.tensor([marker(chunk, step, block)], device=self.device)
            ctx.key_pool[slot] = value.reshape(1, 1, 1)
            ctx.value_pool[slot] = value.reshape(1, 1, 1)
            if self.expected is not None:
                want = self.expected(chunk, step, block)
                n = int(ctx.seq_len) - 1
                assert n == len(want), f"cell {cell} read {n} history blocks, expected {len(want)}"
                if want:
                    table = ctx.block_table[0, :n].long()
                    got = ctx.key_pool[table].reshape(-1).tolist()
                    assert got == want, f"cell {cell} history {got} != {want}"
                    self.checked += 1
        return torch.zeros(1, 1, 1)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def _expected_history(ordering: Ordering, steps: int, chunk_start: int):
    """History a continuation cell reads, computed from the schedule rule alone."""
    clean = steps

    def expected(local_chunk: int, step: int, block: int) -> list[float]:
        if local_chunk != 0:
            return []
        chunks = list(range(max(0, chunk_start - HISTORY), chunk_start))
        if ordering is Ordering.SAME_STEP and 0 not in chunks:
            chunks.insert(0, 0)
        out = []
        for chunk in chunks:
            if ordering is Ordering.SAME_STEP and chunk != 0:
                out.append(marker(chunk, step, block))
            else:
                out.append(marker(chunk, clean, block))
        return out

    return expected


def _session_history_worker(
    rank: int,
    init_method: str,
    ordering_name: str,
    steps: int,
    groups: int,
    world: int,
) -> None:
    ordering = Ordering(ordering_name)
    backend = "gloo"
    device = torch.device("cpu")
    torch.distributed.init_process_group(
        backend, init_method=init_method, rank=rank, world_size=world, timeout=timedelta(seconds=90)
    )
    pp = GroupCoordinator([list(range(world))], local_rank=rank, torch_distributed_backend=backend)
    try:
        schedule = CellSchedule(
            chunks=2,
            num_denoise_steps=steps,
            step_replicas=steps if ordering is Ordering.SAME_STEP else steps + 1,
            layer_groups=groups,
            ordering=ordering,
            kv_history_chunks=HISTORY,
            blocks=BLOCKS,
        )
        retained_versions = HISTORY * (steps if ordering is Ordering.SAME_STEP else 1) + 1
        spec = ARDiffusionNoisyKVSpec(
            num_layers=BLOCKS,
            num_kv_heads=1,
            head_size=1,
            block_size=1,
            max_chunk_tokens=1,
            max_history_chunks=HISTORY,
            max_history_versions=retained_versions,
        )
        cache = NoisyKVCache(
            spec,
            dtype=torch.float32,
            device=device,
            layer_groups=groups,
            max_batch_size=1,
            step_replicas=schedule.step_replicas,
            session_retention_chunks=spec.history_versions,
        )
        state = NoisyKVState(cache)
        state.bind_rank(rank, pp)
        session = NoisyKVSession(session_id="history")

        plan_a = build_cell_plan(schedule)
        ctx_a = ARDiffusionCellContext(
            spec=CellRunSpec(topology=CellTopology(schedule.step_replicas, groups), rank=rank, pp_group=pp),
            kv=state,
            chunk_start=0,
            session=session,
        )
        assert ctx_a.admit("A", plan_a, chunk_tokens=1, slot=0)
        run_cell_pipeline(ctx=ctx_a, adapter=_MarkerAdapter(device))

        chunk_start = session.committed_chunks
        assert chunk_start == 2, f"rank {rank} committed {chunk_start} chunks"

        plan_b = build_cell_plan(
            CellSchedule(
                chunks=1,
                num_denoise_steps=steps,
                step_replicas=schedule.step_replicas,
                layer_groups=groups,
                ordering=ordering,
                kv_history_chunks=HISTORY,
                blocks=BLOCKS,
            )
        )
        ctx_b = ARDiffusionCellContext(
            spec=CellRunSpec(topology=CellTopology(schedule.step_replicas, groups), rank=rank, pp_group=pp),
            kv=state,
            chunk_start=chunk_start,
            session=session,
        )
        assert ctx_b.admit("B", plan_b, chunk_tokens=1, slot=0)
        checker = _MarkerAdapter(device, _expected_history(ordering, steps, chunk_start))
        run_cell_pipeline(ctx=ctx_b, adapter=checker)
        assert checker.checked > 0, f"rank {rank} never read session history"
        assert session.committed_chunks == 3
    finally:
        pp.destroy()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("steps", [2])
def test_same_step_session_history_crosses_ranks(steps: int) -> None:
    world = steps
    torch.multiprocessing.spawn(
        _session_history_worker,
        args=(f"tcp://127.0.0.1:{get_open_port()}", Ordering.SAME_STEP.value, steps, 1, world),
        nprocs=world,
        join=True,
    )


@pytest.mark.parametrize("steps", [2])
def test_latest_kv_session_history_crosses_ranks(steps: int) -> None:
    world = steps + 1
    torch.multiprocessing.spawn(
        _session_history_worker,
        args=(f"tcp://127.0.0.1:{get_open_port()}", Ordering.LATEST_KV.value, steps, 1, world),
        nprocs=world,
        join=True,
    )
