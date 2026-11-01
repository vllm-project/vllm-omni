# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Edge cases for session history that only show up with more than one rank.

Two contracts are probed here:

* Same-Step reads chunk 0's clean sink from every later chunk, for the life of
  the session. That sink is not a "recent H chunks" artefact, so it has to
  survive a session configured with ``kv_history_chunks=0``.
* A continuation whose history a rank already holds should not have to receive
  it again.
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


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _SeqRecorder(CellAdapter):
    """Records the KV length each cell sees and writes into its slot."""

    def __init__(self) -> None:
        self.seq_lens: list[int] = []

    def forward(self, tasks, kv_contexts, *, hidden):
        del hidden
        for (_req, cell), ctx_list in zip(tasks, kv_contexts):
            view = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            self.seq_lens.append(int(view.seq_len))
            slot = view.video_slots.long()
            view.key_pool[slot] = torch.ones_like(view.key_pool[slot])
            view.value_pool[slot] = torch.ones_like(view.value_pool[slot])
            del cell
        return torch.zeros(1, 1, 1)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def _schedule(*, chunks: int, steps: int, replicas: int, ordering: Ordering, history: int) -> CellSchedule:
    return CellSchedule(
        chunks=chunks,
        num_denoise_steps=steps,
        step_replicas=replicas,
        layer_groups=1,
        ordering=ordering,
        kv_history_chunks=history,
        blocks=BLOCKS,
    )


def _cache(schedule: CellSchedule, history: int) -> NoisyKVCache:
    spec = ARDiffusionNoisyKVSpec(
        num_layers=BLOCKS,
        num_kv_heads=1,
        head_size=1,
        block_size=1,
        max_chunk_tokens=1,
        max_history_chunks=max(1, history),
        max_history_versions=max(1, history),
    )
    return NoisyKVCache(
        spec,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=schedule.step_replicas,
        session_retention_chunks=spec.history_versions,
    )


def _sink_without_history_worker(rank: int, init_method: str, steps: int) -> None:
    """Same-Step, ``kv_history_chunks=0``: the sink must still reach a continuation."""
    world = steps
    torch.distributed.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world, timeout=timedelta(seconds=90)
    )
    pp = GroupCoordinator([list(range(world))], local_rank=rank, torch_distributed_backend="gloo")
    try:
        schedule = _schedule(chunks=2, steps=steps, replicas=steps, ordering=Ordering.SAME_STEP, history=0)
        cache = _cache(schedule, history=0)
        state = NoisyKVState(cache)
        state.bind_rank(rank, pp)
        session = NoisyKVSession(session_id="sink")

        plan_a = build_cell_plan(schedule)
        ctx_a = ARDiffusionCellContext(
            spec=CellRunSpec(topology=CellTopology(steps, 1), rank=rank, pp_group=pp),
            kv=state,
            chunk_start=0,
            session=session,
        )
        assert ctx_a.admit("A", plan_a, chunk_tokens=1, slot=0)
        run_cell_pipeline(ctx=ctx_a, adapter=_SeqRecorder())

        follow_up = build_cell_plan(
            _schedule(chunks=1, steps=steps, replicas=steps, ordering=Ordering.SAME_STEP, history=0)
        )
        ctx_b = ARDiffusionCellContext(
            spec=CellRunSpec(topology=CellTopology(steps, 1), rank=rank, pp_group=pp),
            kv=state,
            chunk_start=session.committed_chunks,
            session=session,
        )
        assert ctx_b.admit("B", follow_up, chunk_tokens=1, slot=0)
        recorder = _SeqRecorder()
        run_cell_pipeline(ctx=ctx_b, adapter=recorder)

        # Same-Step always reads chunk 0's clean sink, so the continuation's first
        # cell covers one history chunk plus its own write.
        assert recorder.seq_lens[0] == 2, (
            f"rank {rank} read {recorder.seq_lens[0]} tokens of context; "
            "Same-Step requires the chunk 0 sink even when H=0"
        )
    finally:
        pp.destroy()
        torch.distributed.destroy_process_group()


def _repeat_transfer_worker(rank: int, init_method: str, steps: int) -> None:
    """A continuation must not re-receive history an earlier one delivered."""
    world = steps + 1
    torch.distributed.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world, timeout=timedelta(seconds=90)
    )
    pp = GroupCoordinator([list(range(world))], local_rank=rank, torch_distributed_backend="gloo")
    try:

        def schedule_for(chunks: int) -> CellSchedule:
            return _schedule(chunks=chunks, steps=steps, replicas=steps + 1, ordering=Ordering.LATEST_KV, history=2)

        cache = _cache(schedule_for(2), history=2)
        state = NoisyKVState(cache)
        state.bind_rank(rank, pp)
        session = NoisyKVSession(session_id="repeat")

        def run(req: str, chunks: int) -> int:
            before = cache.transport.bytes_received
            ctx = ARDiffusionCellContext(
                spec=CellRunSpec(topology=CellTopology(steps + 1, 1), rank=rank, pp_group=pp),
                kv=state,
                chunk_start=session.committed_chunks,
                session=session,
            )
            assert ctx.admit(req, build_cell_plan(schedule_for(chunks)), chunk_tokens=1, slot=0)
            received = cache.transport.bytes_received - before
            run_cell_pipeline(ctx=ctx, adapter=_SeqRecorder())
            return received

        run("A", 2)
        # The first continuation reads the whole two-chunk window.
        first = run("B", 1)
        # The second only needs the chunk the first one produced; chunk 1 of the
        # window was already delivered.
        second = run("C", 1)

        if rank == steps:
            # The producing rank only sends; it never receives its own history.
            assert (first, second) == (0, 0)
            return
        assert first > 0, f"rank {rank} received no history at all"
        assert second * 2 == first, (
            f"rank {rank} received {second} bytes on the second continuation and {first} on the first; "
            "the already-delivered chunk was sent again"
        )
    finally:
        pp.destroy()
        torch.distributed.destroy_process_group()


def test_same_step_keeps_its_sink_when_the_history_window_is_zero() -> None:
    steps = 2
    torch.multiprocessing.spawn(
        _sink_without_history_worker,
        args=(f"tcp://127.0.0.1:{get_open_port()}", steps),
        nprocs=steps,
        join=True,
    )


def test_a_continuation_does_not_re_receive_history_it_already_holds() -> None:
    steps = 2
    torch.multiprocessing.spawn(
        _repeat_transfer_worker,
        args=(f"tcp://127.0.0.1:{get_open_port()}", steps),
        nprocs=steps + 1,
        join=True,
    )
