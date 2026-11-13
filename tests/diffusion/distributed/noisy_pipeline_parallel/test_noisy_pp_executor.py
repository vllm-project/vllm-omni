# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU executor loop with a stub adapter (no distributed)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    CellSchedule,
    Ordering,
    build_cell_plan,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CountingAdapter(CellAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def compatibility_key(self, cell):
        return cell

    def forward(self, batch, members, *, member_inputs):
        del members, member_inputs
        tasks = tuple((m.request_id, (m.chunk, m.step, m.block)) for m in batch.members)
        self.calls.append(tasks)
        return {"hidden_states": torch.zeros(batch.size, 4, 8)}

    def pack_activation(self, output):
        return output if isinstance(output, dict) else {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def test_consumer_stash_meta_cross_stage_vs_same_stage():
    """Stash key is per-request consumer (chunk, step): step+1 only when leaving a stage."""
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
        _consumer_stash_meta_for_task,
    )

    # S=D+1, G=2: ranks 0,1 step0; 2,3 step1. Producer (c,s) → consumer key.
    topo = CellTopology(step_replicas=2, layer_groups=2)
    cell = (3, 0, 1)  # chunk=3, step=0, last block of group
    # Mid-stage: rank 0 → 1, keep step.
    assert _consumer_stash_meta_for_task("r", cell, topology=topo, src=0) == ("r", 3, 0)
    # Cross stage: rank 1 (stage-last) → 2, consumer runs step+1.
    assert _consumer_stash_meta_for_task("r", cell, topology=topo, src=1) == ("r", 3, 1)
    # S=1 wrap last→0 also stage-last → next denoise step.
    topo_s1 = CellTopology(step_replicas=1, layer_groups=3)
    assert _consumer_stash_meta_for_task("r", cell, topology=topo_s1, src=2) == ("r", 3, 1)


def test_executor_runs_all_slots_r1():
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=2,
            num_kv_heads=2,
            head_size=4,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    kv = NoisyKVState(cache)
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=kv,
    )
    assert ctx.admit("A", plan, chunk_tokens=4, slot=0)
    adapter = CountingAdapter()
    run_cell_pipeline(ctx=ctx, members={item.req: adapter for item in (*ctx.inflight, *ctx.pending)})
    assert adapter.calls
    assert not ctx.inflight
    assert kv.resident_versions == 0


def test_mixed_chunk_tokens_refused_while_inflight():
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=2,
            head_size=4,
            block_size=4,
            max_chunk_tokens=8,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=2,
    )
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), max_batch_size=2, rank=0),
        kv=NoisyKVState(cache),
    )
    assert ctx.admit("A", plan, chunk_tokens=4, slot=0)
    assert not ctx.admit("B", plan, chunk_tokens=8, slot=0)


def test_continuous_batch_two_requests_same_tokens():
    plan = build_cell_plan(
        CellSchedule(
            chunks=1,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=2,
            head_size=4,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=2,
    )
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), max_batch_size=2, rank=0),
        kv=NoisyKVState(cache),
    )
    ctx.enqueue("A", plan, chunk_tokens=4)
    ctx.enqueue("B", plan, chunk_tokens=4)
    adapter = CountingAdapter()
    run_cell_pipeline(ctx=ctx, members={item.req: adapter for item in (*ctx.inflight, *ctx.pending)})
    reqs_seen = {req for call in adapter.calls for req, _task in call}
    assert reqs_seen == {"A", "B"}
    assert not ctx.inflight
    assert not ctx.pending


def test_executor_releases_versions_when_forward_raises():
    plan = build_cell_plan(
        CellSchedule(
            chunks=1,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=2,
            head_size=4,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    kv = NoisyKVState(cache)
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=kv,
    )
    assert ctx.admit("A", plan, chunk_tokens=4, slot=0)

    class BoomAdapter(CountingAdapter):
        def forward(self, batch, members, *, member_inputs):
            del batch, members, member_inputs
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        adapter = BoomAdapter()
        run_cell_pipeline(ctx=ctx, members={item.req: adapter for item in (*ctx.inflight, *ctx.pending)})
    assert not ctx.inflight
    assert not ctx.pending
    assert len(cache.pool.keys) == 0


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("chunks", [1, 3])
def test_sink_clean_activation_edges_follow_final_replica(groups, chunks):
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import _activation_edges
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Inflight, rank_of_block

    schedule = CellSchedule(
        chunks=chunks,
        num_denoise_steps=2,
        step_replicas=2,
        layer_groups=groups,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=1,
        blocks=4,
    )
    plan = build_cell_plan(schedule)
    topology = CellTopology(step_replicas=2, layer_groups=groups)
    inflight = (Inflight(req="A", t0=0, plan=plan),)
    # The final noisy pass must wrap within its replica to the clean pass,
    # instead of dropping the latent on the last physical rank.
    tick = plan.completion_tick((0, 1, 3))
    edges = _activation_edges(topology=topology, inflight=inflight, slot=tick, schedule=schedule)
    src, dst = rank_of_block(1, 3, schedule), rank_of_block(2, 0, schedule)
    assert edges == ([] if chunks == 1 or src == dst else [(src, dst, (("A", (0, 1, 3)),))])
    if chunks == 1:
        return
    for chunk, step in [(0, 2), (1, 1), (2, 1)]:
        tick = plan.completion_tick((chunk, step, 3))
        edges = _activation_edges(topology=topology, inflight=inflight, slot=tick, schedule=schedule)
        assert not any(task == (chunk, step, 3) for _, _, tasks in edges for _, task in tasks)


def test_sink_bootstrap_requests_are_admitted_sequentially():
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=1,
        )
    )
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1, num_kv_heads=2, head_size=4, block_size=4, max_chunk_tokens=4, max_history_chunks=1
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=2,
    )
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(1, 1), max_batch_size=2), kv=NoisyKVState(cache)
    )
    assert ctx.admit("A", plan, chunk_tokens=4, slot=0)
    assert not ctx.admit("B", plan, chunk_tokens=4, slot=0)
    ctx.drop("A")
    assert ctx.admit("B", plan, chunk_tokens=4, slot=0)


def test_same_rank_clean_receives_final_latent_without_adapter_cache():
    """Executor keeps the final latent when last-noisy → clean stays on one rank.

    Cross-rank wraps (``G>1``) belong in the distributed fixed-shape harness.
    """
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=0,
        )
    )
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            block_size=1,
            max_chunk_tokens=1,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(1, 1), rank=0),
        kv=NoisyKVState(cache),
    )

    class StatelessAdapter(CellAdapter):
        def compatibility_key(self, cell):
            return cell

        def forward(self, batch, members, *, member_inputs):
            del members
            m0 = batch.members[0]
            cell = (m0.chunk, m0.step, m0.block)
            hidden = member_inputs[0] if member_inputs else None
            if cell == (0, 1, 0):
                assert hidden is not None, "executor dropped the final latent before clean"
                torch.testing.assert_close(hidden["latent"], torch.tensor([7.0]))
            else:
                # Neither the initial chunk nor the tail may inherit stale input.
                assert hidden is None
            return {"latent": torch.tensor([7.0]).expand(batch.size).clone()}

    assert ctx.admit("A", plan, chunk_tokens=1, slot=0)
    adapter = StatelessAdapter()
    run_cell_pipeline(ctx=ctx, members={item.req: adapter for item in (*ctx.inflight, *ctx.pending)})
    assert not ctx.inflight
    assert ctx.kv.resident_versions == 0


def test_run_slot_batches_uses_per_request_stash_rows():
    """PP stashes one row per request; max_batch_size splits pop by identity."""
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
        _run_slot_batches,
    )

    cell = (0, 0, 0)
    tasks = (("A", cell), ("B", cell), ("C", cell))
    seen: list[list[float]] = []

    class _Adapter(CellAdapter):
        def compatibility_key(self, c):
            return c[2]  # block only

        def forward(self, batch, members, *, member_inputs):
            del members
            rows = []
            for inp in member_inputs:
                assert isinstance(inp, dict) and "latent" in inp
                rows.append(float(inp["latent"].reshape(-1)[0]))
            seen.append(rows)
            return {"latent": torch.zeros(batch.size, 1)}

    members = {"A": _Adapter(), "B": _Adapter(), "C": _Adapter()}
    act_stash = {
        ("A", 0, 0): {"latent": torch.full((1,), 1.0)},
        ("B", 0, 0): {"latent": torch.full((1,), 2.0)},
        ("C", 0, 0): {"latent": torch.full((1,), 3.0)},
    }
    _run_slot_batches(
        tasks=tasks,
        contexts=[None, None, None],
        members=members,
        max_batch_size=2,
        pending_hidden={},
        act_stash=act_stash,
    )
    assert act_stash == {}
    assert seen == [[1.0, 2.0], [3.0]]


def test_pack_task_outputs_skips_incompatible_latent_geometry():
    """Same token count but different H/W must not torch.cat on the slot pack."""
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
        _pack_task_outputs,
    )

    a = {"latent": torch.zeros(1, 4, 1, 2, 4)}
    b = {"latent": torch.zeros(1, 4, 1, 4, 2)}
    assert a["latent"].numel() == b["latent"].numel()
    assert _pack_task_outputs([a, b]) is None
    packed = _pack_task_outputs([a, {"latent": torch.ones(1, 4, 1, 2, 4)}])
    assert packed is not None
    assert packed["latent"].shape == (2, 4, 1, 2, 4)


def test_mixed_latent_geometry_same_slot_retains_shapes():
    """Two requests with equal tokens but swapped H/W both finish with own shapes."""
    plan = build_cell_plan(
        CellSchedule(
            chunks=1,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    # 1*4*1*2*4 == 1*4*1*4*2 == 32 tokens; pool uses max_chunk_tokens only.
    shape_a = (1, 4, 1, 2, 4)
    shape_b = (1, 4, 1, 4, 2)
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=2,
            head_size=4,
            block_size=4,
            max_chunk_tokens=32,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=2,
    )
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), max_batch_size=2, rank=0),
        kv=NoisyKVState(cache),
    )
    ctx.enqueue("A", plan, chunk_tokens=32)
    ctx.enqueue("B", plan, chunk_tokens=32)

    finished: dict[str, tuple[int, ...]] = {}

    class _GeomAdapter(CellAdapter):
        def __init__(self, req: str, shape: tuple[int, ...]) -> None:
            self.req = req
            self.shape = shape

        def compatibility_key(self, cell):
            return (cell[2], self.shape)

        def forward(self, batch, members, *, member_inputs):
            del member_inputs
            latents = []
            for member_ref, member in zip(batch.members, members, strict=True):
                assert isinstance(member, _GeomAdapter)
                latent = torch.full(member.shape, float(ord(member.req[0])))
                finished[member_ref.request_id] = tuple(latent.shape)
                latents.append(latent)
            # Homogeneous ComputeBatch may cat; cross-geometry packing is executor-side.
            return {"latent": torch.cat(latents, dim=0)}

    run_cell_pipeline(
        ctx=ctx,
        members={"A": _GeomAdapter("A", shape_a), "B": _GeomAdapter("B", shape_b)},
    )
    assert finished == {"A": shape_a, "B": shape_b}
    assert not ctx.inflight
    assert not ctx.pending


def test_expand_activation_edges_agrees_without_local_outs():
    """Sender/receiver expand identically from shared member geometry, not local outs.

    Simulates two ranks whose current Cells differ: the receiver may not have
    computed the edge's src_tasks yet. Expansion must still match the sender.
    """
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
        _expand_activation_edges_for_geometry,
    )

    class _SpecAdapter(CellAdapter):
        def __init__(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> None:
            self._shape = shape
            self._dtype = dtype
            self.latent_shape = shape
            self.dtype = dtype

        def compatibility_key(self, cell):
            return (cell[2], self._shape, str(self._dtype))

        def forward(self, batch, members, *, member_inputs):
            del batch, members, member_inputs
            raise AssertionError("expand must not run forward")

        def activation_spec(self, *, batch: int, include_hidden: bool):
            _b, c, t, h, w = self._shape
            spec: list[tuple[str, tuple[int, ...], torch.dtype]] = [("latent", (batch, c, t, h, w), self._dtype)]
            if include_hidden:
                spec.append(("hidden_states", (batch, t * h * w, 8), self._dtype))
            return spec

    topology = CellTopology(step_replicas=1, layer_groups=2)
    same_a = _SpecAdapter((1, 4, 1, 2, 4))
    same_b = _SpecAdapter((1, 4, 1, 2, 4))
    mixed_a = _SpecAdapter((1, 4, 1, 2, 4))
    mixed_b = _SpecAdapter((1, 4, 1, 4, 2))
    # Receiver may only know a fallback adapter for its own current Cell.
    fallback = _SpecAdapter((1, 4, 1, 8, 8))

    same_edge = [(0, 1, (("A", (0, 0, 0)), ("B", (0, 0, 0))))]
    mixed_edge = [(0, 1, (("A", (1, 0, 0)), ("B", (1, 0, 0))))]

    # Two ranks: sender holds outs for A/B; receiver is on an unrelated Cell and
    # has empty local outs. Both expand from the shared members map only.
    sender_members = {"A": same_a, "B": same_b}
    receiver_members = {"A": same_a, "B": same_b}
    sender_same = _expand_activation_edges_for_geometry(
        same_edge, members=sender_members, topology=topology, fallback=fallback
    )
    receiver_same = _expand_activation_edges_for_geometry(
        same_edge, members=receiver_members, topology=topology, fallback=fallback
    )
    assert sender_same == receiver_same == same_edge

    sender_members = {"A": mixed_a, "B": mixed_b}
    receiver_members = {"A": mixed_a, "B": mixed_b}
    sender_mixed = _expand_activation_edges_for_geometry(
        mixed_edge, members=sender_members, topology=topology, fallback=fallback
    )
    receiver_mixed = _expand_activation_edges_for_geometry(
        mixed_edge, members=receiver_members, topology=topology, fallback=fallback
    )
    assert sender_mixed == receiver_mixed
    assert sender_mixed == [
        (0, 1, (("A", (1, 0, 0)),)),
        (0, 1, (("B", (1, 0, 0)),)),
    ]


def test_activation_edges_only_send_boundary_members():
    """A boundary member must not drag a mid-group member into PP transport."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import _activation_edges

    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=2,
        blocks=4,
        kv_history_chunks=1,
        ordering=Ordering.SELF_FORCING,
    )

    def item(req, cell):
        return SimpleNamespace(
            req=req,
            t0=0,
            plan=SimpleNamespace(
                schedule=schedule,
                task=lambda _slot, rank: cell if rank == 0 else None,
            ),
        )

    inflight = (item("boundary", (0, 0, 1)), item("local", (0, 0, 0)))
    edges = _activation_edges(topology=CellTopology(1, 2), inflight=inflight, slot=0, schedule=schedule)
    assert edges == [(0, 1, (("boundary", (0, 0, 1)),))]
