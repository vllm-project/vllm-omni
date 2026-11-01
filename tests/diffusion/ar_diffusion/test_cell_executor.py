# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU executor loop with a stub adapter (no distributed)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    _consumer_stash_meta,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CountingAdapter(CellAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def forward(self, tasks, kv_contexts, *, hidden):
        self.calls.append(tasks)
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def test_consumer_stash_meta_cross_stage_vs_same_stage():
    """Stash key is the consumer cell: step+1 only when leaving a denoise stage."""
    # S=D+1, G=2: ranks 0,1 step0; 2,3 step1. Producer (c,s) → consumer key.
    topo = CellTopology(step_replicas=2, layer_groups=2)
    src_tasks = (("r", (3, 0, 1)),)  # chunk=3, step=0, last block of group
    # Mid-stage: rank 0 → 1, keep step.
    assert _consumer_stash_meta(src_tasks, topology=topo, src=0) == ("", 3, 0)
    # Cross stage: rank 1 (stage-last) → 2, consumer runs step+1.
    assert _consumer_stash_meta(src_tasks, topology=topo, src=1) == ("", 3, 1)
    # S=1 wrap last→0 also stage-last → next denoise step.
    topo_s1 = CellTopology(step_replicas=1, layer_groups=3)
    assert _consumer_stash_meta(src_tasks, topology=topo_s1, src=2) == ("", 3, 1)


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
    run_cell_pipeline(ctx=ctx, adapter=adapter)
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
    run_cell_pipeline(ctx=ctx, adapter=adapter)
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
        def forward(self, tasks, kv_contexts, *, hidden):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_cell_pipeline(ctx=ctx, adapter=BoomAdapter())
    assert not ctx.inflight
    assert not ctx.pending
    assert len(cache.pool.keys) == 0


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("chunks", [1, 3])
def test_sink_clean_activation_edges_follow_final_replica(groups, chunks):
    from vllm_omni.experimental.ar_diffusion.cell_executor import _activation_edges
    from vllm_omni.experimental.ar_diffusion.cell_schedule import Inflight, rank_of_block

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
        def forward(self, tasks, kv_contexts, *, hidden):
            _req, cell = tasks[0]
            if cell == (0, 1, 0):
                assert hidden is not None, "executor dropped the final latent before clean"
                torch.testing.assert_close(hidden["latent"], torch.tensor([7.0]))
            else:
                # Neither the initial chunk nor the tail may inherit stale input.
                assert hidden is None
            return {"latent": torch.tensor([7.0])}

    assert ctx.admit("A", plan, chunk_tokens=1, slot=0)
    run_cell_pipeline(ctx=ctx, adapter=StatelessAdapter())
    assert not ctx.inflight
    assert ctx.kv.resident_versions == 0
