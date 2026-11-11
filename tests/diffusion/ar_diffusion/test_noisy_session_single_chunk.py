# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for one-chunk Same-Step session sink lifecycle."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellSchedule,
    CellTopology,
    Ordering,
    build_cell_plan,
    run_cell_pipeline,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import expected_prior_cells
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
    clean_history_cells,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Recorder(CellAdapter):
    def __init__(self) -> None:
        self.seq_lens: list[int] = []

    def forward(self, tasks, kv_contexts, *, hidden):
        del hidden
        for _task, contexts in zip(tasks, kv_contexts):
            context = contexts[0] if isinstance(contexts, list) else contexts
            self.seq_lens.append(context.seq_len)
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def _schedule(*, chunk_base: int, chunks: int = 1, persistent: bool = False) -> CellSchedule:
    return CellSchedule(
        chunks=chunks,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=2,
        blocks=1,
        blocks_per_rank=1,
        chunk_base=chunk_base,
        same_step_session=persistent,
    )


def _cache() -> NoisyKVCache:
    return NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=2,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=1,
    )


def _run(cache: NoisyKVCache, session_id: str, plan, *, persist: bool = False) -> _Recorder:
    kv = NoisyKVState(cache)
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=kv,
    )
    retain = clean_history_cells(plan.schedule, num_blocks=1) if persist else set()
    ctx.enqueue(session_id, plan, chunk_tokens=4, retain=retain)
    recorder = _Recorder()
    run_cell_pipeline(ctx=ctx, adapter=recorder)
    kv.end_request(session_id, retain=retain)
    return recorder


def test_persistent_single_chunk_same_step_writes_and_retains_its_sink():
    cache = _cache()
    schedule = _schedule(chunk_base=0, persistent=True)
    plan = build_cell_plan(schedule)
    produced = {plan.task(slot, rank) for slot in range(plan.num_slots) for rank in range(plan.world)} - {None}

    recorder = _run(cache, "session", plan, persist=True)

    assert (0, 1, 0) in produced, "persistent request skipped the clean sink"
    assert ("session", 0, 1, 0) in cache.pool.keys, "the sink was not retained for continuation"
    assert len(recorder.seq_lens) == 2, "persistent admission must execute the noisy cell and clean sink"
    assert (0, 1, 0) in clean_history_cells(schedule, num_blocks=1)


def test_single_chunk_same_step_continuation_reads_sink_within_view_width():
    cache = _cache()
    first = _schedule(chunk_base=0, chunks=4, persistent=True)
    _run(cache, "session", build_cell_plan(first), persist=True)
    continuation = _schedule(chunk_base=4)
    prior = expected_prior_cells(continuation)

    assert prior == frozenset({(0, 1, 0), (2, 0, 0), (3, 0, 0)})
    assert {("session", *cell) for cell in prior} <= set(cache.pool.keys)
    plan = build_cell_plan(continuation, prior_history=prior)
    recorder = _run(cache, "session", plan, persist=True)

    # Two trailing versions, the clean sink, and the current write are all live.
    assert recorder.seq_lens == [16]


def test_one_shot_single_chunk_same_step_keeps_legacy_no_sink_schedule():
    schedule = _schedule(chunk_base=0)
    plan = build_cell_plan(schedule)
    produced = {plan.task(slot, rank) for slot in range(plan.num_slots) for rank in range(plan.world)} - {None}

    assert (0, 1, 0) not in produced
    assert clean_history_cells(schedule, num_blocks=1) == {(0, 0, 0), (0, 1, 0)}
