# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-Step history retention on top of the session-bound Noisy KV path.

Same-Step consumers read the version matching their own denoise step, while the
diagonal orderings read the clean version of each retained chunk. Retaining only
the clean cells therefore leaves a Same-Step continuation with nothing to read,
which is what these cases pin.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import (
    CellSchedule,
    Ordering,
    build_cell_plan,
    expected_prior_cells,
    rank_of_block,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
    clean_history_cells,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DENOISE = 3
HISTORY = 2


def _schedule(ordering: Ordering, *, step_replicas: int, chunk_base: int = 5, chunks: int = 3) -> CellSchedule:
    return CellSchedule(
        chunks=chunks,
        num_denoise_steps=DENOISE,
        step_replicas=step_replicas,
        layer_groups=1,
        ordering=ordering,
        kv_history_chunks=HISTORY,
        blocks=1,
        blocks_per_rank=1,
        chunk_base=chunk_base,
    )


def test_same_step_retains_every_denoise_step_in_the_window():
    """A Same-Step consumer reads the matching step, so each step must survive."""
    schedule = _schedule(Ordering.SAME_STEP, step_replicas=DENOISE)

    cells = clean_history_cells(schedule, num_blocks=1)

    for chunk in range(schedule.chunk_end - HISTORY, schedule.chunk_end):
        for step in range(DENOISE):
            assert (chunk, step, 0) in cells, f"same-step history lost (chunk={chunk}, step={step})"
    # Chunk 0's clean sink stays for the life of the session.
    assert (0, DENOISE, 0) in cells


def test_diagonal_retains_only_the_clean_version():
    """Self-Forcing/Latest-KV read clean KV, so one version per chunk is enough."""
    schedule = _schedule(Ordering.SELF_FORCING, step_replicas=1)

    cells = clean_history_cells(schedule, num_blocks=1)

    assert cells == {(chunk, DENOISE, 0) for chunk in range(schedule.chunk_end - HISTORY, schedule.chunk_end)}


@pytest.mark.parametrize("denoise", [2, 3, 4])
def test_same_step_retention_per_rank_does_not_grow_with_d(denoise):
    """Every step of the window survives, but each rank owns one step of it.

    The pool is per rank, so a rank keeps ``H`` versions plus, on the sink owner,
    chunk 0's sink -- the parked share of ``2H + 1``, independent of ``D``.
    """
    schedule = CellSchedule(
        chunks=7,
        num_denoise_steps=denoise,
        step_replicas=denoise,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=HISTORY,
        blocks=1,
        blocks_per_rank=1,
    )
    kept = clean_history_cells(schedule, num_blocks=1)

    per_rank = [sum(1 for cell in kept if rank_of_block(cell[1], cell[2], schedule) == rank) for rank in range(denoise)]

    assert len(kept) == HISTORY * denoise + 1
    assert max(per_rank) == HISTORY + 1


def _one_step_spec(history_versions: int | None = None) -> ARDiffusionNoisyKVSpec:
    return ARDiffusionNoisyKVSpec(
        num_layers=1,
        num_kv_heads=1,
        head_size=1,
        block_size=4,
        max_chunk_tokens=4,
        max_history_chunks=HISTORY,
        max_history_versions=history_versions,
    )


def _one_step_schedule(chunk_base: int, chunks: int) -> CellSchedule:
    # Same-Step needs step_replicas == D, so keep D=1 for a single-rank executor.
    return CellSchedule(
        chunks=chunks,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=HISTORY,
        blocks=1,
        blocks_per_rank=1,
        chunk_base=chunk_base,
    )


@pytest.mark.parametrize(("history_versions", "fits"), [(None, False), (2 * HISTORY + 1, True)])
def test_pool_holds_a_parked_session_next_to_a_live_request(history_versions, fits):
    """A retained Same-Step session stays resident while another request runs.

    Per rank that is its own step of the window plus the sink, next to the live
    request's history, the sink and the write: the live-window default is too
    small, and ``2H + 1`` history versions is enough.
    """
    cache = NoisyKVCache(
        _one_step_spec(history_versions),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=1,
    )
    _run(cache, "parked", build_cell_plan(_one_step_schedule(0, 4)))

    if fits:
        _run(cache, "live", build_cell_plan(_one_step_schedule(0, 4)))
    else:
        with pytest.raises(RuntimeError, match="VersionPool exhausted"):
            _run(cache, "live", build_cell_plan(_one_step_schedule(0, 4)))


class _SeqRecorder(CellAdapter):
    """Records the KV view width each cell sees."""

    def __init__(self) -> None:
        self.seq_lens: dict[tuple, int] = {}

    def forward(self, tasks, kv_contexts, *, hidden):
        del hidden
        for (_req, cell), ctx_list in zip(tasks, kv_contexts):
            view = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            self.seq_lens[cell] = int(view.seq_len)
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def _run(cache: NoisyKVCache, kv_id: str, plan) -> _SeqRecorder:
    # The runner builds one NoisyKVState per call; only the pool persists.
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=NoisyKVState(cache),
    )
    retain = clean_history_cells(plan.schedule, num_blocks=1)
    ctx.enqueue(kv_id, plan, chunk_tokens=4, retain=retain)
    adapter = _SeqRecorder()
    run_cell_pipeline(ctx=ctx, adapter=adapter)
    ctx.kv.end_request(kv_id, retain=retain)
    return adapter


def test_same_step_produces_the_clean_step_only_for_chunk_zero():
    """The clean window a diagonal ordering keeps does not exist under Same-Step.

    Chunk 0 runs one extra clean forward (the session sink); later chunks stop at
    the last noisy step. Demanding their clean cells as prior would fail every
    continuation closed, so the prior contract has to follow the ordering.
    """
    schedule = _schedule(Ordering.SAME_STEP, step_replicas=DENOISE, chunk_base=0, chunks=4)

    plan = build_cell_plan(schedule)
    produced = {plan.task(slot, rank) for slot in range(plan.num_slots) for rank in range(plan.world)} - {None}

    assert (0, DENOISE, 0) in produced
    assert (1, DENOISE, 0) not in produced
    assert (3, DENOISE, 0) not in produced

    cont = _schedule(Ordering.SAME_STEP, step_replicas=DENOISE, chunk_base=4, chunks=2)
    demanded = expected_prior_cells(cont)
    assert demanded, "a continuation must still demand its history"
    assert all(step < DENOISE or chunk == 0 for chunk, step, _ in demanded), sorted(demanded)


def test_prior_contract_matches_what_the_session_retains():
    """What the next request demands must be what the previous request kept.

    This is the invariant that ties ``expected_prior_cells`` to
    ``clean_history_cells``: if the two drift, a continuation either fails the
    residency check or silently reads nothing.
    """
    for ordering, replicas in ((Ordering.SAME_STEP, DENOISE), (Ordering.LATEST_KV, DENOISE + 1)):
        first = _schedule(ordering, step_replicas=replicas, chunk_base=0, chunks=7)
        cont = _schedule(ordering, step_replicas=replicas, chunk_base=7, chunks=3)

        kept = clean_history_cells(first, num_blocks=1)
        needed = expected_prior_cells(cont)

        assert needed <= kept, f"{ordering.name}: continuation demands released cells {sorted(needed - kept)}"


def test_same_step_continuation_reads_the_matching_step_of_retained_history():
    """The continuation's plan reads the per-step history, not just the sink."""
    cont = _schedule(Ordering.SAME_STEP, step_replicas=DENOISE, chunk_base=7, chunks=3)
    plan = build_cell_plan(cont, prior_history=expected_prior_cells(cont))

    reads = {
        src.version
        for slot in range(plan.num_slots)
        for rank in range(plan.world)
        for task in (plan.task(slot, rank),)
        if task is not None
        for src in plan.sources(task, rank)
    }

    # Matching step of the trailing window, and chunk 0's clean sink.
    assert (6, 0, 0) in reads
    assert (6, DENOISE, 0) not in reads
    assert (0, DENOISE, 0) in reads


@pytest.mark.parametrize("cont_chunks", [3, 1])
def test_same_step_continuation_runs_against_retained_history(cont_chunks):
    """A continuation planned from the prior contract passes the executor.

    ``cont_chunks=1`` reads the sink too, so its KV view must leave room for it
    even though a cold one-chunk request has no sink.
    """
    cache = NoisyKVCache(
        _one_step_spec(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=1,
    )

    first = _one_step_schedule(chunk_base=0, chunks=4)
    cont = _one_step_schedule(chunk_base=4, chunks=cont_chunks)

    first_adapter = _run(cache, "s1", build_cell_plan(first))
    # The continuation carries its own chunk_base and the prior the session kept;
    # a clean-only prior contract would raise here instead.
    cont_adapter = _run(cache, "s1", build_cell_plan(cont, prior_history=expected_prior_cells(cont)))

    # The continuation reads retained history rather than starting cold: its later
    # cells see the window filled to H + 2 chunks (history + sink + own write).
    assert max(first_adapter.seq_lens.values()) <= (HISTORY + 2) * 4
    assert max(cont_adapter.seq_lens.values()) == (HISTORY + 2) * 4
