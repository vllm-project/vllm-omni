# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for cell-level chunk timetable and Self-Forcing / Latest-KV / Same-Step plans."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.cell_schedule import (
    CellSchedule,
    Ordering,
    build_cell_plan,
    group_of_block,
    is_group_boundary,
    max_local_blocks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _s1(chunks, world, ordering, *, kv=False, steps=3) -> CellSchedule:
    # kv=True keeps a history window; self_forcing always runs the clean step.
    return CellSchedule(
        chunks=chunks,
        num_denoise_steps=steps,
        step_replicas=1,
        layer_groups=world,
        ordering=ordering,
        kv_history_chunks=6 if kv else 0,
    )


def test_layer_groups_match_vllm_pp_indices():
    """Schedule block ownership must match Wan/vLLM ``get_pp_indices`` (not floor)."""
    from vllm.distributed.utils import get_pp_indices

    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=4,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=0,
        blocks=30,
    )
    for g in range(4):
        start, end = get_pp_indices(30, g, 4)
        for block in range(start, end):
            assert group_of_block(block, schedule) == g
    # Floor split would put block 22 on group 3; get_pp_indices keeps it on 2.
    assert group_of_block(22, schedule) == 2
    assert get_pp_indices(30, 3, 4) == (23, 30)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
def test_s1_builds_without_idle_slot(world, chunks):
    plan = build_cell_plan(_s1(chunks, world, Ordering.SELF_FORCING))
    assert plan.num_slots >= chunks
    for t in range(plan.num_slots):
        assert any(plan.task(t, rank) is not None for rank in range(world))


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6])
def test_s1_serial_kv_sources_are_clean(world, chunks):
    plan = build_cell_plan(_s1(chunks, world, Ordering.SELF_FORCING, kv=True))
    for t in range(plan.num_slots):
        for rank in range(world):
            task = plan.task(t, rank)
            if task is None:
                continue
            for src in plan.sources(task, rank):
                assert src.version[1] == plan.schedule.num_denoise_steps
                assert src.version[2] == task[2]
                assert src.owner == rank


def test_s1_kv_transfers_empty():
    plan = build_cell_plan(_s1(4, 2, Ordering.SELF_FORCING, kv=True))
    for t in range(plan.num_slots):
        assert plan.transfers(t) == ()


def test_vertical_diagonal_example():
    schedule = CellSchedule(
        chunks=4,
        num_denoise_steps=2,
        step_replicas=3,
        layer_groups=2,
        ordering=Ordering.LATEST_KV,
        kv_history_chunks=6,
    )
    plan = build_cell_plan(schedule)
    # K=1 default: cell = (chunk, step, block=g)
    assert plan.task(0, 0) == (0, 0, 0)
    assert plan.task(3, 0) == (3, 0, 0)
    assert plan.task(2, 2) == (0, 1, 0)
    srcs = plan.sources((3, 0, 0), 0)
    versions = {src.version for src in srcs}
    assert (2, 0, 0) in versions
    assert (1, 0, 0) in versions
    assert (0, 1, 0) in versions
    owners = {src.version: src.owner for src in srcs}
    assert owners[(0, 1, 0)] == 2
    xfers = plan.transfers(2)
    assert any(x.version == (0, 1, 0) and x.src == 2 and x.dst == 0 for x in xfers)
    clean_srcs = plan.sources((3, 2, 0), 4)
    assert {s.version for s in clean_srcs} == {(2, 2, 0), (1, 2, 0), (0, 2, 0)}
    assert all(s.owner == 4 for s in clean_srcs)


def test_same_step_history_stays_local_except_clean_sink():
    plan = build_cell_plan(
        CellSchedule(
            chunks=4,
            num_denoise_steps=2,
            step_replicas=2,
            layer_groups=2,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=6,
        )
    )
    for t in range(plan.num_slots):
        for rank in range(plan.world):
            task = plan.task(t, rank)
            if task is None:
                continue
            for src in plan.sources(task, rank):
                assert plan.completion_tick(src.version) < t
                if src.version[0] != 0:
                    assert src.owner == rank
                    assert src.version[1] == task[1]
        assert all(x.version[0:2] == (0, 2) for x in plan.transfers(t))
    assert {s.version for s in plan.sources((3, 0, 0), 0)} == {(0, 2, 0), (1, 0, 0), (2, 0, 0)}


@pytest.mark.parametrize(("ordering", "step_replicas"), [(Ordering.LATEST_KV, 4), (Ordering.SAME_STEP, 3)])
def test_vertical_h0_is_free(ordering, step_replicas):
    """H is free: vertical topologies accept an explicit zero history window."""
    schedule = CellSchedule(
        chunks=3,
        num_denoise_steps=3,
        step_replicas=step_replicas,
        layer_groups=1,
        ordering=ordering,
        kv_history_chunks=0,
    )
    plan = build_cell_plan(schedule)
    for t in range(plan.num_slots):
        for rank in range(plan.world):
            task = plan.task(t, rank)
            if task is None:
                continue
            expected = {(0, 3, 0)} if ordering is Ordering.SAME_STEP and task[0] > 0 else set()
            assert {s.version for s in plan.sources(task, rank)} == expected


def test_schedule_steps_follow_ordering():
    """Schedule step counts follow ordering and are independent of history."""
    sf = CellSchedule(
        chunks=2,
        num_denoise_steps=3,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=6,
    )
    assert sf.num_schedule_steps == 4
    sf_no_hist = CellSchedule(
        chunks=2,
        num_denoise_steps=3,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=0,
    )
    assert sf_no_hist.num_schedule_steps == 4
    lk = CellSchedule(
        chunks=2, num_denoise_steps=3, step_replicas=4, layer_groups=1, ordering=Ordering.LATEST_KV, kv_history_chunks=6
    )
    assert lk.num_schedule_steps == 4
    same = CellSchedule(
        chunks=2, num_denoise_steps=2, step_replicas=2, layer_groups=1, ordering=Ordering.SAME_STEP, kv_history_chunks=4
    )
    assert same.num_schedule_steps == 3


@pytest.mark.parametrize("step_replicas", [1, 4])  # S=1 (D=3) and S=D+1 both rejected
def test_rejects_non_d_same_step(step_replicas):
    # Same-Step uses S=D; S=1 is valid only when D=1.
    with pytest.raises(ValueError, match=r"same_step requires step_replicas=D"):
        CellSchedule(
            chunks=4,
            num_denoise_steps=3,
            step_replicas=step_replicas,
            layer_groups=1,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=4,
        )


def test_accepts_s1_same_step_when_d_is_1():
    # D=1: S=1 coincides with S=D, the only Same-Step replica count.
    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=4,
    )
    assert schedule.num_schedule_steps == 2


def test_vertical_diagonal_slot_counts():
    latest = build_cell_plan(
        CellSchedule(
            chunks=3,
            num_denoise_steps=2,
            step_replicas=3,
            layer_groups=1,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
        )
    )
    same = build_cell_plan(
        CellSchedule(
            chunks=3,
            num_denoise_steps=2,
            step_replicas=2,
            layer_groups=1,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=6,
        )
    )
    world = 3
    assert latest.num_slots == 3 + world - 1
    assert same.num_slots == 2 + 1 + (2 + 2 - 1)


def test_block_expand_multiplies_ticks():
    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=1,
        step_replicas=2,
        layer_groups=1,
        ordering=Ordering.LATEST_KV,
        kv_history_chunks=2,
        blocks=4,
        blocks_per_rank=4,
    )
    plan = build_cell_plan(schedule)
    coarse_latest = 2 + 2 - 1
    assert max_local_blocks(schedule) == 4
    assert plan.num_slots == coarse_latest * 4
    assert plan.task(0, 0) == (0, 0, 0)
    assert plan.task(1, 0) == (0, 0, 1)
    assert plan.task(3, 0) == (0, 0, 3)
    assert is_group_boundary((0, 0, 3), schedule)
    assert not is_group_boundary((0, 0, 1), schedule)
    # Same block history only.
    srcs = plan.sources((1, 0, 2), 0)
    assert all(s.version[2] == 2 for s in srcs)


def test_rejects_intermediate_s():
    with pytest.raises(ValueError, match="step_replicas must be 1"):
        CellSchedule(
            chunks=4,
            num_denoise_steps=4,
            step_replicas=2,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=4,
        )


def test_accepts_same_step_s_eq_t():
    schedule = CellSchedule(
        chunks=4,
        num_denoise_steps=3,
        step_replicas=3,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=4,
    )
    plan = build_cell_plan(schedule)
    assert plan.schedule.step_replicas == plan.schedule.num_denoise_steps == 3
    assert plan.schedule.num_schedule_steps == 4
    assert plan.world == 3


@pytest.mark.parametrize("step_replicas", [3, 4])  # S=D and S=D+1 both rejected
def test_rejects_vertical_self_forcing(step_replicas):
    # Self-Forcing is single-replica (layer PP); vertical replicas are latest_kv/same_step.
    with pytest.raises(ValueError, match="self_forcing requires step_replicas=1"):
        CellSchedule(
            chunks=4,
            num_denoise_steps=3,
            step_replicas=step_replicas,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=4,
        )


def test_s1_self_forcing_history_defaults_clean():
    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=3,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=6,
    )
    assert schedule.num_schedule_steps == 4


@pytest.mark.parametrize("step_replicas", [1, 3])  # S=1 layer PP and S=D both rejected
def test_rejects_non_d_plus_1_latest_kv(step_replicas):
    # Latest-KV is the vertical clean-history topology (S=D+1) only.
    with pytest.raises(ValueError, match=r"latest_kv requires step_replicas=D\+1"):
        CellSchedule(
            chunks=4,
            num_denoise_steps=3,
            step_replicas=step_replicas,
            layer_groups=1,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
        )


def test_i2_sources_strictly_earlier():
    plan = build_cell_plan(
        CellSchedule(
            chunks=4,
            num_denoise_steps=2,
            step_replicas=3,
            layer_groups=2,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
        )
    )
    for t in range(plan.num_slots):
        for rank in range(plan.world):
            task = plan.task(t, rank)
            if task is None:
                continue
            for src in plan.sources(task, rank):
                assert plan.completion_tick(src.version) < t


def test_wait_ready_nonempty_after_cell_expand():
    """Remote history first-consume must produce wait_ready entries (REPORT regression)."""
    plan = build_cell_plan(
        CellSchedule(
            chunks=8,
            num_denoise_steps=4,
            step_replicas=5,
            layer_groups=1,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
            blocks=30,
            blocks_per_rank=30,
        )
    )
    wait_entries = sum(len(plan.wait_ready(slot, rank)) for slot in range(plan.num_slots) for rank in range(plan.world))
    assert wait_entries > 0


@pytest.mark.parametrize("steps,groups,blocks", [(1, 1, 1), (1, 2, 5), (2, 1, 3), (3, 2, 5)])
@pytest.mark.parametrize("history", [0, 1, 3])
@pytest.mark.parametrize("chunks", [1, 2, 8])
def test_same_step_clean_sink_plan(steps, groups, blocks, history, chunks):
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
    cells = [
        (t, rank, plan.task(t, rank))
        for t in range(plan.num_slots)
        for rank in range(plan.world)
        if plan.task(t, rank) is not None
    ]
    assert len(cells) == (chunks * steps + int(chunks > 1)) * blocks
    assert schedule.num_schedule_steps == steps + int(chunks > 1)
    clean = [(t, rank, cell) for t, rank, cell in cells if cell[1] == steps]
    assert {cell for _, _, cell in clean} == ({(0, steps, b) for b in range(blocks)} if chunks > 1 else set())
    assert all(rank // groups == steps - 1 for _, rank, _ in clean)
    if chunks > 1:
        assert min(t for t, _, cell in cells if cell[0] > 0) > max(t for t, _, _ in clean)
    transfers = [x for t in range(plan.num_slots) for x in plan.transfers(t)]
    assert len({(x.version, x.dst) for x in transfers}) == len(transfers)
    assert all(x.version[0:2] == (0, steps) for x in transfers)
    if chunks == 1:
        assert transfers == []
    for t, rank, cell in cells:
        chunk, step, block = cell
        sources = plan.sources(cell, rank)
        expected = set(range(max(0, chunk - history), chunk))
        if chunk:
            expected.add(0)
        assert {src.version[0] for src in sources} == expected
        assert len(sources) == len(expected)
        for src in sources:
            assert src.version == (src.version[0], steps if src.version[0] == 0 else step, block)
            assert plan.completion_tick(src.version) < t
            assert plan.last_use(rank)[src.version] >= t
        if chunk and rank // groups != steps - 1:
            sink = (0, steps, block)
            assert sum(x.version == sink and x.dst == rank for x in transfers) == 1
