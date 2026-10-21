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
    return CellSchedule(
        chunks=chunks,
        num_denoise_steps=steps,
        stages=1,
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
        stages=1,
        layer_groups=4,
        ordering=Ordering.LATEST_KV,
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
@pytest.mark.parametrize(
    "ordering",
    [Ordering.SELF_FORCING, Ordering.LATEST_KV, Ordering.SAME_STEP],
)
def test_s1_builds_without_idle_slot(world, chunks, ordering):
    plan = build_cell_plan(_s1(chunks, world, ordering))
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


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_s1_latest_kv_uses_latest_completed(world, chunks):
    plan = build_cell_plan(_s1(chunks, world, Ordering.LATEST_KV, kv=True))
    for t in range(plan.num_slots):
        for rank in range(world):
            task = plan.task(t, rank)
            if task is None:
                continue
            for src in plan.sources(task, rank):
                assert plan.completion_tick(src.version) < t
                assert src.owner == rank
                assert src.version[2] == task[2]


def test_s1_kv_transfers_empty():
    plan = build_cell_plan(_s1(4, 2, Ordering.LATEST_KV, kv=True))
    for t in range(plan.num_slots):
        assert plan.transfers(t) == ()


def test_vertical_diagonal_example():
    schedule = CellSchedule(
        chunks=4,
        num_denoise_steps=2,
        stages=3,
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


def test_vertical_same_step_matches_topology_not_latest_kv():
    """Same-Step shares diagonal placement with Latest-KV but only reads matching local step."""
    common = dict(chunks=4, num_denoise_steps=2, stages=3, layer_groups=2, kv_history_chunks=6)
    latest = build_cell_plan(CellSchedule(**common, ordering=Ordering.LATEST_KV))
    same = build_cell_plan(CellSchedule(**common, ordering=Ordering.SAME_STEP))
    assert same.num_slots == latest.num_slots
    for t in range(same.num_slots):
        for rank in range(same.world):
            assert same.task(t, rank) == latest.task(t, rank)
    # Latest promotes chunk-0 to step 1; Same-Step stays on step 0.
    latest_versions = {src.version for src in latest.sources((3, 0, 0), 0)}
    same_versions = {src.version for src in same.sources((3, 0, 0), 0)}
    assert (0, 1, 0) in latest_versions
    assert same_versions == {(2, 0, 0), (1, 0, 0), (0, 0, 0)}
    assert all(v[1] == 0 for v in same_versions)
    # Clean stage still matches step (no Latest promotion across steps).
    clean_srcs = same.sources((3, 2, 0), 4)
    assert {s.version for s in clean_srcs} == {(2, 2, 0), (1, 2, 0), (0, 2, 0)}
    # Same step × block is co-located → no KV P2P under Same-Step.
    for t in range(same.num_slots):
        assert same.transfers(t) == ()
    for t in range(same.num_slots):
        for rank in range(same.world):
            task = same.task(t, rank)
            if task is None:
                continue
            for src in same.sources(task, rank):
                assert src.owner == rank
                assert src.version[1] == task[1]
                assert same.wait_ready(t, rank) == frozenset()


def test_vertical_serial_same_topology_no_chunk_overlap():
    common = dict(chunks=3, num_denoise_steps=2, stages=3, layer_groups=1, kv_history_chunks=6)
    serial = build_cell_plan(CellSchedule(**common, ordering=Ordering.SELF_FORCING))
    latest = build_cell_plan(CellSchedule(**common, ordering=Ordering.LATEST_KV))
    same = build_cell_plan(CellSchedule(**common, ordering=Ordering.SAME_STEP))
    world = 3
    assert serial.num_slots == 3 * world
    assert latest.num_slots == same.num_slots == 3 + world - 1
    for t in range(serial.num_slots):
        chunks = {task[0] for task in (serial.task(t, r) for r in range(world)) if task is not None}
        assert len(chunks) <= 1
    assert serial.task(world, 0) == (1, 0, 0)
    assert serial.completion_tick((0, 2, 0)) == world - 1
    for src in serial.sources((1, 0, 0), 0):
        assert src.version[1] == 2


def test_block_expand_multiplies_ticks():
    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=1,
        stages=2,
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
    with pytest.raises(ValueError, match="stages must be 1"):
        CellSchedule(
            chunks=4,
            num_denoise_steps=4,
            stages=2,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=4,
        )


def test_i2_sources_strictly_earlier():
    plan = build_cell_plan(
        CellSchedule(
            chunks=4,
            num_denoise_steps=2,
            stages=3,
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
            stages=5,
            layer_groups=1,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=6,
            blocks=30,
            blocks_per_rank=30,
        )
    )
    wait_entries = sum(len(plan.wait_ready(slot, rank)) for slot in range(plan.num_slots) for rank in range(plan.world))
    assert wait_entries > 0
