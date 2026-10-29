# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Integer geometry scheduling, shape ownership, bounded memory and tails."""

import pytest

from vllm_omni.diffusion.models.minimax_h3.vae_schedule import TileTask, allocate_tiles, plan_temporal_schedule

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def clips(spatial=12, count=7, batch=1, tail=False, edges=False):
    return tuple(
        tuple(
            TileTask(
                i,
                j,
                batch,
                (batch, 24, 3 if tail and i == count - 1 else 7, 16, 8 if edges and j == spatial - 1 else 16),
                (i * 5, i * 5 + 7, int(i == 0), int(i == count - 1)),
                (0, j * 192),
            )
            for j in range(spatial)
        )
        for i in range(count)
    )


def plan(data, ranks, budget=2**40):
    return plan_temporal_schedule(
        data, ranks, budget_bytes=budget, output_bytes_per_latent_pixel=256, input_element_bytes=4
    )


@pytest.mark.parametrize("ranks", [1, 2, 3, 4, 6, 7, 8, 16])
@pytest.mark.parametrize("spatial", [1, 6, 12, 28])
def test_replicated_deterministic_plan(ranks, spatial):
    data = clips(spatial=spatial, tail=True, edges=True)
    result = plan(data, ranks)
    assert 1 <= result.window <= min(16, len(data))
    assert result == plan(data, ranks)
    for start in range(0, len(data), result.window):
        tasks = sum(data[start : start + result.window], ())
        buckets = allocate_tiles(tasks, ranks)
        assert buckets == allocate_tiles(tasks, ranks)
        assert {i for b in buckets for share in b.ranks for i in share} == set(range(len(tasks)))
        for bucket in buckets:
            assert all(bucket.ranks)
            assert all(tasks[i].shape == bucket.shape for share in bucket.ranks for i in share)
            assert all(
                tasks[i].context == data[tasks[i].clip][tasks[i].spatial].context
                for share in bucket.ranks
                for i in share
            )


def test_divisible_uniform_prefers_smallest_window():
    assert plan(clips(spatial=12), 4).window == 1


@pytest.mark.parametrize("spatial", [12, 28])
def test_eight_rank_uniform_layouts_choose_two_without_timing_model(spatial):
    result = plan(clips(spatial=spatial, count=7), 8)
    assert result.window == 2 and result.balancing_period == 2
    native, pooled = result.candidates[:2]
    assert pooled.waste_units < native.waste_units
    assert sum(pooled.rank_tasks) == spatial * 7
    assert pooled.extra_bytes > 0
    assert result.reason == "minimum_assignment_waste"


def test_fewer_tiles_than_ranks_can_schedule_without_changing_batch():
    result = plan(clips(spatial=1, count=16, batch=2), 16)
    assert result.window == 16
    assert all(b.shape[0] == 2 for b in allocate_tiles(sum(clips(1, 16, 2), ()), 16))


def test_zero_incremental_memory_falls_back():
    result = plan(clips(), 8, budget=0)
    assert result.window == 1 and result.reason == "incremental_memory_budget"
    assert all(c.reason == "incremental_memory_budget" for c in result.candidates[1:])


def test_limited_memory_uses_feasible_small_window():
    data = clips(spatial=3, count=16)
    unconstrained = plan(data, 8)
    three = unconstrained.candidates[2]
    result = plan(data, 8, budget=three.extra_bytes)
    assert 1 < result.window < unconstrained.window
    assert next(c for c in result.candidates if c.window == result.window).extra_bytes <= three.extra_bytes


def test_insufficient_remaining_clips_is_not_unconditional_fallback():
    result = plan(clips(spatial=5, count=2), 8)
    assert result.window == 2 and result.balancing_period == 8


def test_bound_and_incomplete_tail_are_included():
    data = clips(spatial=5, count=19, tail=True, edges=True)
    result = plan(data, 7)
    assert len(result.candidates) == 16
    for candidate in result.candidates:
        assert candidate.waste_units >= 0
        assert candidate.extra_bytes >= 0
        assert sum(candidate.rank_tasks) >= 19 * 5
    assert result == plan(data, 7)


def test_different_shapes_are_not_counted_as_one_divisible_bucket():
    # Total S=8 is divisible, but seven full tiles and one edge tile form
    # separate compatible buckets. Pooling can reduce replicated edge work.
    result = plan(clips(spatial=8, count=8, edges=True), 8)
    assert result.window > 1
    for chunk_start in range(0, 8, result.window):
        tasks = sum(clips(8, 8, edges=True)[chunk_start : chunk_start + result.window], ())
        assert len(allocate_tiles(tasks, 8)) == 2


@pytest.mark.parametrize("ranks", [0, -1, True, 2.0])
def test_invalid_group_is_not_fallback(ranks):
    with pytest.raises(ValueError, match="group size"):
        plan(clips(), ranks)


def test_invalid_shape_is_not_fallback():
    bad = TileTask(0, 0, 1, (2, 24, 7, 16, 16), (0, 7, 0, 0), (0, 0))
    with pytest.raises(ValueError, match="shape"):
        plan(((bad,),), 1)


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_memory_budget_is_not_fallback(budget):
    with pytest.raises(ValueError, match="memory budget"):
        plan(clips(), 8, budget=budget)
