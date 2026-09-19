# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the SeedVR2 window planner and Plan A routing.

These tests do not need a GPU, a process group or a checkpoint: they exercise
the geometry, the deterministic planner and the exact routing contract with
integer token ids and tiny feature rows.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2.window_geometry import (
    DEFAULT_WINDOW,
    geometry_fingerprint,
    make_720p_shifted_windows,
    make_720p_windows,
)
from vllm_omni.diffusion.models.seedvr2.window_sp import (
    WindowPlanCache,
    build_rank_window_plan,
    build_redistribution_plans,
    build_window_assignment,
    build_window_layout,
    joint_cu_seqlens,
    redistribute_window_rows,
    schedule_methods,
    transition_count,
)

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
    pytest.mark.sp,
    pytest.mark.core_model,
    pytest.mark.cpu,
]

REGULAR = "720pwin_by_size_bysize"
SHIFTED = "720pswin_by_size_bysize"

# Small token grids that still exercise ragged edges in every dimension.
# The last few are large enough (h, w > 20) that the 720p-normalised window size
# is smaller than the extent, so H and W also get multiple ragged windows.
TOKEN_GRIDS = [
    (4, 3, 3),
    (9, 6, 5),
    (12, 8, 8),
    (7, 5, 4),
    (17, 9, 7),
    (2, 2, 2),
    (1, 3, 3),
    (31, 7, 6),
    (8, 24, 24),
    (12, 22, 21),
    (33, 25, 23),
]

#: Grids where the regular and shifted layouts genuinely differ.
MULTI_WINDOW_GRIDS = [(8, 24, 24), (12, 22, 21), (33, 25, 23)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def reference_window_token_ids(size, method, window=DEFAULT_WINDOW):
    """Independent re-implementation of the reference windowing (index form).

    Mirrors ``na.window_idx``: take ``arange`` of the flattened sequence, view it
    as ``(t, h, w)`` and slice it with the window functions.
    """
    t, h, w = size
    if method == REGULAR:
        slices = make_720p_windows(size, window)
    else:
        slices = make_720p_shifted_windows(size, window)

    flat = torch.arange(t * h * w, dtype=torch.int64).reshape(t, h, w)
    ids = []
    shapes = []
    for st, sh, sw in slices:
        win = flat[st, sh, sw]
        ids.append(win.reshape(-1))
        shapes.append(tuple(win.shape))
    return torch.cat(ids), shapes


def transport_send(local_rows, plan, world_size, received):
    """Push one rank's send buffer into the receivers' inboxes."""
    send = local_rows[plan.send_indices]
    cursor = 0
    for dst, count in enumerate(plan.input_split_sizes):
        received[dst][plan.rank] = send[cursor : cursor + count]
        cursor += count
    assert cursor == send.shape[0]


def transport_recv(received, rank, plan, world_size, num_features):
    """Concatenate one rank's inbox by source rank and apply the receive order."""
    segments = [received[rank][src] for src in range(world_size)]
    recv = torch.cat(segments) if segments else torch.empty(0, num_features, dtype=torch.int64)
    return recv[plan.recv_to_dst_indices]


def transport_simulation(src_ids, dst_ids, plans, num_features=3):
    """Run the Plan A routing on CPU for every rank and return destination rows.

    ``src_ids`` / ``dst_ids`` are per-rank canonical token id tensors.  The
    simulation uses one unique integer row per canonical token so a wrong route
    cannot be hidden by a sum or a shape coincidence.
    """
    world_size = len(src_ids)
    num_tokens = int(sum(len(z) for z in src_ids))
    global_rows = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(1).repeat(1, num_features)

    received = [dict() for _ in range(world_size)]
    for rank, plan in enumerate(plans):
        assert plan.rank == rank
        transport_send(global_rows[src_ids[rank]], plan, world_size, received)

    return [transport_recv(received, rank, plans[rank], world_size, num_features) for rank in range(world_size)]


def make_layouts(size, world_size, cache=None):
    cache = cache or WindowPlanCache()
    src_layout = cache.layout(size, REGULAR)
    dst_layout = cache.layout(size, SHIFTED)
    src_assignment = cache.assignment(src_layout, world_size)
    dst_assignment = cache.assignment(dst_layout, world_size)
    plans = cache.route(src_layout, src_assignment, dst_layout, dst_assignment)
    return src_layout, src_assignment, dst_layout, dst_assignment, plans


# ---------------------------------------------------------------------------
# geometry parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("method", [REGULAR, SHIFTED])
def test_geometry_matches_reference_membership(size, method):
    layout = build_window_layout(size, method)
    ref_ids, ref_shapes = reference_window_token_ids(size, method)
    assert torch.equal(layout.window_token_ids, ref_ids), f"membership/order mismatch for {size} {method}"
    assert layout.window_shapes.tolist() == [list(s) for s in ref_shapes]


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("method", [REGULAR, SHIFTED])
def test_layout_is_exact_partition(size, method):
    layout = build_window_layout(size, method)
    t, h, w = size
    assert layout.num_tokens == t * h * w
    assert torch.equal(torch.sort(layout.window_token_ids).values, torch.arange(t * h * w, dtype=torch.int64)), (
        "layout must cover every token exactly once"
    )
    lengths = layout.window_lengths()
    assert bool((lengths > 0).all())
    assert int(lengths.sum()) == layout.num_tokens
    assert int(layout.window_offsets[-1]) == layout.num_tokens
    assert int(layout.window_offsets[0]) == 0


def test_shifted_is_not_cyclic_roll():
    """The shifted layout clips at the boundary instead of wrapping around.

    With an active shift the first shifted window is the *half* window clipped at
    the origin, in every dimension that has more than one window.  A cyclic roll
    would instead move the tail of the clip to the front and would keep the
    window boundaries unchanged.
    """
    size = (8, 24, 24)  # wt = 2 < t and wh = ww = 20 < h = w, so all shifts are active
    shifted = build_window_layout(size, SHIFTED)
    regular = build_window_layout(size, REGULAR)

    assert shifted.window_shapes[0].tolist() == [1, 10, 10], "first shifted window must be the clipped half window"
    first_window = shifted.window_token_ids[: int(shifted.window_offsets[1])]
    assert int(first_window[0]) == 0, "the clipped layout starts at canonical token 0, not at a wrapped tail"
    assert bool((first_window[1:] > first_window[:-1]).all())
    assert not torch.equal(shifted.window_offsets, regular.window_offsets)
    assert not torch.equal(shifted.window_shapes, regular.window_shapes)

    # A cyclic roll of the canonical order starts the packed order at a wrapped
    # token and keeps the same window sizes, so it matches neither property.
    rolled = torch.roll(torch.arange(regular.num_tokens, dtype=torch.int64), shifts=7)
    assert not torch.equal(shifted.window_token_ids, rolled)


def test_shifted_collapses_to_regular_when_shift_is_sub_token():
    """Reference behaviour: with ``wt == 1`` the half-window shift truncates away."""
    size = (4, 3, 3)
    shifted = build_window_layout(size, SHIFTED)
    regular = build_window_layout(size, REGULAR)
    assert torch.equal(shifted.window_token_ids, regular.window_token_ids)


def test_grid_set_contains_real_layout_changes():
    """Guard the parameter grid: several cases must actually switch window bounds."""
    boundary_changed = 0
    for size in TOKEN_GRIDS:
        regular = build_window_layout(size, REGULAR)
        shifted = build_window_layout(size, SHIFTED)
        if not torch.equal(regular.window_offsets, shifted.window_offsets):
            boundary_changed += 1
    assert boundary_changed >= 3, f"only {boundary_changed} token grids change window boundaries"
    for size in MULTI_WINDOW_GRIDS:
        regular = build_window_layout(size, REGULAR)
        shifted = build_window_layout(size, SHIFTED)
        assert not torch.equal(regular.window_offsets, shifted.window_offsets), size
        assert not torch.equal(regular.window_token_ids, shifted.window_token_ids), size


def test_equal_packed_order_still_reports_ownership_changes():
    """Equal token order does **not** imply the rows stay on the same rank.

    ``(12, 8, 8)`` keeps one window along H and W, so both layouts pack the tokens
    in the same order while their window boundaries -- and therefore the
    token-balanced assignment -- differ.  A planner that skipped redistribution
    whenever the packed order matches would silently produce wrong results here.
    """
    size = (12, 8, 8)
    regular = build_window_layout(size, REGULAR)
    shifted = build_window_layout(size, SHIFTED)
    assert torch.equal(regular.window_token_ids, shifted.window_token_ids)
    assert not torch.equal(regular.window_offsets, shifted.window_offsets)

    plans = build_redistribution_plans(
        regular, build_window_assignment(regular, 2), shifted, build_window_assignment(shifted, 2)
    )
    assert any(plan.network_exchange_required for plan in plans)
    # ... and the rows that do move must still arrive correctly.
    regular_ids = [
        regular.window_token_ids_of(build_window_assignment(regular, 2).rank_window_ids[r]) for r in range(2)
    ]
    shifted_ids = [
        shifted.window_token_ids_of(build_window_assignment(shifted, 2).rank_window_ids[r]) for r in range(2)
    ]
    out = transport_simulation(regular_ids, shifted_ids, plans)
    global_rows = torch.arange(regular.num_tokens, dtype=torch.int64).unsqueeze(1).repeat(1, 3)
    for rank in range(2):
        assert torch.equal(out[rank], global_rows[shifted_ids[rank]])


def test_frame_split_fixture_needs_only_a_local_permutation():
    """When each rank keeps all of its tokens, the transition is a local reorder.

    ``(2, 32, 32)`` with SP=2 balances along the frame axis, so both layouts hand
    every rank the same token set and only the packed order changes -- the fast
    path that must not issue an all-to-all.
    """
    size = (2, 32, 32)
    regular = build_window_layout(size, REGULAR)
    shifted = build_window_layout(size, SHIFTED)
    plans = build_redistribution_plans(
        regular, build_window_assignment(regular, 2), shifted, build_window_assignment(shifted, 2)
    )
    assert all(plan.network_exchange_required is False for plan in plans)
    for plan in plans:
        assert plan.input_split_sizes[plan.rank] == plan.num_send_rows
        assert plan.recv_to_dst_indices.numel() == plan.num_recv_rows


@pytest.mark.parametrize("size", TOKEN_GRIDS)
def test_geometry_fingerprint_deterministic(size):
    a = geometry_fingerprint(size, REGULAR, DEFAULT_WINDOW)
    b = geometry_fingerprint(size, REGULAR, DEFAULT_WINDOW)
    assert a == b and len(a) == 16
    assert geometry_fingerprint(size, SHIFTED, DEFAULT_WINDOW) != a


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("method", [REGULAR, SHIFTED])
def test_planner_deterministic_and_covers(size, world_size, method):
    layout = build_window_layout(size, method)
    first = build_window_assignment(layout, world_size)
    second = build_window_assignment(layout, world_size)
    assert all(torch.equal(a, b) for a, b in zip(first.rank_window_ids, second.rank_window_ids))

    owners = first.owner_rank
    assert owners.numel() == layout.num_windows
    assert bool((owners >= 0).all()) and bool((owners < world_size).all())
    # rank windows partition the window ids, and tokens follow.
    all_windows = torch.cat([ids for ids in first.rank_window_ids if ids.numel()])
    assert torch.equal(torch.sort(all_windows).values, torch.arange(layout.num_windows, dtype=torch.int64))
    all_tokens = torch.cat([layout.window_token_ids_of(ids) for ids in first.rank_window_ids if ids.numel()])
    assert torch.equal(torch.sort(all_tokens).values, torch.arange(layout.num_tokens, dtype=torch.int64))
    # windows are listed in ascending id order inside a rank
    for ids in first.rank_window_ids:
        if ids.numel() > 1:
            assert bool((ids[1:] > ids[:-1]).all())


def naive_lpt_oracle(lengths, world_size):
    loads = [0] * world_size
    buckets = [[] for _ in range(world_size)]
    for w in sorted(range(len(lengths)), key=lambda w: (-lengths[w], w)):
        r = min(range(world_size), key=lambda r: (loads[r], r))
        buckets[r].append(w)
        loads[r] += lengths[w]
    return [sorted(b) for b in buckets], loads


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [2, 4])
def test_planner_matches_naive_lpt(size, world_size):
    layout = build_window_layout(size, REGULAR)
    assignment = build_window_assignment(layout, world_size)
    buckets, loads = naive_lpt_oracle([int(n) for n in layout.window_lengths().tolist()], world_size)
    assert [ids.tolist() for ids in assignment.rank_window_ids] == buckets
    assert list(assignment.rank_token_counts) == loads


def test_planner_handles_equal_windows_and_single_window():
    equal = build_window_layout((8, 4, 4), REGULAR)
    assignment = build_window_assignment(equal, 4)
    counts = assignment.rank_token_counts
    assert max(counts) - min(counts) <= 1

    single = build_window_layout((1, 2, 2), REGULAR)
    assert single.num_windows == 1
    one = build_window_assignment(single, 4)
    assert one.rank_token_counts == (single.num_tokens, 0, 0, 0)
    empty_plan = build_rank_window_plan(single, one, rank=3)
    assert empty_plan.num_local_tokens == 0 and empty_plan.num_local_windows == 0
    assert empty_plan.video_cu_seqlens.tolist() == [0]


def test_planner_rejects_bad_inputs():
    layout = build_window_layout((4, 3, 3), REGULAR)
    with pytest.raises(ValueError):
        build_window_assignment(layout, 0)
    with pytest.raises(ValueError):
        build_rank_window_plan(layout, build_window_assignment(layout, 2), rank=5)


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_assignment_metrics(size, world_size):
    layout = build_window_layout(size, REGULAR)
    assignment = build_window_assignment(layout, world_size)
    assert assignment.load_imbalance() >= 1.0 - 1e-9
    assert assignment.ideal_lower_bound >= math.ceil(layout.num_tokens / world_size) - 1
    assert assignment.max_rank_tokens >= assignment.ideal_lower_bound or assignment.max_window_tokens > 0


# ---------------------------------------------------------------------------
# rank-local metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("method", [REGULAR, SHIFTED])
def test_rank_plan_metadata(size, world_size, method):
    layout = build_window_layout(size, method)
    assignment = build_window_assignment(layout, world_size)
    per_rank = [build_rank_window_plan(layout, assignment, r) for r in range(world_size)]

    for rank, plan in enumerate(per_rank):
        assert plan.video_cu_seqlens.dtype == torch.int32
        assert int(plan.video_cu_seqlens[-1]) == plan.num_local_tokens
        assert int(plan.video_cu_seqlens[0]) == 0
        lengths = (plan.video_cu_seqlens[1:] - plan.video_cu_seqlens[:-1]).to(torch.int64)
        expected = layout.window_lengths()[assignment.rank_window_ids[rank]] if plan.num_local_windows else None
        if expected is not None:
            assert torch.equal(lengths, expected)
        assert plan.window_shapes.shape[0] == plan.num_local_windows

    # every canonical token appears exactly once across ranks, in rank order
    concatenated = torch.cat([p.global_token_ids for p in per_rank if p.num_local_tokens])
    assert torch.equal(torch.sort(concatenated).values, torch.arange(layout.num_tokens, dtype=torch.int64))


def test_joint_cu_seqlens_adds_text_per_window():
    video = torch.tensor([0, 4, 9, 10], dtype=torch.int32)
    joint = joint_cu_seqlens(video, 7)
    assert joint.dtype == torch.int32
    assert joint.tolist() == [0, 11, 23, 31]


def test_cu_seqlens_overflow_is_rejected():
    huge = torch.tensor([0, 2**31], dtype=torch.int64)
    from vllm_omni.diffusion.models.seedvr2.window_sp import to_cu_seqlens

    with pytest.raises(ValueError, match="int32"):
        to_cu_seqlens(huge, context="test")
    with pytest.raises(ValueError, match="start at 0"):
        to_cu_seqlens(torch.tensor([1, 2], dtype=torch.int64), context="test")
    with pytest.raises(ValueError, match="non-decreasing"):
        to_cu_seqlens(torch.tensor([0, 5, 3], dtype=torch.int64), context="test")


# ---------------------------------------------------------------------------
# routing: A -> B, B -> A, A -> B -> A
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_route_regular_to_shifted_is_exact(size, world_size):
    src_layout, src_assignment, dst_layout, dst_assignment, plans = make_layouts(size, world_size)
    src_ids = [src_layout.window_token_ids_of(src_assignment.rank_window_ids[r]) for r in range(world_size)]
    dst_ids = [dst_layout.window_token_ids_of(dst_assignment.rank_window_ids[r]) for r in range(world_size)]

    out = transport_simulation(src_ids, dst_ids, plans)
    for rank in range(world_size):
        expected = (
            torch.arange(sum(len(z) for z in src_ids), dtype=torch.int64)[dst_ids[rank]].unsqueeze(1).repeat(1, 3)
        )
        assert torch.equal(out[rank], expected), f"rank {rank} received the wrong rows for {size} SP={world_size}"


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_route_shifted_to_regular_is_built_independently(size, world_size):
    """B -> A is built from its own membership, not derived from A -> B."""
    cache = WindowPlanCache()
    regular = cache.layout(size, REGULAR)
    shifted = cache.layout(size, SHIFTED)
    regular_assignment = cache.assignment(regular, world_size)
    shifted_assignment = cache.assignment(shifted, world_size)

    forward = cache.route(regular, regular_assignment, shifted, shifted_assignment)
    backward = cache.route(shifted, shifted_assignment, regular, regular_assignment)
    assert forward[0].src_key == regular.key and forward[0].dst_key == shifted.key
    assert backward[0].src_key == shifted.key and backward[0].dst_key == regular.key

    src_ids = [shifted.window_token_ids_of(shifted_assignment.rank_window_ids[r]) for r in range(world_size)]
    dst_ids = [regular.window_token_ids_of(regular_assignment.rank_window_ids[r]) for r in range(world_size)]
    out = transport_simulation(src_ids, dst_ids, backward)
    num_tokens = regular.num_tokens
    for rank in range(world_size):
        expected = torch.arange(num_tokens, dtype=torch.int64)[dst_ids[rank]].unsqueeze(1).repeat(1, 3)
        assert torch.equal(out[rank], expected)


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_route_round_trip_restores_bitwise(size, world_size):
    cache = WindowPlanCache()
    regular = cache.layout(size, REGULAR)
    shifted = cache.layout(size, SHIFTED)
    ra = cache.assignment(regular, world_size)
    sa = cache.assignment(shifted, world_size)
    forward = cache.route(regular, ra, shifted, sa)
    backward = cache.route(shifted, sa, regular, ra)

    regular_ids = [regular.window_token_ids_of(ra.rank_window_ids[r]) for r in range(world_size)]
    shifted_ids = [shifted.window_token_ids_of(sa.rank_window_ids[r]) for r in range(world_size)]

    after_forward = transport_simulation(regular_ids, shifted_ids, forward)
    # feed the forward results back through B -> A
    global_rows = torch.arange(regular.num_tokens, dtype=torch.int64).unsqueeze(1).repeat(1, 3)
    received = [dict() for _ in range(world_size)]
    for rank, plan in enumerate(backward):
        transport_send(after_forward[rank], plan, world_size, received)

    for rank in range(world_size):
        back = transport_recv(received, rank, backward[rank], world_size, 3)
        assert torch.equal(back, global_rows[regular_ids[rank]])


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_split_sizes_are_reciprocal(size, world_size):
    _, _, _, _, plans = make_layouts(size, world_size)
    for r, plan in enumerate(plans):
        for d in range(world_size):
            assert plan.input_split_sizes[d] == plans[d].output_split_sizes[r]


@pytest.mark.parametrize("size", TOKEN_GRIDS)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_network_flag_is_group_consistent(size, world_size):
    _, _, _, _, plans = make_layouts(size, world_size)
    flags = {p.network_exchange_required for p in plans}
    assert len(flags) == 1


def test_sp1_transition_is_local_permutation():
    size = (4, 3, 3)
    cache = WindowPlanCache()
    regular = cache.layout(size, REGULAR)
    shifted = cache.layout(size, SHIFTED)
    plans = cache.route(regular, cache.assignment(regular, 1), shifted, cache.assignment(shifted, 1))
    plan = plans[0]
    assert plan.network_exchange_required is False
    assert plan.input_split_sizes == (regular.num_tokens,)
    assert plan.output_split_sizes == (regular.num_tokens,)

    # send_indices must reorder canonical rows into the shifted packed order.
    from vllm_omni.diffusion.models.seedvr2.window_sp import materialize_redistribution_plan

    device_plan = materialize_redistribution_plan(plan, device="cpu")
    hidden = torch.arange(regular.num_tokens, dtype=torch.float32).unsqueeze(1)
    out = redistribute_window_rows(hidden, device_plan, group=None)  # type: ignore[arg-type]
    expected = torch.arange(regular.num_tokens, dtype=torch.float32)[shifted.window_token_ids].unsqueeze(1)
    assert torch.equal(out, expected)


def test_same_layout_transition_needs_no_network():
    size = (9, 6, 5)
    cache = WindowPlanCache()
    regular = cache.layout(size, REGULAR)
    assignment = cache.assignment(regular, 4)
    plans = cache.route(regular, assignment, regular, assignment)
    assert all(p.network_exchange_required is False for p in plans)
    for r, plan in enumerate(plans):
        assert plan.input_split_sizes[r] == plan.num_send_rows
        assert plan.output_split_sizes[r] == plan.num_recv_rows
        assert torch.equal(plan.send_indices, torch.arange(plan.num_send_rows, dtype=torch.int64))
        assert torch.equal(plan.recv_to_dst_indices, torch.arange(plan.num_recv_rows, dtype=torch.int64))


def test_empty_rank_participates_in_routing():
    """W < P: ranks without windows still have valid (empty) split metadata."""
    size = (1, 2, 2)  # exactly one window
    _, _, _, _, plans = make_layouts(size, 8)
    empty = [p for p in plans if p.num_send_rows == 0 and p.num_recv_rows == 0]
    assert len(empty) == 7
    for plan in empty:
        assert all(c == 0 for c in plan.input_split_sizes)
        assert all(c == 0 for c in plan.output_split_sizes)
    assert plans[0].input_split_sizes[1:] == (0,) * 7


# ---------------------------------------------------------------------------
# text reduction semantics
# ---------------------------------------------------------------------------


def test_global_window_mean_is_not_rank_mean():
    windows = {"r0": [1.0, 3.0], "r1": [9.0]}
    global_count = sum(len(v) for v in windows.values())
    correct = sum(sum(v) for v in windows.values()) / global_count
    wrong = sum(sum(v) / len(v) for v in windows.values()) / len(windows)
    assert correct == pytest.approx(4.333333333333333, rel=1e-12)
    assert wrong == pytest.approx(5.5, rel=1e-12)
    assert correct != wrong


def test_global_window_mean_formula_matches_hand_computation():
    """SP=2 with unequal local window counts: local sums then one global divide."""
    local_sums = [torch.tensor([[4.0, 10.0]]), torch.tensor([[9.0, 3.0]])]
    global_count = 3
    total = torch.zeros_like(local_sums[0])
    for s in local_sums:
        total = total + s
    result = total / global_count
    expected = torch.tensor([[(1 + 3 + 9) / 3, (4 + 6 + 3) / 3]])
    assert torch.allclose(result, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# schedule / transitions
# ---------------------------------------------------------------------------


def test_transition_count_follows_schedule_not_constant():
    methods = (REGULAR, SHIFTED)
    assert transition_count(methods, 32) == 31
    # A schedule that repeats the same layout for two layers has fewer transitions.
    custom = (REGULAR, REGULAR, SHIFTED, SHIFTED)
    assert transition_count(custom, 4) == 1
    assert schedule_methods(4, methods) == (REGULAR, SHIFTED, REGULAR, SHIFTED)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_key_includes_sp_world_size():
    cache = WindowPlanCache()
    layout = cache.layout((9, 6, 5), REGULAR)
    two = cache.assignment(layout, 2)
    four = cache.assignment(layout, 4)
    assert two.rank_token_counts != four.rank_token_counts
    assert cache.assignment(layout, 2) is two  # same key hits the cache


def test_cache_eviction_is_bounded():
    cache = WindowPlanCache(capacity=2)
    for size in [(2, 2, 2), (3, 3, 3), (4, 3, 3), (5, 3, 3)]:
        cache.layout(size, REGULAR)
    assert len(cache._layouts) == 2
    assert cache.misses == 4


def test_cache_separates_planner_versions():
    cache = WindowPlanCache()
    layout = cache.layout((9, 6, 5), REGULAR)
    a = cache.assignment(layout, 2, planner_version=1)
    b = cache.assignment(layout, 2, planner_version=2)
    assert a is not b


def test_invalid_layout_is_rejected():
    with pytest.raises(ValueError):
        build_window_layout((0, 4, 4), REGULAR)
    with pytest.raises(ValueError):
        build_window_layout((4, 4, 4), "not_a_window_method")


def test_redistribution_rejects_wrong_row_count():
    from vllm_omni.diffusion.models.seedvr2.window_sp import materialize_redistribution_plan

    _, _, _, _, plans = make_layouts((4, 3, 3), 2)
    device_plan = materialize_redistribution_plan(plans[0], device="cpu")
    bad = torch.zeros((device_plan.send_indices.numel() + 1, 4))
    with pytest.raises(ValueError, match="rows"):
        redistribute_window_rows(bad, device_plan, group=None)  # type: ignore[arg-type]
