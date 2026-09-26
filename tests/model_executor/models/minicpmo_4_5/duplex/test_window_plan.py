# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU accounting tests for the MiniCPM-o 4.5 duplex KV window planner.

Pure integer/complex math: no engine, no checkpoint, no GPU. Two claims carry
the design and each gets pinned here: a block-aligned shift leaves every retained
row in the physical slot it already occupies (so a trim moves no memory), and
rotating a cached key by ``-delta`` equals RoPE at the renumbered position (so a
trim needs no forward pass).
"""

from __future__ import annotations

import cmath

import pytest

try:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
        DuplexWindowGeometry,
        PositionReanchor,
        align_up,
        cdiv,
        plan_position_reanchor,
        reanchor_positions,
        unit_starts,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parent
    while repo_root.name and not (repo_root / "vllm_omni").is_dir():
        repo_root = repo_root.parent

    spec = importlib.util.spec_from_file_location(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan",
        repo_root / "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_plan.py",
    )
    assert spec is not None and spec.loader is not None
    _wp = importlib.util.module_from_spec(spec)
    sys.modules["vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan"] = _wp
    spec.loader.exec_module(_wp)

    DuplexWindowGeometry = _wp.DuplexWindowGeometry
    PositionReanchor = _wp.PositionReanchor
    align_up = _wp.align_up
    cdiv = _wp.cdiv
    plan_position_reanchor = _wp.plan_position_reanchor
    reanchor_positions = _wp.reanchor_positions
    unit_starts = _wp.unit_starts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Block size 4, a 10-token session context prefix (reference audio + system
# prompt), a 64-token attention budget, 4096 positions and 2 tokens of sample
# room: small enough to assert on by hand, same shape as the real stage.
BLOCK = 4


def _geometry(**overrides) -> DuplexWindowGeometry:
    kwargs: dict[str, int] = {
        "prefix_tokens": 10,
        "window_tokens": 64,
        "block_size": BLOCK,
        "max_model_len": 4096,
        "sample_room": 2,
    }
    kwargs.update(overrides)
    return DuplexWindowGeometry(**kwargs)


def test_rejects_impossible_geometry():
    with pytest.raises(ValueError, match="window"):
        _geometry(window_tokens=0)
    with pytest.raises(ValueError, match="prefix"):
        _geometry(prefix_tokens=-1)
    with pytest.raises(ValueError, match="block_size"):
        _geometry(block_size=0)
    # A trigger below the low watermark would trim the sequence it just rebuilt.
    with pytest.raises(ValueError, match="high_watermark_tokens"):
        _geometry(high_watermark_tokens=8)


def test_nothing_is_reclaimed_inside_the_window():
    geometry = _geometry(high_watermark_tokens=96)
    # The trigger is prefix + high watermark == 106; below it, no plan at all.
    assert plan_position_reanchor(geometry, computed_tokens=100, pending_tokens=6) is None
    assert plan_position_reanchor(geometry, computed_tokens=100, pending_tokens=7) is not None


def test_a_trim_lands_on_the_low_watermark():
    geometry = _geometry(high_watermark_tokens=96)
    plan = plan_position_reanchor(geometry, computed_tokens=200, pending_tokens=20)
    assert plan is not None
    # Cut so that what survives past the sink is the low watermark, and no more:
    # the tail is kept within one page of the budget so a trim never over-reclaims.
    target = geometry.prefix_tokens + geometry.window_tokens
    assert plan.delta == align_up(220 - target, BLOCK) == 148
    assert plan.moved_from == plan.sink_end + plan.delta == 160
    assert plan.delta % BLOCK == 0
    # The sink never shrinks: it ends on the first whole page after the prefix.
    assert plan.sink_end == cdiv(geometry.prefix_tokens, BLOCK) * BLOCK == 12
    retained = 220 - plan.delta
    assert target - BLOCK < retained <= target


def test_production_planner_drop_length_and_retained_tail():
    # Production parameters from reviewer @linyueqian:
    # prefix=96, block_size=16, window=6000, high=8000, computed=8100, pending=12
    geo = DuplexWindowGeometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )
    plan = plan_position_reanchor(geo, computed_tokens=8100, pending_tokens=12)
    assert plan is not None
    assert plan.sink_blocks == 6
    assert plan.sink_end == 96
    # projected = 8112, target = 6096, drop_needed = 2016 (already aligned to 16)
    assert plan.delta == 2016
    assert plan.moved_from == 96 + 2016 == 2112
    # Resulting retained length must be exactly target (6096), not 6192!
    retained_length = (8100 + 12) - plan.delta
    assert retained_length == 6096


def test_the_shift_is_a_whole_number_of_pages_so_no_row_moves():
    """The claim that makes a trim a memory pass rather than a copy.

    Deleting ``n`` gap entries shifts every later logical index down by ``n``.
    Because ``delta == n * block_size``, a tail token's new logical index lands on
    the table slot that already names its own page, and its offset within that
    page is unchanged -- so the physical slot is identical before and after.
    """
    geometry = _geometry(high_watermark_tokens=96)
    plan = plan_position_reanchor(geometry, computed_tokens=200, pending_tokens=20)
    assert plan is not None
    n_gap = plan.delta // BLOCK
    table_before = list(range(64))  # one fake physical page id per logical slot
    table_after = table_before[: plan.sink_blocks] + table_before[plan.sink_blocks + n_gap :]

    for position in range(plan.moved_from, 64 * BLOCK):
        before = divmod(position, BLOCK)
        after = divmod(reanchor_positions(position, plan), BLOCK)
        assert table_before[before[0]] == table_after[after[0]]
        assert before[1] == after[1]


def test_unit_aligned_cut_keeps_whole_units():
    # Page-aligned unit starts: prefix_tokens=16 (divisible by BLOCK=4), units=[12]*8
    units = [12] * 8
    geometry = _geometry(prefix_tokens=16, window_tokens=30, high_watermark_tokens=40)
    boundaries = unit_starts(geometry, units)
    assert boundaries[0] == 16
    # Unit starts: 16, 28, 40, 52, 64, ... all divisible by 4!
    plan = plan_position_reanchor(geometry, computed_tokens=16 + sum(units), pending_tokens=12, unit_tokens=units)
    assert plan is not None
    # Landed on a block-aligned unit start, so no unit was split in half.
    assert plan.moved_from in boundaries
    assert plan.moved_from % BLOCK == 0
    # Post trim length <= target
    target = geometry.prefix_tokens + geometry.window_tokens
    assert (16 + sum(units) + 12) - plan.delta <= target


def test_unit_aligned_cut_rejects_non_page_aligned_units():
    # If units cannot be cut along page boundaries without splitting a unit in half,
    # plan_position_reanchor should return None to trigger fallback rather than slicing a unit.
    units_odd = [7] * 8
    geo_no_align = DuplexWindowGeometry(
        prefix_tokens=11,
        window_tokens=20,
        block_size=8,
        max_model_len=4096,
        high_watermark_tokens=30,
    )
    # Unit starts: 11, 18, 25, 32, 39, 46, 53, 60, 67. None are divisible by 8!
    plan = plan_position_reanchor(geo_no_align, computed_tokens=11 + 56, pending_tokens=7, unit_tokens=units_odd)
    assert plan is None


def test_reanchor_is_a_uniform_translation_of_the_tail():
    geometry = _geometry(high_watermark_tokens=96)
    plan = plan_position_reanchor(geometry, computed_tokens=200, pending_tokens=20)
    assert plan is not None
    assert plan.delta == plan.moved_from - plan.sink_end

    assert reanchor_positions(3, plan) == 3
    tail_old = plan.moved_from + 5
    tail_new = reanchor_positions(tail_old, plan)
    assert tail_new == tail_old - plan.delta
    # Relative distances inside the tail are untouched by definition, which is
    # what lets one rotation fix every pair at once.
    assert reanchor_positions(plan.moved_from + 37, plan) - tail_new == 32
    # The reclaimed range is dense: the tail lands exactly on the sink's end.
    assert reanchor_positions(plan.moved_from, plan) == plan.sink_end

    for dropped in (plan.sink_end, plan.moved_from - 1):
        with pytest.raises(ValueError, match="dropped"):
            reanchor_positions(dropped, plan)


def test_reanchor_declines_when_there_is_nothing_left_to_reclaim():
    geometry = _geometry(high_watermark_tokens=96)
    # Target at or past the projected length: compacting would not help, so the
    # caller has to finish the session with context_length_exceeded instead of moving
    # the window start backwards.
    assert plan_position_reanchor(geometry, computed_tokens=100, pending_tokens=20, target_tokens=160) is None
    assert plan_position_reanchor(geometry, computed_tokens=100, pending_tokens=20, target_tokens=120) is None
    plan = plan_position_reanchor(geometry, computed_tokens=100, pending_tokens=20, target_tokens=20)
    assert plan is not None
    assert plan.sink_end >= geometry.prefix_tokens
    assert reanchor_positions(2, plan) == 2


@pytest.mark.parametrize("head_dim", [8, 16])
@pytest.mark.parametrize("delta", [4, 64, 1024])
def test_rotating_cached_keys_equals_rope_at_the_new_position(head_dim: int, delta: int):
    """The other half of the claim: the rotation is the re-RoPE.

    A key written at position ``p`` holds ``z * exp(i * p * f)`` per frequency
    pair. Re-anchoring to ``p - delta`` can therefore be done on the cached
    tensor by rotating ``-delta * f`` -- no recompute of the model, and exact for
    any p, including a tail whose keys were written at different positions.
    """
    plan = PositionReanchor(delta=delta, moved_from=4096, sink_blocks=4096 // BLOCK)
    inv_freq = [0.5 ** (2.0 * i / head_dim) for i in range(head_dim // 2)]
    rng = iter(range(1, 10_000))
    for pair, freq in enumerate(inv_freq):
        re = complex(next(rng) % 7 - 3, next(rng) % 7 - 3)
        for position in (plan.moved_from + 1, plan.moved_from + 3 * pair + 17, 999_999):
            cached = re * cmath.exp(1j * position * freq)
            rotated = cached * cmath.exp(1j * plan.rope_angle_shift_tokens * freq)
            direct = re * cmath.exp(1j * reanchor_positions(position, plan) * freq)
            assert rotated == pytest.approx(direct, rel=1e-9, abs=1e-12), f"pair {pair} at {position}"


def test_reanchor_rejects_a_fractional_position():
    plan = PositionReanchor(delta=8, moved_from=64, sink_blocks=16)
    with pytest.raises(ValueError, match="integer"):
        reanchor_positions(70.5, plan)


# Realistic Stage 0 numbers: a ~100-token head context, the 6000-token low
# watermark from the duplex window policy, 16-token KV blocks, and the 40960
# positions the checkpoint's config.json advertises.
def _stage0_geometry(**overrides) -> DuplexWindowGeometry:
    kwargs: dict[str, int] = {
        "prefix_tokens": 96,
        "window_tokens": 6000,
        "block_size": 16,
        "max_model_len": 40960,
        "sample_room": 2048,
        # The reference implementation's trigger/stride pair, in tokens: roll
        # back to the window once 2000 tokens have piled on past it.
        "high_watermark_tokens": 8000,
    }
    kwargs.update(overrides)
    return DuplexWindowGeometry(**kwargs)


def test_ten_minutes_of_audio_recomputes_nothing():
    """What the rebuild-based window pays, measured against this design.

    600 one-second units is about 7.2k appended tokens plus the spoken
    interleaves. A window that rebuilds the prompt re-prefills the whole retained
    tail every time it rolls; a trim re-prefills nothing and still bounds the
    session's residency, so the position ceiling is never in sight.
    """
    geometry = _stage0_geometry()
    units = [12] * 600  # 1 s of audio: 10 pooled embeddings + unit closure.
    spoken = 20  # tokens the model speaks per unit, at 1 s of audio.

    position = geometry.prefix_tokens
    trims = 0
    peak_rows = 0
    written = geometry.prefix_tokens
    # resident_bound() from the engine tier, plus whatever single append was
    # already in flight when the trim came due: content rides to the trigger, and
    # a trim then pulls it back to the low watermark.
    bound = geometry.prefix_tokens + geometry.trigger_tokens + max(units) + spoken + geometry.sample_room
    for unit in units:
        assert position + unit + spoken <= bound, "the session outgrew its own resident bound"
        plan = plan_position_reanchor(geometry, computed_tokens=position, pending_tokens=unit + spoken)
        if plan is not None:
            trims += 1
            # Rows the rotation touches: the reclaimed tail, and never more than
            # what is resident. A re-prefill would run those same rows through
            # every layer of the model instead.
            peak_rows = max(peak_rows, position - plan.moved_from)
            position -= plan.delta
        position += unit + spoken
        written += unit + spoken

    assert trims > 0, "the window has to roll over a session this long"
    assert peak_rows <= bound
    # Ten minutes of conversation wrote ~19k positions while never holding more
    # than the watermark: recycling, not the length, is what bounds memory.
    assert written > bound
    # A 40k position budget against an 8k resident bound: the ceiling is a
    # backstop this session never approaches, let alone crosses.
    assert position + geometry.sample_room < geometry.max_model_len


def test_a_long_session_trims_once_per_spare_window_not_once_per_append():
    """The frequency the whole design is priced at."""
    geometry = _stage0_geometry()
    position = geometry.prefix_tokens
    append = 12
    gaps: list[int] = []
    since = 0
    peak = position
    for _ in range(4000):
        plan = plan_position_reanchor(geometry, computed_tokens=position, pending_tokens=append)
        if plan is not None:
            gaps.append(since)
            since = 0
            position -= plan.delta
        since += 1
        position += append
        peak = max(peak, position)
    # 48k positions appended against a window that only overflows once the spare
    # tokens past it are used up: one trim services ~160 one-second appends. The
    # first gap is longer because the window has to fill from nothing first.
    spare = geometry.trigger_tokens - geometry.window_tokens
    assert gaps and all(gap > spare // (2 * append) for gap in gaps)
    assert all(gap <= (spare + append - 1) // append + 1 for gap in gaps[1:])
    # The watermark, not the 40960-token ceiling, is what holds the session down.
    assert peak <= geometry.prefix_tokens + geometry.trigger_tokens + append
    assert peak + geometry.sample_room < geometry.max_model_len

    # The trigger is a policy choice, so a stage with no window is still legal --
    # it simply trades the rotation for a shorter session cap.
    unwindowed = _stage0_geometry(window_tokens=40960, high_watermark_tokens=None)
    assert unwindowed.trigger_tokens == 40960
