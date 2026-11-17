# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the online CAPS segmenter (RFC #6496 mechanism 2).

The first four tests walk two worked examples token by token, pinning the cut
position and the per-level thresholds so the online state machine cannot
silently drift back into an offline helper.
"""

import pytest

from vllm_omni.model_executor.stage_input_processors.streaming_text_segmentation import (
    CapacityAdaptiveSegmenter,
    classify_punctuation_level,
    compute_thresholds,
    derive_text_token_capacity,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# "No fear of words, no fear of years. Better later than never."
_TOKENS_ONE_SEGMENT = [
    "No",
    " fear",
    " of",
    " words,",
    " no",
    " fear",
    " of",
    " years.",
    " Better",
    " later",
    " than",
    " never.",
]

# Same sentence plus a clause carrying " What's more," and " fresh,".
_TOKENS_TWO_LEVELS = [
    "No",
    " fear",
    " of",
    " words,",
    " no",
    " fear",
    " of",
    " years.",
    " Better",
    " later",
    " than",
    " never.",
    " What's",
    " more,",
    " tomorrow",
    " is",
    " always",
    " fresh,",
    " with",
    " no",
    " mistakes",
    " in",
    " it",
    " yet.",
]

# Token positions used by the two-level examples.
_PERIOD_BEFORE_L1 = 12  # " never." -- below the level-1 threshold
_COMMA_BEFORE_L2 = 14  # " more,"   -- below the level-2 threshold
_COMMA_AT_L2 = 18  # " fresh,"  -- reaches the level-2 threshold


def _segmenter(**kwargs) -> CapacityAdaptiveSegmenter:
    """Segmenter whose capacity equals the remaining capacity it is given."""
    return CapacityAdaptiveSegmenter(warmup_expansion_ratio=1.0, safety_margin=0, **kwargs)


def _cuts(capacity: int, tokens: list[str]) -> list[tuple[int, int, bool]]:
    """Drive one segment online and return ``(text_tokens, punct_level, forced)``."""
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=capacity)
    cuts = segmenter.append_tokens(tokens)
    final = segmenter.finish()
    if final is not None:
        cuts.append(final)
    return [(cut.text_tokens, cut.punct_level, cut.is_forced) for cut in cuts]


def test_single_level_example_splits_at_the_sentence_final_boundary():
    """Capacity 10 with a level-1 ratio of 0.7 cuts at token 8.

    The comma at token 4 stays open because the clause level needs 8 tokens.
    """
    assert _cuts(10, _TOKENS_ONE_SEGMENT) == [(8, 1, False), (4, 0, True)]


def test_weaker_boundary_waits_for_its_own_stricter_threshold():
    """Capacity 20 with L1=0.7 and L2=0.8 cuts at token 18.

    The period at token 12 and the comma at token 14 are both too early; only
    the comma at token 18 clears the clause threshold.
    """
    assert _cuts(20, _TOKENS_TWO_LEVELS)[0] == (_COMMA_AT_L2, 2, False)


def test_stronger_boundary_below_its_threshold_does_not_cut():
    """The period at token 12 is under the level-1 threshold, so nothing cuts."""
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=20)
    assert segmenter.append_tokens(_TOKENS_TWO_LEVELS[:_PERIOD_BEFORE_L1]) == []


def test_weaker_boundary_below_its_threshold_does_not_cut():
    """The comma at token 14 is under the level-2 threshold, so nothing cuts."""
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=20)
    assert segmenter.append_tokens(_TOKENS_TWO_LEVELS[:_COMMA_BEFORE_L2]) == []


def test_threshold_is_met_by_the_token_that_reaches_it():
    """Off-by-one regression: capacity 10 * 0.7 = 7 splits at the 7th token."""
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=10)
    cuts = segmenter.append_tokens(["a", "b", "c", "d", "e", "f", "done."])
    assert len(cuts) == 1


def test_token_by_token_matches_batched_append():
    """The online decision must not depend on how tokens are batched."""
    online = _segmenter()
    online.start_segment(remaining_capacity=20)
    token_by_token = [cut for token in _TOKENS_TWO_LEVELS if (cut := online.append_token(token)) is not None]

    batched = _segmenter()
    batched.start_segment(remaining_capacity=20)
    assert token_by_token == batched.append_tokens(_TOKENS_TWO_LEVELS)


def test_committed_boundary_is_not_moved_by_later_tokens():
    """A cut commits on the boundary token; the next segment starts empty."""
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=20)
    segmenter.append_tokens(_TOKENS_TWO_LEVELS[: _COMMA_AT_L2 - 1])
    cut = segmenter.append_token(" fresh,")
    assert cut is not None and cut.text_tokens == _COMMA_AT_L2
    assert segmenter.token_count == 0


def test_ceiling_forces_a_cut_without_punctuation():
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=10)
    cuts = segmenter.append_tokens(["w"] * 10)
    assert [(cut.text_tokens, cut.is_forced) for cut in cuts] == [(10, True)]


def test_finish_forces_the_tail_split():
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=10)
    segmenter.append_tokens(["hello", " world"])
    final = segmenter.finish()
    assert final is not None and (final.text_tokens, final.is_forced) == (2, True)


def test_finish_returns_none_when_the_tail_is_empty():
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=10)
    assert segmenter.finish() is None


def test_a_short_prefix_stays_open_until_finish():
    """A prefix shorter than the capacity stays open until finish()."""
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=10)
    assert segmenter.append_token("No") is None


def test_append_token_requires_a_started_segment():
    segmenter = _segmenter()
    with pytest.raises(RuntimeError, match="start_segment"):
        segmenter.append_token("token")


def test_start_segment_requires_an_expansion_ratio_estimate():
    segmenter = CapacityAdaptiveSegmenter()
    with pytest.raises(RuntimeError, match="expansion-ratio estimate"):
        segmenter.start_segment(remaining_capacity=100)


@pytest.mark.parametrize(
    "token,expected_level",
    [
        ("done.", 1),
        ("好。", 1),
        ("really?", 1),
        ("now!", 1),
        ("fresh,", 2),
        ("words;", 2),
        ('done."', 1),
        ("word", 0),
        ("", 0),
        ("tail ", 0),
        ("\n", 3),
        ("gap—", 3),
        ("wait…", 3),
    ],
)
def test_classify_punctuation_level_reads_the_right_boundary(token, expected_level):
    assert classify_punctuation_level(token) == expected_level


@pytest.mark.parametrize(
    "remaining,ratio,safety,max_tokens,expected",
    [
        (108, 2.0, 8, None, 50),
        (108, 2.0, 8, 20, 20),
        (108, 2.0, 8, 80, 50),
    ],
)
def test_capacity_is_capped_but_never_bypassed(remaining, ratio, safety, max_tokens, expected):
    """max_text_tokens only lowers the capacity derived from the remaining capacity."""
    capacity = derive_text_token_capacity(
        remaining_capacity=remaining,
        expansion_ratio=ratio,
        safety_margin=safety,
        max_text_tokens=max_tokens,
    )
    assert capacity == expected


def test_capacity_below_one_token_is_rejected():
    with pytest.raises(ValueError, match="cannot fit one predicted text token"):
        derive_text_token_capacity(remaining_capacity=9, expansion_ratio=2.0, safety_margin=8)


def test_thresholds_tighten_with_the_punctuation_level():
    thresholds = compute_thresholds(text_token_capacity=100)
    assert thresholds.min_tokens_level1 < thresholds.min_tokens_level2 < thresholds.min_tokens_level3


def test_threshold_ceiling_equals_the_capacity():
    assert compute_thresholds(text_token_capacity=37).force_split_at == 37


def test_non_monotonic_capacity_ratios_are_rejected():
    with pytest.raises(ValueError, match="level1 <= level2 <= level3"):
        compute_thresholds(text_token_capacity=10, level1_capacity_ratio=0.9, level2_capacity_ratio=0.8)


def test_an_observation_moves_the_duration_estimate_and_raises_safety():
    """Warmup 7.0 then rho 2.0: the EMA moves, safety keeps the higher value."""
    segmenter = CapacityAdaptiveSegmenter(warmup_expansion_ratio=7.0)
    segmenter.observe_segment(acoustic_steps=100, text_tokens=50)  # rho = 2.0
    assert segmenter.duration_ratio == pytest.approx(0.7 * 7.0 + 0.3 * 2.0)
    assert segmenter.safety_ratio == pytest.approx(7.0)


def test_ema_smooths_subsequent_duration_observations():
    segmenter = CapacityAdaptiveSegmenter(ema_alpha=0.5, warmup_expansion_ratio=2.0)
    segmenter.observe_segment(acoustic_steps=100, text_tokens=50)  # 2.0 -> EMA stays 2.0
    segmenter.observe_segment(acoustic_steps=200, text_tokens=50)  # 4.0 -> 0.5*4 + 0.5*2
    assert segmenter.duration_ratio == pytest.approx(3.0)


def test_safety_ratio_never_decreases():
    segmenter = CapacityAdaptiveSegmenter(safety_margin=0)
    segmenter.observe_segment(acoustic_steps=200, text_tokens=20)  # rho = 10.0
    segmenter.observe_segment(acoustic_steps=20, text_tokens=20)  # rho = 1.0
    assert segmenter.safety_ratio == pytest.approx(10.0)


def test_capacity_uses_the_monotonic_safety_ratio_not_the_duration_ema():
    """A later fast segment must not widen the hard capacity.

    The duration EMA falls towards the fast observation, but capacity keeps
    being sized from the monotonic safety ratio, so the planned segment cannot
    exceed what the acoustic stage was already shown to need.
    """
    segmenter = CapacityAdaptiveSegmenter(safety_margin=0)
    segmenter.observe_segment(acoustic_steps=200, text_tokens=20)  # rho = 10.0
    segmenter.observe_segment(acoustic_steps=20, text_tokens=20)  # rho = 1.0
    assert segmenter.duration_ratio < segmenter.safety_ratio
    assert segmenter.start_segment(remaining_capacity=100).force_split_at == 10


def test_short_segment_observations_are_ignored():
    """A too-short segment is too noisy to estimate or plan from."""
    segmenter = CapacityAdaptiveSegmenter(warmup_expansion_ratio=2.0, min_duration_tokens=8)
    segmenter.observe_segment(acoustic_steps=100, text_tokens=4)
    assert segmenter.safety_ratio == pytest.approx(2.0)


def test_observed_ratio_is_bounded_by_the_guard_rail():
    """A pathological observation cannot tighten the capacity without bound."""
    segmenter = CapacityAdaptiveSegmenter(max_expansion_ratio=16.0)
    segmenter.observe_segment(acoustic_steps=100_000, text_tokens=10)
    assert segmenter.safety_ratio == pytest.approx(16.0)


def test_degenerate_observations_are_ignored():
    segmenter = CapacityAdaptiveSegmenter(warmup_expansion_ratio=2.0)
    segmenter.observe_segment(acoustic_steps=0, text_tokens=10)
    segmenter.observe_segment(acoustic_steps=10, text_tokens=0)
    assert segmenter.safety_ratio == pytest.approx(2.0)


def test_frozen_thresholds_ignore_a_mid_segment_observation():
    segmenter = _segmenter()
    frozen = segmenter.start_segment(remaining_capacity=100)
    segmenter.observe_segment(acoustic_steps=500, text_tokens=50)
    assert segmenter.thresholds == frozen


def test_measured_ratio_shapes_the_next_segment_capacity():
    """Closed loop: a realized rho of 10 shrinks the next segment's capacity."""
    segmenter = CapacityAdaptiveSegmenter(warmup_expansion_ratio=1.0, safety_margin=0)
    segmenter.observe_segment(acoustic_steps=500, text_tokens=50)  # rho = 10
    assert segmenter.start_segment(remaining_capacity=100).force_split_at == 10


def test_seam_contract_from_code2wav_measurement_to_next_capacity():
    """Seam the live wiring will connect (#6496 mechanism 2).

    Code2Wav reports a finished segment as (realized acoustic frames, text
    tokens); the segmenter turns that into the next segment's text capacity.
    ``observe_segment`` is where rho arrives and ``start_segment`` is where the
    next capacity is applied, so this pins both ends of the unwired loop.
    """
    segmenter = CapacityAdaptiveSegmenter(safety_margin=0)
    segmenter.observe_segment(acoustic_steps=240, text_tokens=120)  # rho = 2.0
    assert segmenter.start_segment(remaining_capacity=100).force_split_at == 50


def test_segment_never_plans_more_acoustic_steps_than_remaining():
    """#5889 capacity property: capacity * safety_ratio <= remaining capacity."""
    segmenter = CapacityAdaptiveSegmenter(safety_margin=0)
    segmenter.observe_segment(acoustic_steps=300, text_tokens=150)  # rho = 2.0
    for remaining in (30, 61, 149, 150, 1000):
        capacity = segmenter.start_segment(remaining_capacity=remaining).force_split_at
        assert capacity * segmenter.safety_ratio <= remaining


def test_a_committed_cut_reopens_the_next_segment_with_the_same_thresholds():
    """A batch spanning two boundaries yields two online cuts, not one.

    After a cut the next segment starts empty and reuses the frozen thresholds
    until the caller refreshes them with ``start_segment()``.
    """
    segmenter = _segmenter()
    segmenter.start_segment(remaining_capacity=10)  # level-1 threshold is 7
    tokens = ["w"] * 6 + ["end."] + ["w"] * 6 + ["end."]
    cuts = segmenter.append_tokens(tokens)
    assert [cut.text_tokens for cut in cuts] == [7, 7]


def test_threshold_lookup_rejects_an_unknown_level():
    thresholds = compute_thresholds(text_token_capacity=10)
    with pytest.raises(ValueError, match="punctuation level"):
        thresholds.min_tokens_for_level(0)


def test_constructor_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="ema_alpha"):
        CapacityAdaptiveSegmenter(ema_alpha=0.0)
    with pytest.raises(ValueError, match="warmup_expansion_ratio"):
        CapacityAdaptiveSegmenter(warmup_expansion_ratio=0.5)
    with pytest.raises(ValueError, match="level1 <= level2 <= level3"):
        CapacityAdaptiveSegmenter(level1_capacity_ratio=0.9, level2_capacity_ratio=0.8)


def test_segments_fill_at_least_the_weakest_early_threshold():
    """Deterministic fragmentation bound: a cut never passes the level-1 ceiling.

    Level 1 has the lowest threshold, so no non-final segment can be shorter
    than ``min_tokens_level1``; segment count is therefore bounded by
    ``ceil(total / min_tokens_level1) + 1``.
    """
    capacity = 20
    # Alternating words and sentence-final separators: every second token is an
    # eligible level-1 boundary once the threshold is reached.
    tokens = [token for index in range(60) for token in (f"w{index}", "end.")]

    segmenter = _segmenter()
    thresholds = segmenter.start_segment(remaining_capacity=capacity)
    cuts = segmenter.append_tokens(tokens)
    final = segmenter.finish()
    if final is not None:
        cuts.append(final)

    committed = cuts[:-1]
    assert committed and all(cut.text_tokens >= thresholds.min_tokens_level1 for cut in committed)
    assert len(cuts) <= -(-len(tokens) // thresholds.min_tokens_level1) + 1
