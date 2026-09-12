# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import random

import pytest

from vllm_omni.model_executor.stage_input_processors.streaming_text_segmentation import (
    CapacityAdaptiveSegmenter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _zh_chars(text: str) -> list[str]:
    """Char-tokenize zh text; punctuation stays its own single-char token."""
    return list(text)


def test_budget_is_capacity_divided_by_measured_rho():
    segmenter = CapacityAdaptiveSegmenter()
    segmenter.observe_segment(acoustic_steps=100, text_tokens=50)  # rho = 2.0

    assert segmenter.open_segment(remaining_capacity=200) == 100
    assert segmenter.open_segment(remaining_capacity=150) == 75


def test_ema_smooths_rho_estimates():
    segmenter = CapacityAdaptiveSegmenter(ema_alpha=0.5)
    segmenter.observe_segment(acoustic_steps=100, text_tokens=50)  # rho = 2.0
    segmenter.observe_segment(acoustic_steps=200, text_tokens=50)  # rho = 4.0

    assert segmenter.rho_hat == pytest.approx(3.0)  # 0.5 * 4 + 0.5 * 2


def test_warmup_uses_caller_default_until_first_observation():
    segmenter = CapacityAdaptiveSegmenter()

    assert segmenter.open_segment(remaining_capacity=200, max_text_tokens=30) == 30
    assert segmenter.open_segment(remaining_capacity=200) == 1

    segmenter.observe_segment(acoustic_steps=50, text_tokens=50)  # rho = 1.0
    assert segmenter.open_segment(remaining_capacity=200) == 200


def test_max_text_tokens_caps_budget():
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)

    assert segmenter.open_segment(remaining_capacity=200, max_text_tokens=50) == 50


def test_frozen_budget_is_immune_to_mid_segment_observations():
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    budget = segmenter.open_segment(remaining_capacity=100)
    assert budget == 100

    # A mid-segment observation must not shift the already-frozen budget.
    segmenter.observe_segment(acoustic_steps=500, text_tokens=50)  # rho = 10
    assert segmenter.active_budget == 100

    # It does shape the *next* segment's budget.
    assert segmenter.open_segment(remaining_capacity=100) == 10


def test_segment_never_exceeds_acoustic_capacity():
    """#5889 regression property: budget * rho_hat <= remaining_capacity."""
    segmenter = CapacityAdaptiveSegmenter()
    segmenter.observe_segment(acoustic_steps=60, text_tokens=30)  # rho = 2.0

    for capacity in (30, 61, 149, 150, 1000):
        budget = segmenter.open_segment(remaining_capacity=capacity)
        assert budget * segmenter.rho_hat <= capacity


def test_punctuation_rank_prefers_sentence_final_over_clause():
    tokens = _zh_chars("一二三四五六七八。九十，然后继续")
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    segmenter.open_segment(remaining_capacity=12)

    cut = segmenter.select_cut(tokens)
    # Window is [0.7*12, 12) = [8, 12): "。" at index 8 and a weaker "，" at
    # index 11 both qualify; strength wins over rightmost-ness.
    assert cut.cut_index == 9
    assert cut.reason == "sentence_final"
    assert cut.fill_fraction >= 0.7


def test_weaker_punctuation_accepted_when_budget_drains():
    # No sentence-final inside the window; clause "，" at index 10 qualifies.
    tokens = _zh_chars("一二三四五六七八九十，然后继续说话内容很长")
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    segmenter.open_segment(remaining_capacity=13)

    cut = segmenter.select_cut(tokens)
    assert cut.reason == "clause"
    assert cut.cut_index == 11


def test_punctuation_below_fill_window_is_rejected_and_hard_cuts():
    # Sentence-final at index 5 fills only 5/12 < 0.7 * 12; text continues
    # past the budget, so the segment hard-cuts at the ceiling.
    tokens = _zh_chars("短句。然后后面还有很长很长的内容继续往下说")
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    segmenter.open_segment(remaining_capacity=12)

    cut = segmenter.select_cut(tokens)
    assert cut.reason == "hard_cut"
    assert cut.cut_index == 12
    assert cut.fill_fraction == 1.0


def test_text_ending_inside_window_cuts_at_end_of_text():
    tokens = _zh_chars("今天的内容就到这里")
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    segmenter.open_segment(remaining_capacity=20)

    cut = segmenter.select_cut(tokens)
    assert cut.reason == "end_of_text"
    assert cut.cut_index == len(tokens)


def test_committed_abbreviation_token_is_never_a_sentence_cut():
    tokens = ["this", "is", "e.g.", "中文"]
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    segmenter.open_segment(remaining_capacity=4)

    cut = segmenter.select_cut(tokens)
    assert cut.reason != "sentence_final"
    assert cut.cut_index == len(tokens)


def test_multi_char_punctuation_run_is_classified_by_first_char():
    tokens = _zh_chars("他说完了……然后继续")
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0)
    segmenter.open_segment(remaining_capacity=8)

    cut = segmenter.select_cut(tokens)
    assert cut.reason == "sentence_final"


@pytest.mark.parametrize("budget", (8, 12, 20))
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_fragmentation_bound_holds_with_random_punctuation(budget: int, seed: int) -> None:
    """Greedy fill-window segmentation never over-fragments.

    Every non-final segment fills at least floor(alpha * budget) tokens, so
    segment count is bounded by ceil(total / floor(alpha * budget)) + 1.
    """
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    punct = "。，、"
    tokens: list[str] = []
    for _ in range(250):
        if rng.random() < 0.06:
            tokens.append(rng.choice(punct))
        else:
            tokens.append(rng.choice(alphabet))

    segmenter = CapacityAdaptiveSegmenter(warmup_rho=1.0, min_fill_fraction=0.7)
    consumed = 0
    segments = 0
    min_fill = 1.0
    floor_fill = max(1, int(0.7 * budget)) / budget
    while consumed < len(tokens):
        segmenter.open_segment(remaining_capacity=budget)
        cut = segmenter.select_cut(tokens[consumed:])
        assert cut.cut_index >= max(1, int(0.7 * budget)) or consumed + cut.cut_index >= len(tokens)
        # Only non-final segments must fill the window; the tail segment may
        # legitimately be short (end_of_text).
        if consumed + cut.cut_index < len(tokens):
            min_fill = min(min_fill, cut.fill_fraction)
        consumed += cut.cut_index
        segments += 1

    bound = -(-len(tokens) // max(1, int(0.7 * budget))) + 1
    assert min_fill >= floor_fill
    assert segments <= bound


def test_open_segment_requires_positive_capacity():
    segmenter = CapacityAdaptiveSegmenter()
    with pytest.raises(ValueError, match="remaining_capacity"):
        segmenter.open_segment(remaining_capacity=0)


def test_select_cut_requires_open_segment():
    segmenter = CapacityAdaptiveSegmenter()
    with pytest.raises(RuntimeError, match="open_segment"):
        segmenter.select_cut(["a"])


def test_constructor_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="ema_alpha"):
        CapacityAdaptiveSegmenter(ema_alpha=0.0)
    with pytest.raises(ValueError, match="min_fill_fraction"):
        CapacityAdaptiveSegmenter(min_fill_fraction=1.5)
    with pytest.raises(ValueError, match="warmup_rho"):
        CapacityAdaptiveSegmenter(warmup_rho=-1.0)


def test_degenerate_observations_are_ignored():
    segmenter = CapacityAdaptiveSegmenter(warmup_rho=2.0)
    segmenter.observe_segment(acoustic_steps=0, text_tokens=10)
    segmenter.observe_segment(acoustic_steps=10, text_tokens=0)
    assert segmenter.rho_hat == 2.0
