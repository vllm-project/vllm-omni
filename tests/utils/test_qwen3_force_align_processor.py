# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.utils import qwen3_force_align_processor as processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_prompt_has_boundary_timestamp_markers():
    prompt = processor.build_prompt(["hello", "world"])

    assert prompt.count("<timestamp>") == 4
    assert "hello<timestamp><timestamp>world" in prompt
    # Official aligner format: audio placeholder + words, no chat template
    # (a leading "<|im_start|>user\n" shifts the predicted markers one bin late).
    assert prompt == f"{processor.AUDIO_PLACEHOLDER}hello<timestamp><timestamp>world<timestamp><timestamp>"
    assert "<|im_start|>" not in prompt
    assert processor.build_prompt([]) == f"{processor.AUDIO_PLACEHOLDER}<timestamp><timestamp>"


def test_build_prompt_matches_official_encode_timestamp():
    pytest.importorskip("qwen_asr")
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

    text = "It's 3 o'clock, 你好 world."
    word_list, official_prompt = Qwen3ForceAlignProcessor().encode_timestamp(text, "english")

    assert processor.build_prompt(list(word_list)) == official_prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        # Punctuation inside a whitespace segment is stripped, not split on:
        # the hand-rolled splitter used to break these into many fake words.
        ("U.S.A", ["USA"]),
        ("hello, world!", ["hello", "world"]),
        ("don't stop", ["don't", "stop"]),
        # CJK characters peel off as individual tokens, Latin runs stay whole.
        ("你好world", ["你", "好", "world"]),
        ("我爱 China", ["我", "爱", "China"]),
        ("   spaced   out   ", ["spaced", "out"]),
    ],
)
def test_tokenize_space_lang_matches_official_segmentation(text, expected):
    assert processor._tokenize_space_lang(text) == expected


def test_segment_words_falls_back_to_port_without_qwen_asr(monkeypatch):
    # With qwen_asr unavailable, segmentation must use the built-in port and
    # still produce the faithful result for the common (non-JP/KO) case.
    monkeypatch.setattr(processor, "_get_official_processor", lambda: None)

    assert processor.segment_words("U.S.A is here", "auto") == ["USA", "is", "here"]


@pytest.mark.parametrize(
    "values,expected",
    [
        ([], []),
        ([7], [7]),
        # Equal values must extend the subsequence.
        ([0, 100, 100, 100, 250], [0, 100, 100, 100, 250]),
        # Keep the first endpoint when all subsequences have length one.
        ([400, 300, 200, 100], [400, 400, 400, 400]),
        # Keep the first predecessor, rather than the smallest tail value.
        ([0, 400, 200, 800], [0, 400, 400, 800]),
        ([4, 1, 3, 2, 5], [1, 1, 3, 3, 5]),
        # A run of three rejected values uses interpolation.
        ([0, 800, 700, 600, 500, 1000], [0, 800, 850, 900, 950, 1000]),
    ],
)
def test_fix_timestamp_preserves_sequence_selection(values, expected):
    assert processor.fix_timestamp(values) == expected
