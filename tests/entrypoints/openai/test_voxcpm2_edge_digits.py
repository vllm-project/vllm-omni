# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the VoxCPM2 edge digit-run prosody shaping."""

import pytest

from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import (
    shape_voxcpm2_edge_digits,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _edge_pad_on(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SPEECH_EDGE_PAD", "1")


def test_leading_ten_gets_comma_lead_in():
    assert shape_voxcpm2_edge_digits("10今天天气真好") == "10，今天天气真好"


def test_leading_single_digit_gets_comma_lead_in():
    assert shape_voxcpm2_edge_digits("1今天天气真好") == "1，今天天气真好"


def test_leading_two_digit_run_gets_comma_lead_in():
    assert shape_voxcpm2_edge_digits("48今天天气真好") == "48，今天天气真好"


def test_trailing_bare_digit_gets_full_stop():
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。10") == "今天天气真好，一起去吃饭。10。"
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。7") == "今天天气真好，一起去吃饭。7。"


def test_trailing_digit_with_comma_becomes_full_stop():
    # "一零" (87%, n=100) is triggered by the comma; the rule replaces it.
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。10，") == "今天天气真好，一起去吃饭。10。"
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。7，") == "今天天气真好，一起去吃饭。7。"


def test_trailing_digit_with_full_stop_untouched():
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。10。") == "今天天气真好，一起去吃饭。10。"


def test_trailing_digit_before_non_terminal_char_untouched():
    # the stop-pad only fires when the digit-run ends the text; a following
    # non-terminal char keeps the text as-is (prosody belongs to the sentence)
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。10啊") == "今天天气真好，一起去吃饭。10啊"


def test_non_digit_ending_untouched():
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭啊") == "今天天气真好，一起去吃饭啊"
    assert shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭啊。") == "今天天气真好，一起去吃饭啊。"


def test_lone_leading_zero_is_comma_broken():
    # the run-pad fires first (0，); the sentinel replacement happens in
    # normalize_voxcpm2_text after TN, keyed on the same break
    assert shape_voxcpm2_edge_digits("0今天天气真好") == "0，今天天气真好"


def test_lone_trailing_zero_is_sentinel_held():
    out = shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。0。")
    assert out.endswith("\uee00。")


def test_zero_inside_multi_digit_run_is_not_sentinel_split():
    # THE regression: the old endswith("0。") check split "10。" into
    # "1<sentinel>。" = "一零" (77%, n=100). The lookbehind must not fire.
    out = shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。10")
    assert "1\uee00" not in out
    assert out == "今天天气真好，一起去吃饭。10。"


def test_edge_pad_disabled_returns_text_untouched(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SPEECH_EDGE_PAD", "0")
    text = "10今天天气真好，一起去吃饭。10，"
    assert shape_voxcpm2_edge_digits(text) == text


def test_empty_text_passthrough():
    assert shape_voxcpm2_edge_digits("") == ""
