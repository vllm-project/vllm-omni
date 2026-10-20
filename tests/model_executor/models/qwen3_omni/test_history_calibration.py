# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.qwen3_omni.duplex.history import (
    asr_prefix,
    text_units,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_normalization_preserves_original_offsets():
    original = "今天用ＡＩ，Don't stop！"
    units = text_units(original)
    assert [u.normalized for u in units] == ["今", "天", "用", "ai", "don't", "stop"]
    assert [original[u.start : u.end] for u in units] == ["今", "天", "用", "ＡＩ", "Don't", "stop"]


@pytest.mark.parametrize(
    ("text", "asr", "expected"),
    [
        ("你好，今天我们介绍三个方法。", "你好今天我们介绍", "你好，今天我们介"),
        ("Hello, we will explain three options today.", "hello we will explain three", "Hello, we will explain"),
        ("今天用ＡＩ，Don't stop！", "今天用ai don't stop", "今天用ＡＩ，Don't"),
        ("We don’t stop here.", "WE DON'T STOP", "We don’t"),
        ("First,\nsecond; third.", "first! second? third", "First,\nsecond"),
        ("One two three.", "one two three extra words", "One two"),
        ("你好，世界！", "你好世界", "你好，世"),
    ],
)
def test_asr_selects_original_prefix_and_withholds_last_matched_unit(text, asr, expected):
    result = asr_prefix(text, asr)
    assert result.accepted, result
    assert text[: result.char_end] == expected


@pytest.mark.parametrize(
    ("text", "asr", "expected"),
    [
        ("今天下午开会请提前到场。", "今天下午开慧请提前到", "今天下午"),
        ("Please do not stop the server now.", "please do stop the server now", "Please"),
        ("There are 15 files in the folder.", "there are 50 files in the folder", "There"),
        ("Go left then go left then stop.", "go left go left", "Go"),
        ("Keep your original files safe.", "keep your original file safe", "Keep your"),
        ("Please keep all files safe.", "please keep uh all files safe", "Please"),
        ("不要关闭服务然后重启机器。", "关闭服务然后重启", ""),
        ("Hello world.", "thanks for watching this video", ""),
        ("Hello world again.", "hello there world again", ""),
    ],
)
def test_first_difference_stops_matching_without_skipping_or_resuming(text, asr, expected):
    result = asr_prefix(text, asr)
    assert result.accepted, result
    assert text[: result.char_end] == expected


@pytest.mark.parametrize(
    ("text", "asr", "reason"),
    [
        ("我们明天一起去公园散步。", "我们", "insufficient_text"),
        ("One two three.", "", "insufficient_text"),
        ("", "one two three", "insufficient_text"),
        ("你好" * 300, "你好今天我们说话", "text_limit"),
        ("你好今天我们说话", "你好" * 300, "text_limit"),
    ],
)
def test_asr_refuses_short_or_oversized_evidence(text, asr, reason):
    result = asr_prefix(text, asr)
    assert not result.accepted
    assert result.reason == reason
    assert result.char_end == 0


def test_tail_guard_is_applied_to_matched_prefix_before_a_difference():
    text = "One two three four five."
    result = asr_prefix(text, "one two three other five", tail_guard_units=2)
    assert result.accepted
    assert text[: result.char_end] == "One"
    assert asr_prefix(text, "one two three", tail_guard_units=3).char_end == 0


@pytest.mark.parametrize("limits", [{"tail_guard_units": 0}, {"max_units": 0}])
def test_invalid_limits_raise(limits):
    with pytest.raises(ValueError, match="Invalid ASR prefix limits"):
        asr_prefix("One two three.", "one two three", **limits)
