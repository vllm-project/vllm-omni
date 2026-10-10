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
        ("今天下午开会请提前到场。", "今天下午开慧请提前到", "今天下午开会请提前"),
    ],
)
def test_asr_selects_original_prefix_and_withholds_unverified_tail(text, asr, expected):
    result = asr_prefix(text, asr)
    assert result.accepted, result
    assert text[: result.char_end] == expected


@pytest.mark.parametrize(
    ("text", "asr"),
    [
        ("不要关闭服务然后重启机器。", "关闭服务然后重启"),
        ("Please do not stop the server now.", "please do stop the server now"),
        ("There are 15 files in the folder.", "there are 50 files in the folder"),
        ("我们明天一起去公园散步。", "我们"),
        ("Go left then go left then stop.", "go left go left"),
        ("Hello world.", "thanks for watching this video"),
    ],
)
def test_asr_refuses_sensitive_ambiguous_or_short_evidence(text, asr):
    result = asr_prefix(text, asr)
    assert not result.accepted
    assert result.char_end == 0


def test_long_inputs_refuse_before_allocating_alignment_matrix():
    assert asr_prefix("你好" * 300, "你好今天我们说话").reason == "text_limit"
