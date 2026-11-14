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


def test_leading_digit_bound_to_unit_or_date_keeps_no_comma():
    # [important regression, linyueqian]: the leading rule must only fire for
    # STANDALONE edge numbers. A digit bound to a unit/date/time/measure char
    # is a normal numeric expression; the inserted comma garbles or does not
    # help (measured: "3点到" reads worse with the comma, n=5; the dates and
    # magnitudes below read perfectly without it, n=3 each).
    assert shape_voxcpm2_edge_digits("1月1日开会") == "1月1日开会"
    assert shape_voxcpm2_edge_digits("3点到") == "3点到"
    assert shape_voxcpm2_edge_digits("10月份见") == "10月份见"
    assert shape_voxcpm2_edge_digits("2万元已到账") == "2万元已到账"
    assert shape_voxcpm2_edge_digits("5个苹果") == "5个苹果"
    assert shape_voxcpm2_edge_digits("3斤苹果") == "3斤苹果"


def test_leading_standalone_digit_run_still_gets_comma():
    # the designed shapes (digit followed by a non-bound Han char) keep the pad
    assert shape_voxcpm2_edge_digits("10今天天气真好") == "10，今天天气真好"
    assert shape_voxcpm2_edge_digits("1今天天气真好") == "1，今天天气真好"
    assert shape_voxcpm2_edge_digits("48今天天气真好") == "48，今天天气真好"
    assert shape_voxcpm2_edge_digits("0今天天气真好") == "0，今天天气真好"


def test_lone_trailing_zero_is_sentinel_held():
    out = shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。0。")
    assert out.endswith("\uee00。")


def test_lone_trailing_zero_before_final_newline_keeps_the_zero():
    # "$" also matches before a final newline: the matched "0。" here is NOT
    # the last two characters ("。\n" is), so a text[:-2] slice dropped the
    # wrong span and re-appended the stop -> "今天00。" with TN off.
    out = shape_voxcpm2_edge_digits("今天天气真好，一起去吃饭。0。\n")
    assert out == "今天天气真好，一起去吃饭。\uee00。\n"
    # and the restore path (TN off) puts back exactly one 0
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import normalize_voxcpm2_text

    monkey_text = out
    import os

    os.environ["VLLM_OMNI_SPEECH_TEXT_NORMALIZE"] = "0"
    assert normalize_voxcpm2_text(monkey_text, None) == "今天天气真好，一起去吃饭。0。\n"


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


def test_normalize_with_unavailable_normalizer_keeps_sentinel(monkeypatch):
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import normalize_voxcpm2_text

    monkeypatch.setenv("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "1")
    # the cached-None normalizer (import/construct failure) must degrade to
    # the documented TN-off path: no crash, sentinel still applied+restored
    shaped = shape_voxcpm2_edge_digits("10今天天气真好，一起去吃饭。10")
    out = normalize_voxcpm2_text(shaped, None)
    assert "10" in out and "" not in out  # sentinel round-trips away


def test_normalize_none_guard_with_tn_trigger(monkeypatch):
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import normalize_voxcpm2_text

    monkeypatch.setenv("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "1")
    # the regression: a numeric string + None normalizer used to AttributeError
    out = normalize_voxcpm2_text("10，今天天气真好", None)
    assert out == "10，今天天气真好"


def test_normalize_off_switch_skips_normalizer(monkeypatch):
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import normalize_voxcpm2_text

    class _Boom:
        def normalize(self, t):
            raise AssertionError("normalizer must not be called when TN is off")

    monkeypatch.setenv("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "0")
    assert normalize_voxcpm2_text("10，今天", _Boom()) == "10，今天"


def test_normalize_calls_normalizer_when_on(monkeypatch):
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import normalize_voxcpm2_text

    class _Stub:
        def __init__(self):
            self.calls = []

        def normalize(self, t):
            self.calls.append(t)
            return t.replace("10", "十")

    monkeypatch.setenv("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "1")
    stub = _Stub()
    out = normalize_voxcpm2_text("10，今天天气真好", stub)
    assert stub.calls == ["10，今天天气真好"]
    assert out == "十，今天天气真好"


def test_normalize_sentinel_survives_normalizer_roundtrip(monkeypatch):
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2_digits import normalize_voxcpm2_text

    # a lone leading zero is sentinel-held into TN and restored after.
    # The extra "3+5" keeps a TN trigger alive after the sentinel removes
    # the leading zero - the original single-number input made this a
    # no-op (the sentinel removed the only TN trigger, so _Stub.normalize
    # was never called and the test passed vacuously).
    class _Stub:
        def __init__(self):
            self.calls = []

        def normalize(self, t):
            self.calls.append(t)
            return t.replace("0", "零").replace("3+5", "三加五")  # would eat a bare leading 0

    monkeypatch.setenv("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "1")
    shaped = shape_voxcpm2_edge_digits("0今天天气真好，3+5=8")
    stub = _Stub()
    out = normalize_voxcpm2_text(shaped, stub)
    assert stub.calls, "the normalizer must actually run (a TN trigger must survive the sentinel)"
    assert out.startswith("0，"), f"sentinel must protect the leading 0 through TN, got {out!r}"


def test_bare_digit_text_gets_full_stop():
    # ZhIMQ's case: the whole text IS a 1-2 digit run
    assert shape_voxcpm2_edge_digits("10") == "10。"
    assert shape_voxcpm2_edge_digits("7") == "7。"
