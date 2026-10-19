# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression cases for streamed Ming speech preprocessing."""

import pytest

from vllm_omni.model_executor.models.ming_flash_omni.front_toolkit import strip_web_markup
from vllm_omni.model_executor.models.ming_flash_omni.native_text_frontend import (
    NativeMingTextFrontend,
    normalize_native_fragment,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("价格 $12.50，公式 $x_1^2$，继续。", "价格 $12.50，公式  ，继续。"),
        ("<p>可以<b>朗读</b></p><script>secret()</script>", " 可以 朗读   "),
        ("`voice_name` 和 ```python\nsecret()\n```正文。", "voice_name 和  正文。"),
    ],
)
def test_markup_partition_independence(text, expected):
    # Arbitrary model token boundaries must not turn prices into math, or
    # release script/code bodies into the spoken response.
    from vllm_omni.model_executor.models.ming_flash_omni.front_toolkit import WebMarkupScrubber

    for width in (1, 3, len(text)):
        scrubber = WebMarkupScrubber()
        actual = "".join(scrubber.feed(text[i : i + width]) for i in range(0, len(text), width))
        actual += scrubber.feed("", final=True)
        assert actual == expected
    assert strip_web_markup(text).strip() == " ".join(expected.split())


@pytest.mark.parametrize("text", ["こんにちは", "안녕하세요"])
def test_unicode_terminal_tail_is_preserved(text):
    frontend = NativeMingTextFrontend()
    assert frontend.feed(text) == []
    fragments = frontend.feed("", final=True)
    assert fragments == [text + "."]
    normalized = normalize_native_fragment(fragments[0], tn_mode="legacy")
    assert normalized.strip()
    assert normalized.startswith(text)
    with pytest.raises(ValueError, match="finalized"):
        frontend.feed("more")


def test_streamed_fragments_match_complete_text_across_decimal_and_markup_boundaries():
    text = "Chinese\t现在价格是 $12.50，需要认真说明一下。<script>secret()</script>还会保留剩余的正常内容。"
    complete = NativeMingTextFrontend().feed(text, final=True)
    for width in (1, 7):
        frontend = NativeMingTextFrontend()
        fragments = []
        for start in range(0, len(text), width):
            fragments.extend(frontend.feed(text[start : start + width]))
        fragments.extend(frontend.feed("", final=True))
        assert fragments == complete
        assert "secret" not in "".join(fragments)
        assert "$12.50" in "".join(fragments)


def test_markup_filter_is_independent_of_linguistic_normalization():
    assert NativeMingTextFrontend().feed("<script>secret()</script>", final=True) == []
    fragments = NativeMingTextFrontend(scrub_markup=False).feed("<b>visible</b>", final=True)
    assert "<b>visible</b>" in "".join(fragments)


def test_legacy_normalization_does_not_run_upstream_rules(monkeypatch):
    from vllm_omni.model_executor.models.ming_flash_omni import text_processing

    monkeypatch.setattr(text_processing, "normalize_numbers", lambda text: text.replace("12", "twelve"))

    def unexpected_tn(text):
        pytest.fail("Unexpected upstream TN")

    assert (
        normalize_native_fragment("There are 12 apples.", tn_mode="legacy", normalizer=unexpected_tn)
        == "There are twelve apples."
    )


def test_long_fragment_keeps_url_and_abbreviation_lexemes_intact():
    text = "甲" * 60 + " 请访问 https://example.com/path 查看详情，Dr. Smith 来自 U.S.A.。"
    fragments = NativeMingTextFrontend(max_length=50).feed(text, final=True)
    spoken = " ".join(fragments)
    assert "https://example.com/path" in spoken
    assert "Dr." in spoken
    assert "U.S.A." in spoken
    assert "example。" not in spoken


def test_numeric_terminal_tail_does_not_gain_sentence_punctuation():
    # A synthetic dot changes the acoustic realization of a short numeric
    # answer. Preserve both bare numbers and punctuation supplied by the user.
    for text in ("0", "12", "12.5", "-12.5", "0.", "12.5!", "12.5，"):
        frontend = NativeMingTextFrontend()
        for char in text:
            assert frontend.feed(char) == []
        assert frontend.feed("", final=True) == [text]
