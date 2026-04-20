# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Voxtral TTS demo text preprocessing helper.

These tests intentionally cover only behavior ported into
examples/online_serving/voxtral_tts/text_preprocess.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_HELPER_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "online_serving" / "voxtral_tts" / "text_preprocess.py"
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_helper():
    spec = importlib.util.spec_from_file_location("voxtral_tts_text_preprocess_demo", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["voxtral_tts_text_preprocess_demo"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


sanitize_tts_input_text_for_demo = _load_helper().sanitize_tts_input_text_for_demo


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("Hello world", "Hello world."),
        ("Hello world!", "Hello world!"),
        ("well - actually", "well \u2014 actually."),
        ("self-aware", "self-aware."),
        ("Hello\nworld", "Hello world."),
        ("Wait....", "Wait..."),
        (
            "Revenue was 1,234,567",
            "Revenue was one million two hundred thirty-four thousand five hundred and sixty-seven.",
        ),
        ("Price: $100", "Price: one hundred dollars."),
        ("Price: $100.50", "Price: one hundred dollars and fifty cents."),
        ("Price: ¥100.50", "Price: one hundred point five-zero yen."),
        ("Price: $1000000000000000000", "Price: $1000000000000000000."),
        ("Launched in (2024)", "Launched in (2024)."),
        ("The route (N/A) is unavailable", "The route\u2014not available\u2014is unavailable."),
        ("### Title\n- item one\n- item two", "Title item one item two."),
        ("See [docs](https://example.com/docs).", "See docs."),
        ("See https://example.com/docs right now", "See link right now."),
        ("A ```python\nprint(1)\n``` B", "A Code example omitted. B."),
        ("<b>Hello</b> <br> world", "Hello world."),
        ("x -- y", "x \u2014 y."),
        ("x\u200by", "xy."),
    ],
)
def test_sanitize_tts_input_text_for_demo_cases(input_text: str, expected: str) -> None:
    assert sanitize_tts_input_text_for_demo(input_text) == expected


@pytest.mark.parametrize("input_text", ["x\u200cy", "x\u200dy"])
def test_sanitize_tts_input_text_for_demo_preserves_zwj_and_zwnj(input_text: str) -> None:
    assert sanitize_tts_input_text_for_demo(input_text) == f"{input_text}."


@pytest.mark.parametrize("input_text", ["\u200b\u200b", "\n\t  "])
def test_sanitize_tts_input_text_for_demo_rejects_empty_after_sanitization(input_text: str) -> None:
    with pytest.raises(ValueError, match="Speech input is empty after sanitization"):
        sanitize_tts_input_text_for_demo(input_text)
