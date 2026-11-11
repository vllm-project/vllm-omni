# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the custom voice file-name helper."""

import pytest

from vllm_omni.utils.custom_voice_io import safe_voice_stem

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("alice", "alice", id="plain"),
        pytest.param("Voice_01-v2.final", "Voice_01-v2.final", id="allowed-punctuation"),
        pytest.param("  bob  ", "bob", id="surrounding-spaces"),
        pytest.param("my voice", "my_voice", id="inner-space"),
        pytest.param("a@#$b", "a_b", id="symbol-run"),
        pytest.param("voiceé", "voice_", id="non-ascii"),
        pytest.param("/", "_", id="slash-only"),
        pytest.param("a/b", "a_b", id="slash"),
        pytest.param("a\\b", "a_b", id="backslash"),
        pytest.param("../etc/passwd", ".._etc_passwd", id="parent-traversal"),
        pytest.param("/abs/path", "_abs_path", id="absolute-path"),
        pytest.param("a" * 250, "a" * 200, id="capped-at-200"),
        pytest.param("a" * 200, "a" * 200, id="at-cap"),
    ],
)
def test_safe_voice_stem(name, expected):
    assert safe_voice_stem(name) == expected


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "  ..  "])
def test_empty_or_dot_names_are_rejected(name):
    with pytest.raises(ValueError, match="Invalid voice name"):
        safe_voice_stem(name)
