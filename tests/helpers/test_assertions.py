# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.helpers.assertions import _assert_transcript_matches


def test_short_transcript_repeat_passes_containment_fallback():
    _assert_transcript_matches(
        " How... how are you?",
        audio_bytes=None,
        expected_text="how are you",
        threshold=0.9,
    )


def test_short_transcript_unrelated_text_still_fails():
    with pytest.raises(AssertionError, match="Transcript doesn't match input"):
        _assert_transcript_matches(
            " I don't know, sorry.",
            audio_bytes=None,
            expected_text="how are you",
            threshold=0.9,
        )
