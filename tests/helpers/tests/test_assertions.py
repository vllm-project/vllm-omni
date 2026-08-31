# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from tests.helpers import assertions
from tests.helpers.assertions import (
    _assert_transcript_matches,
    _resolve_audio_transcript,
    assert_audio_speech_response,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def _capture_transcribe(monkeypatch) -> dict:
    captured: dict = {}

    def fake_convert(raw_bytes, model_size="small", language=None):
        captured["model_size"] = model_size
        captured["language"] = language
        return "London"

    monkeypatch.setattr(assertions, "convert_audio_bytes_to_text", fake_convert)
    return captured


def test_resolve_transcript_honors_declared_language(monkeypatch):
    captured = _capture_transcribe(monkeypatch)
    response = SimpleNamespace(audio_content=None, audio_bytes=b"fake-wav")

    _resolve_audio_transcript(
        response,
        {"modalities": ["text", "audio"], "transcript_language": "en"},
        "full_model",
        speech_api=False,
    )

    assert captured["language"] == "en"


def test_resolve_transcript_leaves_language_unset_by_default(monkeypatch):
    captured = _capture_transcribe(monkeypatch)
    response = SimpleNamespace(audio_content=None, audio_bytes=b"fake-wav")

    _resolve_audio_transcript(
        response,
        {"modalities": ["text", "audio"]},
        "full_model",
        speech_api=False,
    )

    assert captured["language"] is None
    assert captured["model_size"] == "small"


@pytest.mark.parametrize(
    ("request_config", "expected_text"),
    [
        ({"input": "spoken words"}, "spoken words"),
        (
            {
                "input": "<|style:whispering|>spoken words",
                "transcript_expected_text": "spoken words",
            },
            "spoken words",
        ),
    ],
)
def test_speech_transcript_expected_text(monkeypatch, request_config, expected_text):
    captured: dict = {}

    monkeypatch.setattr(assertions, "convert_audio_bytes_to_text", lambda *_args, **_kwargs: "spoken words")

    def capture_match(_transcript, _audio_bytes, expected_text, **_kwargs):
        captured["expected_text"] = expected_text

    monkeypatch.setattr(assertions, "_assert_transcript_matches", capture_match)
    response = SimpleNamespace(success=True, audio_bytes=b"fake-wav", audio_format="audio/wav")
    request_config["response_format"] = "wav"

    assert_audio_speech_response(
        response,
        request_config,
        "advanced_model",
    )

    assert captured["expected_text"] == expected_text


def test_escalated_transcript_keeps_declared_language(monkeypatch):
    # The escalated pass must honour the same language, otherwise a request that
    # pins one silently falls back to auto-detection on retry.
    captured = _capture_transcribe(monkeypatch)

    with pytest.raises(AssertionError, match="after ASR escalation"):
        _assert_transcript_matches(
            "totally different words",
            audio_bytes=b"fake-wav",
            expected_text="how are you",
            threshold=0.9,
            escalation_model="large-v3",
            language="en",
        )

    assert captured["model_size"] == "large-v3"
    assert captured["language"] == "en"
