# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.helpers import assertions
from tests.helpers.assertions import (
    _assert_transcript_matches,
    _resolve_audio_transcript,
    assert_images_generations_response,
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


def test_images_generations_response_decodes_expected_size():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), "green").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    assert_images_generations_response(
        {"data": [{"b64_json": encoded}]},
        {
            "json": {
                "n": 1,
                "size": "32x16",
                "response_format": "b64_json",
            }
        },
    )
