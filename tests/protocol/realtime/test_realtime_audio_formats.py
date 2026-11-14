# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime audio-format negotiation: the spellings a client may use, and the
precedence the validators apply when a session, a response and a conversation
item each carry their own declaration.

The behaviour worth pinning is not the happy path but the three seams:
a format may arrive as a string or as an object, the same format has several
accepted spellings, and when a payload carries both a flat key and a nested
``audio.{input,output}.format`` the flat key wins. Drift here is silent: an
unrecognised spelling is simply passed through and later read as unsupported.
"""

from __future__ import annotations

import pytest

from vllm_omni.protocol.realtime.formats import (
    REALTIME_INPUT_AUDIO_FORMATS,
    REALTIME_OUTPUT_AUDIO_FORMATS,
    is_supported_realtime_input_format,
    parse_realtime_audio_format,
    realtime_audio_format_object,
    realtime_output_format,
    validate_conversation_item_audio_formats,
    validate_realtime_response_audio_formats,
    validate_realtime_session_audio_formats,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CANONICAL_SPELLINGS = [
    ("pcm16", "pcm16"),
    ("PCM16", "pcm16"),
    ("audio/pcm", "pcm16"),
    ("pcm", "pcm16"),
    ("audio/pcm16", "pcm16"),
    ("pcm_s16le", "pcm16"),
    ("s16le", "pcm16"),
    ("audio/pcm_f32le", "pcm_f32le"),
    ("pcm_f32le", "pcm_f32le"),
    ("f32le", "pcm_f32le"),
    ("audio/g711_ulaw", "g711_ulaw"),
    ("g711-ulaw", "g711_ulaw"),
    ("ulaw", "g711_ulaw"),
    ("mulaw", "g711_ulaw"),
    ("audio/g711_alaw", "g711_alaw"),
    ("g711-alaw", "g711_alaw"),
    ("alaw", "g711_alaw"),
    ("wav", "wav"),
    ("audio/wav", "wav"),
]


@pytest.mark.parametrize(("spelling", "canonical"), _CANONICAL_SPELLINGS)
def test_a_string_format_resolves_to_its_canonical_name(spelling: str, canonical: str) -> None:
    parsed, rate = parse_realtime_audio_format(spelling)

    assert parsed == canonical
    assert rate is None


def test_an_unrecognised_format_is_passed_through_with_its_own_case() -> None:
    assert parse_realtime_audio_format("MP3") == ("MP3", None)


@pytest.mark.parametrize("rate_key", ["rate", "sample_rate_hz", "sample_rate"])
def test_a_format_object_reads_the_rate_from_any_accepted_key(rate_key: str) -> None:
    assert parse_realtime_audio_format({"type": "audio/pcm", rate_key: 16000}) == ("pcm16", 16000)


@pytest.mark.parametrize("rate", [0, -1, 0.0, "16000", None])
def test_a_missing_usable_rate_is_reported_as_none(rate: object) -> None:
    assert parse_realtime_audio_format({"type": "pcm16", "rate": rate}) == ("pcm16", None)


def test_a_format_object_accepts_format_instead_of_type() -> None:
    assert parse_realtime_audio_format({"format": "s16le", "rate": 8000}) == ("pcm16", 8000)


def test_a_format_object_without_a_string_type_is_returned_untouched() -> None:
    payload = {"rate": 16000}

    parsed, rate = parse_realtime_audio_format(payload)

    assert parsed is payload
    assert rate == 16000


@pytest.mark.parametrize("raw", [None, 42, ["pcm16"]])
def test_a_non_string_non_mapping_format_is_returned_untouched(raw: object) -> None:
    assert parse_realtime_audio_format(raw) == (raw, None)


def test_the_output_format_maps_pcm_to_the_pcm16_codec_and_defaults_to_wav() -> None:
    assert realtime_output_format("pcm") == "pcm16"
    assert realtime_output_format("g711_ulaw") == "g711_ulaw"
    assert realtime_output_format("G711_ALAW") == "g711_alaw"
    assert realtime_output_format("wav") == "wav"
    assert realtime_output_format(None) == "wav"
    assert realtime_output_format("") == "wav"


def test_the_supported_input_predicate_reads_canonical_names_case_insensitively() -> None:
    assert is_supported_realtime_input_format("pcm16")
    assert is_supported_realtime_input_format("G711_ULAW")
    assert not is_supported_realtime_input_format("opus")
    assert not is_supported_realtime_input_format(None)


def test_a_format_object_is_built_from_the_canonical_type_and_an_optional_rate() -> None:
    assert realtime_audio_format_object("pcm16") == {"type": "audio/pcm"}
    assert realtime_audio_format_object("s16le", sample_rate_hz=24000) == {"type": "audio/pcm", "rate": 24000}
    assert realtime_audio_format_object("pcm_f32le") == {"type": "audio/pcm_f32le"}
    assert realtime_audio_format_object("g711_ulaw") == {"type": "audio/g711_ulaw"}
    assert realtime_audio_format_object("g711_alaw") == {"type": "audio/g711_alaw"}
    assert realtime_audio_format_object("opus") == {"type": "audio/wav"}


def test_a_session_without_an_audio_declaration_is_accepted() -> None:
    assert validate_realtime_session_audio_formats({}) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"input_audio_format": "pcm16", "output_audio_format": "wav"},
        {"audio": {"input": {"format": "s16le"}, "output": {"format": "pcm"}}},
        {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}},
    ],
)
def test_a_session_may_declare_formats_flat_or_nested_or_as_an_object(payload: dict) -> None:
    assert validate_realtime_session_audio_formats(payload) is None


def test_a_session_flat_key_wins_over_the_nested_object() -> None:
    payload = {
        "input_audio_format": "opus",
        "audio": {"input": {"format": "pcm16"}},
    }

    assert validate_realtime_session_audio_formats(payload) == "Unsupported input_audio_format: opus"


def test_a_session_rejects_an_unsupported_input_or_output_format() -> None:
    assert (
        validate_realtime_session_audio_formats({"audio": {"input": {"format": "opus"}}})
        == "Unsupported input_audio_format: opus"
    )
    assert (
        validate_realtime_session_audio_formats({"output_audio_format": "mp3"})
        == "Unsupported output_audio_format: mp3"
    )


def test_a_consumer_may_narrow_the_accepted_session_sets() -> None:
    payload = {"input_audio_format": "g711_ulaw"}

    assert validate_realtime_session_audio_formats(payload) is None
    assert validate_realtime_session_audio_formats(payload, input_audio_formats={"pcm16"}) is not None


def test_a_response_may_not_declare_an_output_format_outside_the_output_set() -> None:
    assert validate_realtime_response_audio_formats({"output_audio_format": "wav"}) is None
    assert validate_realtime_response_audio_formats({"audio": {"output": {"format": "pcm"}}}) is None
    assert (
        validate_realtime_response_audio_formats({"response_format": "mp3"}) == "Unsupported output_audio_format: mp3"
    )


def test_a_conversation_item_is_scanned_only_for_its_audio_parts() -> None:
    text_part = {"type": "input_text", "format": "opus"}

    assert validate_conversation_item_audio_formats({"content": [text_part]}) is None
    assert (
        validate_conversation_item_audio_formats({"content": [{"type": "input_audio", "format": "opus"}]})
        == "Unsupported input_audio format in conversation.item.create: opus"
    )
    assert (
        validate_conversation_item_audio_formats({"content": [{"type": "audio", "format": {"type": "g711_ulaw"}}]})
        is None
    )


@pytest.mark.parametrize("item", [None, "audio", {"content": "not-a-list"}, {"content": [None, 3]}])
def test_a_conversation_item_of_an_unexpected_shape_is_ignored(item: object) -> None:
    assert validate_conversation_item_audio_formats(item) is None


def test_the_declared_sets_are_lowercase_and_agree_with_the_validators() -> None:
    assert all(name == name.lower() for name in REALTIME_INPUT_AUDIO_FORMATS | REALTIME_OUTPUT_AUDIO_FORMATS)
    assert "pcm16" in REALTIME_INPUT_AUDIO_FORMATS
    assert "wav" in REALTIME_OUTPUT_AUDIO_FORMATS
