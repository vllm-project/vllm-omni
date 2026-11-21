# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech request-schema and model-adapter rejection contracts without model startup."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest, StreamingSpeechSessionConfig
from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REF_AUDIO_URL = "https://example.com/reference.wav"
REF_TEXT = "Reference transcript."
_AUDEX_TTA_CAPTION = "Heavy rain falling on a tin roof."
_MINIMAX_MUSIC3_LYRICS = "[Verse] Walking down the empty street at midnight"
_MINIMAX_MUSIC3_CAPTION = "A melancholic lo-fi hip-hop track."
_MINIMAX_MUSIC3_MAX_FRAMES = 9000
_YUE2_LYRICS = "A short song"
_YUE2_CAPTION = "Heavy metal, distorted guitars."
_YUE2_ABC = "X:1\nT:Twinkle\nM:4/4\nL:1/4\nK:C\nC C G G|A A G2|"
_YUE2_MAX_FRAMES = 9000


def _assert_rejected(body: dict[str, Any], model_type: str, expected: str | tuple[str, ...]) -> None:
    try:
        request = OpenAICreateSpeechRequest.model_validate(body)
    except ValidationError as exc:
        error = str(exc)
    else:
        adapter_cls = resolve_adapter(model_type)
        assert adapter_cls is not None
        # Invalid requests must be rejected without downloading weights/audio.
        server = SimpleNamespace(_apply_uploaded_speaker=lambda request: None)
        adapter = adapter_cls(SpeechServingContext(server=server))
        error = adapter.validate(request)
        assert error is not None, f"{model_type} accepted invalid request: {body}"
    for fragment in (expected,) if isinstance(expected, str) else expected:
        assert fragment.lower() in error.lower(), error


@pytest.mark.parametrize(
    "extra_json, err_message",
    [
        pytest.param({"input": "", "response_format": "wav"}, ("input", "empty"), id="input_empty"),
        pytest.param({"input": "   ", "response_format": "wav"}, ("input", "empty"), id="input_whitespace_only"),
        pytest.param(
            {"input": "Hello world.", "max_new_tokens": 0},
            ("max_new_tokens", "greater_than_equal"),
            id="max_new_tokens_below_min",
        ),
        pytest.param(
            {
                "input": "Hello world.",
                "references": [
                    {"audio_path": REF_AUDIO_URL, "text": REF_TEXT},
                    {"audio_path": REF_AUDIO_URL, "text": REF_TEXT},
                ],
                "response_format": "wav",
            },
            ("references", "only supports a single reference"),
            id="multi_reference_payload",
        ),
        pytest.param(
            {"input": "Hello world.", "references": [], "response_format": "wav"},
            ("references", "non-empty"),
            id="references_empty_array",
        ),
        pytest.param(
            {"input": "Hello world.", "references": [{"text": REF_TEXT}], "response_format": "wav"},
            ("audio_path", "required"),
            id="references_missing_audio_path",
        ),
        pytest.param(
            {"input": "Hello world.", "references": [1], "response_format": "wav"},
            ("references[0]", "must be an object"),
            id="references_bad_element_type",
        ),
        pytest.param(
            {
                "input": "Hello world.",
                "ref_audio": REF_AUDIO_URL,
                "references": [{"audio_path": "https://example.com/other.wav"}],
                "response_format": "wav",
            },
            ("mutually exclusive", "ref_audio", "references"),
            id="ref_audio_and_references_exclusive",
        ),
        pytest.param(
            {
                "input": "Hello world.",
                "ref_text": "different transcript",
                "references": [{"audio_path": REF_AUDIO_URL, "text": REF_TEXT}],
                "response_format": "wav",
            },
            ("references[0].text", "ref_text", "conflict"),
            id="references_text_conflicts_ref_text",
        ),
    ],
)
def test_speech_higgs_invalid_field_values(extra_json: dict[str, Any], err_message: str | tuple[str, ...]) -> None:
    body = {"model": "higgs_audio_v3", **extra_json}
    _assert_rejected(body, "higgs_audio_v3", err_message)


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param({"extra_params": {"cfg_scale": 99}}, "cfg_scale", id="cfg_scale_out_of_range"),
        pytest.param({"extra_params": {"cfg_scale": "big"}}, "cfg_scale", id="cfg_scale_wrong_type"),
        pytest.param({"extra_params": {"tta_rvq": {}}}, "tta_rvq", id="tta_rvq_not_overridable"),
        pytest.param({"input": "   "}, "caption", id="caption_whitespace_only"),
        pytest.param({"voice": "alloy"}, "voice", id="voice_not_supported"),
        pytest.param(
            {"ref_audio": "https://example.com/ref.wav"},
            "reference audio",
            id="ref_audio_not_supported",
        ),
    ],
)
def test_speech_audex_tta_invalid_field_values(overrides: dict[str, Any], err_message: str | tuple[str, ...]) -> None:
    body = {"model": "audex_tta", "input": _AUDEX_TTA_CAPTION, **overrides}
    _assert_rejected(body, "audex_tta", err_message)


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param({"instructions": None}, ("instructions",), id="caption_missing"),
        pytest.param({"instructions": "   "}, ("instructions",), id="caption_blank"),
        pytest.param({"input": "   "}, ("input",), id="lyrics_blank"),
        pytest.param({"voice": "alloy"}, ("voice",), id="voice_rejected"),
        pytest.param({"speed": 1.5}, ("speed",), id="speed_rejected"),
        pytest.param(
            {"max_new_tokens": _MINIMAX_MUSIC3_MAX_FRAMES + 1},
            ("max_new_tokens", str(_MINIMAX_MUSIC3_MAX_FRAMES)),
            id="max_new_tokens_over_cap",
        ),
        pytest.param({"max_new_tokens": 0}, ("max_new_tokens",), id="max_new_tokens_zero"),
    ],
)
def test_minimax_music3_speech_invalid_field_values(
    overrides: dict[str, Any], err_message: str | tuple[str, ...]
) -> None:
    body: dict[str, Any] = {
        "model": "minimax_music3",
        "input": _MINIMAX_MUSIC3_LYRICS,
        "instructions": _MINIMAX_MUSIC3_CAPTION,
        "seed": 1,
        "max_new_tokens": 250,
        "response_format": "wav",
    }
    for key, value in overrides.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    _assert_rejected(body, "minimax_music3", err_message)


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param({"instructions": None}, ("instructions",), id="caption_missing"),
        pytest.param({"instructions": "   "}, ("instructions",), id="caption_blank"),
        pytest.param({"input": "   "}, ("input",), id="lyrics_blank"),
        pytest.param({"voice": "alloy"}, ("voice",), id="voice_rejected"),
        pytest.param({"speed": 1.5}, ("speed",), id="speed_rejected"),
        pytest.param(
            {"max_new_tokens": _YUE2_MAX_FRAMES + 1},
            ("max_new_tokens", str(_YUE2_MAX_FRAMES)),
            id="max_new_tokens_over_cap",
        ),
        pytest.param({"max_new_tokens": 199}, ("max_new_tokens",), id="max_new_tokens_below_min"),
        pytest.param(
            {"extra_params": {"cot": "melody", "abc": " "}},
            ("abc",),
            id="cot_melody_with_blank_abc",
        ),
        pytest.param(
            {"extra_params": {"abc": _YUE2_ABC}},
            ("cot",),
            id="abc_without_cot",
        ),
        pytest.param(
            {"extra_params": {"temperature": 0.3}},
            ("temperature",),
            id="temperature_rejected",
        ),
    ],
)
def test_yue2_speech_invalid_field_values(overrides: dict[str, Any], err_message: str | tuple[str, ...]) -> None:
    body: dict[str, Any] = {
        "model": "yue2",
        "input": _YUE2_LYRICS,
        "instructions": _YUE2_CAPTION,
        "seed": 1,
        "max_new_tokens": 250,
        "response_format": "wav",
    }
    for key, value in overrides.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    _assert_rejected(body, "yue2", err_message)


@pytest.mark.parametrize("model_type", ["higgs_audio_v3", "audex_tta", "minimax_music3", "yue2"])
def test_valid_model_parameters_are_accepted(model_type: str) -> None:
    body: dict[str, Any] = {
        "model": model_type,
        "input": "A short caption or lyrics",
        "instructions": "A quiet acoustic song",
        "max_new_tokens": 250,
    }
    if model_type == "yue2":
        # ABC is optional with cot=melody: the model can generate its own score.
        body["extra_params"] = {"cot": "melody"}
    request = OpenAICreateSpeechRequest.model_validate(body)
    adapter_cls = resolve_adapter(model_type)
    assert adapter_cls is not None
    server = SimpleNamespace(_apply_uploaded_speaker=lambda request: None)
    assert adapter_cls(SpeechServingContext(server=server)).validate(request) is None


def test_speech_stream_allows_wav_and_rejects_mp3() -> None:
    config = StreamingSpeechSessionConfig.model_validate({"stream_audio": True, "response_format": "wav"})
    assert config.response_format == "wav" and config.stream_audio
    with pytest.raises(ValidationError, match="requires response_format"):
        StreamingSpeechSessionConfig.model_validate({"stream_audio": True, "response_format": "mp3"})
