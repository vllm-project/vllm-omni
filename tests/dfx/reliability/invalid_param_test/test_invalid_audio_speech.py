# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP + WebSocket validation for Qwen3-TTS Base and Higgs Audio V3 ``/v1/audio/speech`` (and related routes)."""

from __future__ import annotations

import io
import json
import uuid
import wave
from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from tests.helpers.stage_config import get_deploy_config_path

# Speech / voice-upload numeric caps for these tests only (not exported from vllm_omni).
# Align where possible with serving_speech defaults: _TTS_MAX_INSTRUCTIONS_LENGTH (500),
# _TTS_MAX_NEW_TOKENS_MAX (4096). Voice form max lengths are not public constants in prod;
# values below keep tests self-contained and deterministic.
_SPEECH_API_MAX_INSTRUCTIONS_CHARS = 500
_SPEECH_API_MAX_NEW_TOKENS = 4096
_VOICE_UPLOAD_MAX_CONSENT_LEN = 1024
_VOICE_UPLOAD_MAX_REF_TEXT_CHARS = 8192
_VOICE_UPLOAD_MAX_SPEAKER_DESCRIPTION_CHARS = 2048
REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
# Qwen3-TTS 0.6B expects 1024-dim speaker embeddings (see serving_speech / talker hidden_size).
_QWEN3_TTS_06B_SPEAKER_EMBEDDING_DIM = 1024
# Pop-only marker for ``test_speech_invalid_field_values``: omit ``ref_audio`` / ``ref_text`` so Base
# ``speaker_embedding`` cases mirror standalone embedding-only JSON (no ref_audio clash).
_SPEECH_INVALID_EMBEDDING_ONLY_BODY = "__speech_invalid_embedding_only_body__"
# Batch-only: merged items inherit batch ``ref_audio`` / ``ref_text`` because ``_pick`` treats item ``None``
# as "unset". Strip clone fields from the batch envelope before asserting CustomVoice preset errors.
_SPEECH_BATCH_NO_CLONE_FIELDS = "__speech_batch_no_clone_fields__"

_SKIP_ISSUE_3649 = pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/3649")


pytestmark = [pytest.mark.slow, pytest.mark.tts]

_L4_SPEECH_HW = hardware_marks(res={"cuda": "L4"})
_SPEECH_SERVER_ARGS = ["--trust-remote-code", "--disable-log-stats"]

_QWEN3_TTS_SPEECH = [
    pytest.param(
        OmniServerParams(
            model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=_SPEECH_SERVER_ARGS,
        ),
        id="qwen3_tts",
        marks=_L4_SPEECH_HW,
    ),
]

_HIGGS_AUDIO_V3_SPEECH = [
    pytest.param(
        OmniServerParams(
            model="bosonai/higgs-audio-v3-tts-4b",
            stage_config_path=get_deploy_config_path("higgs_multimodal_qwen3.yaml"),
            server_args=_SPEECH_SERVER_ARGS,
            env_dict={"VLLM_USE_DEEP_GEMM": "0", "VLLM_MOE_USE_DEEP_GEMM": "0"},
        ),
        id="higgs_audio_v3",
        marks=_L4_SPEECH_HW,
    ),
]


def _apply_batch_overrides(body: dict[str, object], *, loc: str, overrides: dict[str, object]) -> None:
    if loc == "batch":
        body.update(overrides)
        return
    items = body["items"]
    assert isinstance(items, list) and len(items) >= 1
    first = items[0]
    assert isinstance(first, dict)
    body["items"] = [{**first, **overrides}]


def _pcm_wav_mono_bytes(*, duration_s: float = 1.1, sample_rate: int = 44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        nframes = max(1, int(sample_rate * duration_s))
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/audio/speech (JSON)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_malformed_json(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_audio_speech_http_request(
        {
            "raw_body": "{",
            "timeout": 120,
            "err_code": 400,
            "err_message": ("json", "decode", "invalid"),
        }
    )


@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_missing_required_fields(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_audio_speech_http_request(
        {
            "json": {"model": omni_server.model},
            "timeout": 120,
            "err_code": 400,
            "err_message": ("input", "field required", "missing"),
        }
    )


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param({"input": ""}, ("input", "empty"), id="input_empty"),
        pytest.param({"input": "   "}, ("input", "empty"), id="input_whitespace_only"),
        pytest.param({"voice": ""}, "Invalid voice", id="voice_empty", marks=_SKIP_ISSUE_3649),
        pytest.param(
            {"instructions": 123}, ("instructions", "string_type", "valid string"), id="instructions_wrong_type"
        ),
        pytest.param(
            {"response_format": "mpeg"}, ("response_format", "literal_error", "wav"), id="response_format_invalid"
        ),
        pytest.param({"stream_format": 1}, ("stream_format", "literal_error", "audio"), id="stream_format_wrong_type"),
        pytest.param({"stream": "wrong_type"}, ("stream", "bool_parsing", "validation error"), id="stream_wrong_type"),
        pytest.param(
            {
                "voice": "vivian",
                "task_type": "CustomVoice",
                "ref_audio": None,
                "ref_text": None,
            },
            ("CustomVoice", "does not support"),
            id="customvoice_task_not_supported",
        ),
        pytest.param(
            {"task_type": "InvalidEnum"}, ("task_type", "literal_error", "CustomVoice"), id="task_type_invalid"
        ),
        pytest.param({"language": ""}, ("language", "invalid language", "chinese"), id="language_empty"),
        pytest.param({"ref_audio": "ftp://example.com/a.wav"}, ("ref_audio", "URL"), id="ref_audio_bad_scheme"),
        pytest.param(
            {"ref_audio": "not_a_valid_uri"},
            ("ref_audio", "url"),
            id="ref_audio_invalid_uri",
        ),
        pytest.param({"ref_text": ["x"]}, ("ref_text", "string_type", "valid string"), id="ref_text_wrong_type"),
        pytest.param(
            {"x_vector_only_mode": "wrong"},
            ("x_vector_only_mode", "valid boolean"),
            id="x_vector_only_mode_string_not_boolean",
        ),
        pytest.param(
            {"speaker_embedding": [0.01] * _QWEN3_TTS_06B_SPEAKER_EMBEDDING_DIM},
            ("mutually exclusive", "speaker_embedding", "ref_audio"),
            id="speaker_embedding_with_ref_audio_mutually_exclusive",
        ),
        pytest.param({"max_new_tokens": 0}, ("max_new_tokens", "least 1"), id="max_new_tokens_below_min"),
        pytest.param(
            {"max_new_tokens": _SPEECH_API_MAX_NEW_TOKENS + 1},
            ("max_new_tokens", "exceed 4096"),
            id="max_new_tokens_above_max",
        ),
        pytest.param({"seed": -1}, ("seed", "greater_than_equal", "0"), id="seed_negative"),
        pytest.param({"seed": 2**63}, ("seed", "less_than_equal", "9223372036854775807"), id="seed_above_max"),
        pytest.param(
            {"initial_codec_chunk_frames": -1},
            ("initial_codec_chunk_frames", "greater_than_equal", "0"),
            id="initial_codec_chunk_frames_negative",
        ),
        pytest.param(
            {"instructions": "x" * (_SPEECH_API_MAX_INSTRUCTIONS_CHARS + 1)},
            ("instructions", "too long"),
            id="instructions_exceed_api_limit",
        ),
        # Base clone: whitespace-only ref_text fails serving validation (not pydantic).
        pytest.param(
            {
                "task_type": "Base",
                "ref_audio": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
                "ref_text": "   ",
            },
            ("ref_text", "base task", "non-empty"),
            id="ref_text_whitespace_only_base_clone",
        ),
        pytest.param(
            {
                _SPEECH_INVALID_EMBEDDING_ONLY_BODY: True,
                "task_type": "Base",
                "speaker_embedding": [],
            },
            ("speaker_embedding", "non-empty"),
            id="base_speaker_embedding_empty",
        ),
        pytest.param(
            {
                _SPEECH_INVALID_EMBEDDING_ONLY_BODY: True,
                "task_type": "Base",
                "x_vector_only_mode": True,
                "speaker_embedding": [0.01] * 512,
            },
            ("speaker_embedding", "dimensions", str(_QWEN3_TTS_06B_SPEAKER_EMBEDDING_DIM)),
            id="base_speaker_embedding_wrong_dim_short",
        ),
        pytest.param(
            {
                _SPEECH_INVALID_EMBEDDING_ONLY_BODY: True,
                "task_type": "Base",
                "x_vector_only_mode": True,
                "speaker_embedding": [0.01] * 2048,
            },
            ("speaker_embedding", "2048", "dimensions", str(_QWEN3_TTS_06B_SPEAKER_EMBEDDING_DIM)),
            id="base_speaker_embedding_wrong_dim_long",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_invalid_field_values(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    overrides: dict[str, object],
    err_message: str | tuple[str, ...],
) -> None:
    ov = dict(overrides)
    embedding_only = bool(ov.pop(_SPEECH_INVALID_EMBEDDING_ONLY_BODY, False))
    body = {
        "model": omni_server.model,
        "input": "Hello.",
        "voice": "clone",
        "language": "English",
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    if embedding_only:
        body.pop("ref_audio", None)
        body.pop("ref_text", None)
    body.update(ov)
    openai_client.send_audio_speech_http_request(
        {"json": body, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


# Higgs Audio V3: ``references[]`` cookbook alias and serving-layer 400 rejections.
# Raw HTTP keeps JSON bodies intact (OpenAI SDK strips unknown fields such as ``references``).
@pytest.mark.parametrize(
    "extra_json, err_message",
    [
        pytest.param({"input": "", "response_format": "wav"}, ("input", "empty"), id="input_empty"),
        pytest.param({"input": "   ", "response_format": "wav"}, ("input", "empty"), id="input_whitespace_only"),
        pytest.param(
            {"input": "Hello world.", "max_new_tokens": 0}, ("max_new_tokens", "least 1"), id="max_new_tokens_below_min"
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
@pytest.mark.parametrize("omni_server", _HIGGS_AUDIO_V3_SPEECH, indirect=True)
def test_speech_higgs_invalid_field_values(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    extra_json: dict[str, object],
    err_message: str | tuple[str, ...],
) -> None:
    body = {"model": omni_server.model, **extra_json}
    openai_client.send_audio_speech_http_request(
        {"json": body, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/audio/speech/batch (JSON)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_batch_empty_items(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_audio_speech_batch_http_request(
        {"json": {"items": []}, "timeout": 120, "err_code": (400, 422), "err_message": ("items", "least 1 item")}
    )


@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_batch_missing_items(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_audio_speech_batch_http_request(
        {
            "json": {"model": omni_server.model},
            "timeout": 120,
            "err_code": (400, 422),
            "err_message": ("items", "field required", "missing"),
        }
    )


# Align field-level checks with ``POST /v1/audio/speech`` where ``BatchSpeechRequest`` /
# ``SpeechBatchItem`` expose the same knobs. Not on batch schema (single-speech only):
# ``stream``, ``stream_format``, ``seed``, ``speaker_embedding`` — see dedicated speech tests.
@pytest.mark.parametrize(
    "loc, overrides, err_message",
    [
        pytest.param("item", {"input": ""}, ("input", "empty"), id="item_input_empty", marks=_SKIP_ISSUE_3649),
        pytest.param(
            "item",
            {"input": "   "},
            ("input", "empty"),
            id="item_input_whitespace_only",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param("batch", {"voice": ""}, "Invalid voice", id="batch_voice_empty", marks=_SKIP_ISSUE_3649),
        pytest.param("item", {"voice": ""}, "Invalid voice", id="item_voice_empty", marks=_SKIP_ISSUE_3649),
        pytest.param(
            "batch",
            {
                "voice": "nonexistent_voice_xyz",
                "task_type": "CustomVoice",
                "ref_audio": None,
                "ref_text": None,
            },
            "Invalid voice",
            id="batch_voice_unknown_preset",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {
                _SPEECH_BATCH_NO_CLONE_FIELDS: True,
                "voice": "nonexistent_voice_xyz",
                "task_type": "CustomVoice",
            },
            "Invalid voice",
            id="item_voice_unknown_preset",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"instructions": 123},
            ("instructions", "string_type", "valid string"),
            id="batch_instructions_wrong_type",
        ),
        pytest.param(
            "item",
            {"instructions": 123},
            ("instructions", "string_type", "valid string"),
            id="item_instructions_wrong_type",
        ),
        pytest.param(
            "batch",
            {"instructions": "x" * (_SPEECH_API_MAX_INSTRUCTIONS_CHARS + 1)},
            ("instructions", "too long"),
            id="batch_instructions_exceed_api_limit",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {"instructions": "x" * (_SPEECH_API_MAX_INSTRUCTIONS_CHARS + 1)},
            ("instructions", "too long"),
            id="item_instructions_exceed_api_limit",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"response_format": "mpeg"},
            ("response_format", "literal_error", "wav"),
            id="batch_response_format_invalid",
        ),
        pytest.param(
            "item",
            {"response_format": "mpeg"},
            ("response_format", "literal_error", "wav"),
            id="item_response_format_invalid",
        ),
        pytest.param("batch", {"speed": 0.1}, ("speed", "greater_than_equal", "0.25"), id="batch_speed_too_low"),
        pytest.param("item", {"speed": 10.0}, ("speed", "less_than_equal", "4"), id="item_speed_too_high"),
        pytest.param(
            "batch",
            {"task_type": "InvalidEnum"},
            ("task_type", "literal_error", "CustomVoice"),
            id="batch_task_type_invalid",
        ),
        pytest.param(
            "item",
            {"task_type": "InvalidEnum"},
            ("task_type", "literal_error", "CustomVoice"),
            id="item_task_type_invalid",
        ),
        pytest.param(
            "batch",
            {"language": ""},
            ("language", "invalid language", "chinese"),
            id="batch_language_empty",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {"language": ""},
            ("language", "invalid language", "chinese"),
            id="item_language_empty",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"ref_audio": "ftp://example.com/a.wav"},
            ("ref_audio", "URL"),
            id="batch_ref_audio_bad_scheme",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {"ref_audio": "ftp://example.com/a.wav"},
            ("ref_audio", "URL"),
            id="item_ref_audio_bad_scheme",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"ref_audio": "not_a_valid_uri"},
            ("ref_audio", "url"),
            id="batch_ref_audio_invalid_uri",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {"ref_audio": "not_a_valid_uri"},
            ("ref_audio", "url"),
            id="item_ref_audio_invalid_uri",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"ref_text": ["x"]},
            ("ref_text", "string_type", "valid string"),
            id="batch_ref_text_wrong_type",
        ),
        pytest.param(
            "item",
            {"ref_text": ["x"]},
            ("ref_text", "string_type", "valid string"),
            id="item_ref_text_wrong_type",
        ),
        pytest.param(
            "batch",
            {"x_vector_only_mode": "wrong_type"},
            ("x_vector_only_mode", "bool_parsing", "validation error"),
            id="batch_x_vector_only_mode_wrong_type",
        ),
        pytest.param(
            "item",
            {"x_vector_only_mode": "wrong_type"},
            ("x_vector_only_mode", "bool_parsing", "validation error"),
            id="item_x_vector_only_mode_wrong_type",
        ),
        pytest.param(
            "batch",
            {"max_new_tokens": 0},
            ("max_new_tokens", "least 1"),
            id="batch_max_new_tokens_below_min",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {"max_new_tokens": 0},
            ("max_new_tokens", "least 1"),
            id="item_max_new_tokens_below_min",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"max_new_tokens": _SPEECH_API_MAX_NEW_TOKENS + 1},
            ("max_new_tokens", "exceed 4096"),
            id="batch_max_new_tokens_above_max",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {"max_new_tokens": _SPEECH_API_MAX_NEW_TOKENS + 1},
            ("max_new_tokens", "exceed 4096"),
            id="item_max_new_tokens_above_max",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "batch",
            {"initial_codec_chunk_frames": -1},
            ("initial_codec_chunk_frames", "greater_than_equal", "0"),
            id="batch_initial_codec_chunk_frames_negative",
        ),
        pytest.param(
            "item",
            {"initial_codec_chunk_frames": -1},
            ("initial_codec_chunk_frames", "greater_than_equal", "0"),
            id="item_initial_codec_chunk_frames_negative",
        ),
        # Base clone: whitespace-only ref_text (same serving check as single speech).
        pytest.param(
            "batch",
            {
                "task_type": "Base",
                "ref_audio": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
                "ref_text": "   ",
            },
            ("ref_text", "base task", "non-empty"),
            id="batch_ref_text_whitespace_only_base_clone",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "item",
            {
                "task_type": "Base",
                "ref_audio": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
                "ref_text": "   ",
            },
            ("ref_text", "base task", "non-empty"),
            id="item_ref_text_whitespace_only_base_clone",
            marks=_SKIP_ISSUE_3649,
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_batch_invalid_field_values(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    loc: str,
    overrides: dict[str, object],
    err_message: str | tuple[str, ...],
) -> None:
    ov = dict(overrides)
    strip_clone = bool(ov.pop(_SPEECH_BATCH_NO_CLONE_FIELDS, False))
    body = {
        "model": omni_server.model,
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "language": "English",
        "response_format": "wav",
        "items": [{"input": "Hello."}],
    }
    if strip_clone:
        body.pop("ref_audio", None)
        body.pop("ref_text", None)
    _apply_batch_overrides(body, loc=loc, overrides=ov)
    openai_client.send_audio_speech_batch_http_request(
        {"json": body, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket /v1/audio/speech/stream
# ─────────────────────────────────────────────────────────────────────────────

# ``test_speech_stream_invalid_requests``: build ``session.config`` JSON with ``omni_server.model``.
_SPEECH_STREAM_WS_SESSION_STREAM_AUDIO_WAV_FRAMES = object()


@pytest.mark.parametrize(
    "send_frames_spec, err_message",
    [
        pytest.param("{not-json", "Invalid JSON", id="invalid_json"),
        pytest.param(
            json.dumps({"type": "input.text", "text": "hello"}),
            "Expected session.config",
            id="wrong_first_message_type",
        ),
        pytest.param(
            _SPEECH_STREAM_WS_SESSION_STREAM_AUDIO_WAV_FRAMES,
            ("invalid session config", "response_format", "pcm"),
            id="session_config_stream_audio_requires_pcm",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_speech_stream_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    send_frames_spec: Any,
    err_message: str | tuple[str, ...],
) -> None:
    """Bad first WebSocket text frames: non-JSON, wrong ``type``, or invalid ``session.config``."""
    if send_frames_spec is _SPEECH_STREAM_WS_SESSION_STREAM_AUDIO_WAV_FRAMES:
        send_frames = json.dumps(
            {
                "type": "session.config",
                "model": omni_server.model,
                "response_format": "wav",
                "stream_audio": True,
            }
        )
    else:
        send_frames = send_frames_spec
    assert isinstance(send_frames, str)
    openai_client.send_audio_speech_stream_ws_request(
        {
            "send_frames": send_frames,
            "timeout": 120,
            "ws_max_size": None,
            "ws_json_type": "error",
            "err_message": err_message,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/audio/voices (multipart + embedding-only) and DELETE /v1/audio/voices/{name}
# ─────────────────────────────────────────────────────────────────────────────

# Templates for ``test_voices_create_invalid_requests``: resolved in ``_finalize_voices_form_data``.
_VF_LONG_CONSENT = object()
_VF_LONG_REF_TEXT = object()
_VF_LONG_SPEAKER_DESC = object()
_VF_EMBEDDING_NAN_JSON = object()


def _finalize_voices_form_data(template: dict[str, Any], uuid_hex: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in template.items():
        if val is _VF_LONG_CONSENT:
            out[key] = "x" * (_VOICE_UPLOAD_MAX_CONSENT_LEN + 1)
        elif val is _VF_LONG_REF_TEXT:
            out[key] = "y" * (_VOICE_UPLOAD_MAX_REF_TEXT_CHARS + 1)
        elif val is _VF_LONG_SPEAKER_DESC:
            out[key] = "z" * (_VOICE_UPLOAD_MAX_SPEAKER_DESCRIPTION_CHARS + 1)
        elif val is _VF_EMBEDDING_NAN_JSON:
            out[key] = json.dumps([0.1] * 1023 + [float("nan")])
        elif isinstance(val, str):
            out[key] = val.replace("{uuid}", uuid_hex)
        else:
            out[key] = val
    return out


def _voices_upload_multipart_files(kind: str | None) -> dict[str, Any] | None:
    """``None`` => no multipart file (embedding-only or neither)."""
    if kind is None:
        return None
    if kind == "wav_ok":
        return {"audio_sample": ("clip.wav", _pcm_wav_mono_bytes(), "audio/wav")}
    if kind == "wav_short":
        return {"audio_sample": ("clip.wav", _pcm_wav_mono_bytes(duration_s=0.3), "audio/wav")}
    if kind == "wav_bad_body":
        return {"audio_sample": ("clip.wav", b"not valid wav content", "audio/wav")}
    if kind == "wav_pdf_type":
        return {"audio_sample": ("clip.wav", _pcm_wav_mono_bytes(), "application/pdf")}
    raise AssertionError(f"unknown multipart kind {kind!r}")


@pytest.mark.parametrize(
    "multipart_kind, form_template, err_message",
    [
        pytest.param(
            None,
            {"consent": "c1", "name": "v1"},
            ("audio_sample", "speaker_embedding", "provided"),
            id="neither_audio_nor_embedding",
        ),
        pytest.param(
            "wav_ok",
            {"name": "v_miss_consent_{uuid}"},
            ("missing", "consent", "field required"),
            id="missing_consent",
        ),
        pytest.param(
            "wav_ok",
            {"consent": "consent_ok"},
            ("missing", "name", "field required"),
            id="missing_name",
        ),
        pytest.param(
            "wav_ok",
            {"consent": "", "name": "v_empty_consent_{uuid}"},
            ("missing", "consent", "field required"),
            id="empty_consent",
        ),
        pytest.param(
            "wav_ok",
            {"consent": "   ", "name": "v_ws_consent_{uuid}"},
            "consent",
            id="whitespace_consent",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "wav_ok",
            {"consent": "bad/consent", "name": "v_bad_cons_{uuid}"},
            "consent",
            id="consent_path_sep",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "wav_ok",
            {"consent": _VF_LONG_CONSENT, "name": "v_long_cons_{uuid}"},
            ("consent", "too long", "Failed to save"),
            id="consent_too_long",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "wav_ok",
            {"consent": "consent_ok", "name": ""},
            ("missing", "name", "field required"),
            id="empty_name",
        ),
        pytest.param(
            "wav_ok",
            {"consent": "consent_ok", "name": "evil/name"},
            ("voice name", "invalid voice name"),
            id="invalid_name_path_sep",
        ),
        pytest.param(
            "wav_ok",
            {"consent": "consent_ok", "name": "v_long_ref_{uuid}", "ref_text": _VF_LONG_REF_TEXT},
            "ref_text",
            id="ref_text_too_long",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "wav_ok",
            {
                "consent": "consent_ok",
                "name": "v_long_desc_{uuid}",
                "speaker_description": _VF_LONG_SPEAKER_DESC,
            },
            "speaker_description",
            id="speaker_description_too_long",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "wav_pdf_type",
            {"consent": "consent_ok", "name": "v_bad_mime_{uuid}"},
            ("mime", "unsupported", "audio/mp4"),
            id="audio_unsupported_mime",
        ),
        pytest.param(
            "wav_short",
            {"consent": "consent_ok", "name": "v_short_{uuid}"},
            ("too short", "reference audio"),
            id="audio_too_short",
        ),
        pytest.param(
            "wav_bad_body",
            {"consent": "consent_ok", "name": "v_bad_wav_{uuid}"},
            ("decode", "audio", "Format not recognised"),
            id="audio_decode_error",
        ),
        pytest.param(
            None,
            {
                "speaker_embedding": "not valid json [[[",
                "consent": "consent_emb",
                "name": "v_bad_json_{uuid}",
            },
            ("speaker_embedding", "valid JSON"),
            id="speaker_embedding_invalid_json",
        ),
        pytest.param(
            None,
            {
                "speaker_embedding": json.dumps([]),
                "consent": "consent_emb",
                "name": "v_empty_emb_{uuid}",
            },
            ("speaker_embedding", "non-empty"),
            id="speaker_embedding_empty",
        ),
        pytest.param(
            None,
            {
                "speaker_embedding": _VF_EMBEDDING_NAN_JSON,
                "consent": "consent_emb",
                "name": "v_nan_emb_{uuid}",
            },
            ("speaker_embedding", "finite"),
            id="speaker_embedding_nan",
        ),
        pytest.param(
            None,
            {
                "speaker_embedding": json.dumps([0.01] * 512),
                "consent": "consent_emb",
                "name": "v_wrong_dim_{uuid}",
            },
            ("expected", "dimensions", "1024"),
            id="speaker_embedding_wrong_dim",
        ),
        pytest.param(
            "wav_ok",
            {
                "consent": "consent_emb",
                "name": "v_both_{uuid}",
                "speaker_embedding": json.dumps([0.01] * 1024),
            },
            ("mutually exclusive", "audio_sample", "speaker_embedding"),
            id="audio_and_embedding_mutually_exclusive",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_voices_create_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    multipart_kind: str | None,
    form_template: dict[str, Any],
    err_message: str | tuple[str, ...],
) -> None:
    """Invalid ``POST /v1/audio/voices`` bodies (multipart or embedding-only)."""
    uid = uuid.uuid4().hex[:8]
    cfg: dict[str, Any] = {
        "data": _finalize_voices_form_data(form_template, uid),
        "timeout": 120,
        "err_code": 400,
        "err_message": err_message,
    }
    files = _voices_upload_multipart_files(multipart_kind)
    if files is not None:
        cfg["files"] = files
    openai_client.send_audio_voices_create_http_request(cfg)


@pytest.mark.parametrize(
    "voice_name, err_code, err_message",
    [
        pytest.param(
            "missing-voice-404",
            404,
            "not found",
            id="not_found",
        ),
        pytest.param(
            "   ",
            404,
            "not found",
            id="whitespace_only",
        ),
        pytest.param(
            "evil/name",
            404,
            "not found",
            id="path_separator",
        ),
        pytest.param(
            ".",
            405,
            ("Method Not Allowed", "detail"),
            id="dot_single",
        ),
        pytest.param(
            "..",
            404,
            "not found",
            id="dot_dot",
        ),
        pytest.param(
            "bad\x00voice",
            404,
            "not found",
            id="nul_byte",
        ),
        pytest.param(
            "",
            405,
            "Method Not Allowed",
            id="empty_name_path",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_TTS_SPEECH, indirect=True)
def test_voices_delete_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    voice_name: str,
    err_code: int | tuple[int, ...],
    err_message: str | tuple[str, ...],
) -> None:
    """DELETE ``/v1/audio/voices/{name}``: missing voice (404), odd segments (often 404), ``.``paths (405)."""
    openai_client.send_audio_voices_delete_http_request(
        {
            "name": voice_name,
            "timeout": 120,
            "err_code": err_code,
            "err_message": err_message,
        }
    )
