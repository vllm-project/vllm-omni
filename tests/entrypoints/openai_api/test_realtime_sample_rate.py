# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime session audio rates come from the negotiated session format."""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.entrypoints.openai.realtime.connection import SAMPLE_RATE_HZ, OpenAIFullDuplexConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_connection() -> OpenAIFullDuplexConnection:
    # Only the constructor's attribute needs are exercised here; the session
    # state (and its default negotiated formats) is what the rate helpers read.
    return OpenAIFullDuplexConnection(
        websocket=SimpleNamespace(),
        engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=100, multimodal_config=None)),
        model_name="test-model",
        chat_handler=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: object())),
    )


def _wav_rate(b64: str) -> int:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as wav:
        return wav.getframerate()


def test_default_session_negotiates_24khz_both_directions() -> None:
    connection = _make_connection()
    assert connection._input_audio_rate_hz() == SAMPLE_RATE_HZ == 24000
    assert connection._output_audio_rate_hz() == 24000


def test_rates_follow_the_session_format_not_the_module_constant() -> None:
    # The rate must be sourced from the negotiated session format: mutating
    # it (as a future non-24kHz capability would) changes what the helpers
    # report, without touching the module constant.
    connection = _make_connection()
    audio: Any = connection.session.config.audio
    audio.input.format.rate = 16000
    audio.output.format.rate = 48000
    assert connection._input_audio_rate_hz() == 16000
    assert connection._output_audio_rate_hz() == 48000


def test_bare_string_format_falls_back_to_capability_constant() -> None:
    # String formats ("audio/pcm") carry no rate; pcm16 support is 24kHz-only.
    connection = _make_connection()
    connection.session.config.audio.input.format = "audio/pcm"
    connection.session.config.audio.output.format = "pcm16"
    assert connection._input_audio_rate_hz() == 24000
    assert connection._output_audio_rate_hz() == 24000


def test_input_wav_labels_negotiated_input_rate() -> None:
    connection = _make_connection()
    connection.session.config.audio.input.format.rate = 16000
    assert _wav_rate(connection._pcm16_wav_b64(b"\x00\x01\x00\x02")) == 16000
