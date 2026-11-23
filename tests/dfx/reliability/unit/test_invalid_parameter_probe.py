# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for invalid-parameter liveness probes."""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest

from tests.dfx.reliability.invalid_param_test import conftest as invalid_conftest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00")
    return output.getvalue()


def test_minimax_music3_liveness_probe_uses_music_request_shape() -> None:
    requests: list[dict] = []

    class _Client:
        def send_audio_speech_http_request(self, request):
            requests.append(request)
            return [SimpleNamespace(status_code=200, error_message="", content=_wav_bytes())]

    module = SimpleNamespace(
        _MINIMAX_MUSIC3_LYRICS="[Verse]\nTest lyrics",
        _MINIMAX_MUSIC3_CAPTION="A short test song.",
    )

    invalid_conftest._assert_valid_inference(
        _Client(),
        SimpleNamespace(model="MiniMaxAI/MiniMax-Music3"),
        module,
    )

    assert requests == [
        {
            "json": {
                "model": "MiniMaxAI/MiniMax-Music3",
                "input": module._MINIMAX_MUSIC3_LYRICS,
                "response_format": "wav",
                "instructions": module._MINIMAX_MUSIC3_CAPTION,
                "seed": 1,
                "max_new_tokens": 25,
            },
            "timeout": 180,
        }
    ]
