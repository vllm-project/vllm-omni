# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``POST /v1/audio/generate`` validation (live Stable Audio Open server)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

_PARAMS = [
    pytest.param(
        OmniServerParams(
            model="stabilityai/stable-audio-open-1.0",
            server_args=[
                "--trust-remote-code",
                "--enforce-eager",
                "--model-class-name",
                "StableAudioPipeline",
            ],
        ),
        id="stable_audio_open",
        marks=hardware_marks(res={"cuda": "L4"}),
    ),
]


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param({"input": ""}, ("cannot be empty", "input"), id="input_empty"),
        pytest.param({"input": "   "}, ("cannot be empty", "input"), id="input_whitespace_only"),
        pytest.param({"input": 123}, ("input", "a valid string"), id="input_wrong_type"),
        pytest.param(
            {"response_format": "mpeg"},
            ("response_format", "literal_error", "Input should be"),
            id="response_format_invalid",
        ),
        pytest.param({"speed": 0.1}, ("speed", "greater_than_equal", "0.25"), id="speed_too_low"),
        pytest.param({"speed": 99.0}, ("speed", "less_than_equal", "4"), id="speed_too_high"),
        pytest.param({"stream_format": 1}, ("stream_format", "literal_error", "audio"), id="stream_format_wrong_type"),
        pytest.param(
            {"stream_format": "sse"},
            ("sse", "stream_format", "value_error", "not a supported stream_format"),
            id="stream_format_sse_blocked",
        ),
        pytest.param({"audio_length": -1.0}, ("audio_length", "greater_than", "0"), id="audio_length_negative"),
        pytest.param({"audio_length": 0}, ("audio_length", "greater_than", "0"), id="audio_length_zero"),
        # ``OmniOpenAIServingAudioGenerate`` only forwards ``audio_start`` when ``audio_length`` is set.
        pytest.param(
            {"audio_start": -0.5, "audio_length": 5.0},
            ("audio_start_in_s", "between", "0", "512"),
            id="audio_start_negative",
        ),
        pytest.param(
            {"negative_prompt": ["noise"]}, ("negative_prompt", "a valid string"), id="negative_prompt_wrong_type"
        ),
        pytest.param({"guidance_scale": -1.0}, ("guidance_scale", "greater_than", "0"), id="guidance_scale_negative"),
        pytest.param({"guidance_scale": 0}, ("guidance_scale", "greater_than", "0"), id="guidance_scale_zero"),
        pytest.param(
            {"num_inference_steps": 0},
            ("num_inference_steps", "greater_than_equal", "1"),
            id="num_inference_steps_zero",
        ),
        pytest.param(
            {"num_inference_steps": -1},
            ("num_inference_steps", "greater_than_equal", "1"),
            id="num_inference_steps_negative",
        ),
        pytest.param(
            {"num_inference_steps": 6000},
            ("num_inference_steps", "less_than_equal", "1000"),
            id="num_inference_steps_above_max",
        ),
        pytest.param(
            {"guidance_scale": 1001.0},
            ("guidance_scale", "less_than_equal", "1000"),
            id="guidance_scale_above_max",
        ),
        pytest.param(
            {"audio_length": 86401.0}, ("Requested audio length", "exceeds maximum"), id="audio_length_above_max"
        ),
        pytest.param(
            {"audio_start": 86401.0, "audio_length": 1.0},
            ("audio_start_in_s", "between", "0", "512"),
            id="audio_start_above_max",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_audio_generate_invalid_field_values(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    overrides: dict[str, object],
    err_message: str | tuple[str, ...],
) -> None:
    body: dict[str, Any] = {"model": omni_server.model, "input": "ambient electronic pad"}
    body.update(overrides)
    online_client.send_audio_generate_http_request(
        {"json": body, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_audio_generate_missing_input(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    online_client.send_audio_generate_http_request(
        {
            "json": {"model": omni_server.model},
            "timeout": 120,
            "err_code": 400,
            "err_message": ("missing", "input", "field required"),
        }
    )
