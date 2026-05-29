# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable Audio 3 online e2e: text-to-audio via the /v1/audio/generate endpoint.

Mirrors the audio-diffusion serving path used by stable-audio-open
(see examples/online_serving/stable_audio/). Point ``STABLE_AUDIO_3_TEST_MODEL``
at a directory prepared by
``examples/offline_inference/stable_audio_3/download_stable_audio_3.py`` (which
writes the ``model_index.json`` / ``transformer/config.json`` the engine needs).
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

_MODEL = os.environ.get("STABLE_AUDIO_3_TEST_MODEL", "stabilityai/stable-audio-3-medium")

_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            server_args=[
                "--trust-remote-code",
                "--enforce-eager",
                "--model-class-name",
                "StableAudio3Pipeline",
            ],
        ),
        id="stable_audio_3_medium",
        marks=hardware_marks(res={"cuda": "L4"}),
    ),
]


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_stable_audio_3_t2a_online(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """SA3 text-to-audio over HTTP: /v1/audio/generate returns a 200 WAV response."""
    body = {
        "model": omni_server.model,
        "input": "An ambient drone with shimmering overtones",
        "audio_length": 2.0,
        "guidance_scale": 7.0,
        "num_inference_steps": 4,
        "seed": 42,
    }
    responses = openai_client.send_audio_generate_http_request({"json": body, "timeout": 120})

    assert responses, "no response from /v1/audio/generate"
    resp = responses[0]
    assert resp.status_code == 200, f"expected HTTP 200, got {resp.status_code}: {resp.error_message}"
    assert resp.success, f"audio generation failed: {resp.error_message}"
