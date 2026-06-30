# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for Stable Audio Open text-to-audio diffusion.

Counterpart to `tests/e2e/online_serving/test_audiox_expansion.py`: AudioX is
served through chat-completions, while Stable Audio Open is served through the
OpenAI-compatible `POST /v1/audio/generate` endpoint (JSON in, binary WAV out).
This exercises the standard online path used by
`examples/online_serving/text_to_audio/run_curl_stable_audio.sh`.
"""

import os

import pytest

from tests.helpers import skip_if_gated_repo_inaccessible
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

# Stable Audio Open is gated; override to a local bundle locally if needed.
STABLE_AUDIO_TEST_MODEL = os.environ.get("STABLE_AUDIO_TEST_MODEL", "stabilityai/stable-audio-open-1.0")
T2A_PROMPT = "A piano playing a gentle melody with soft room ambience."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})


def _stable_audio_server_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="t2a",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.fixture
def _require_stable_audio_access() -> None:
    """Skip cleanly (before the server boots) if the gated checkpoint is inaccessible."""
    skip_if_gated_repo_inaccessible(STABLE_AUDIO_TEST_MODEL)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _stable_audio_server_cases(STABLE_AUDIO_TEST_MODEL), indirect=True)
def test_stable_audio_t2a_online(
    _require_stable_audio_access: None,
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Stable Audio Open text-to-audio: `/v1/audio/generate` returns a non-empty WAV.

    Uses tiny steps / short duration to keep CI light, matching the offline smoke
    test in `tests/e2e/offline_inference/test_stable_audio_expansion.py`.
    """
    request_config = {
        "json": {
            "model": omni_server.model,
            "input": T2A_PROMPT,
            "audio_length": 2.0,
            "num_inference_steps": 4,
            "guidance_scale": 7.0,
            "negative_prompt": "Low quality.",
            "seed": 42,
            "response_format": "wav",
        },
        "timeout": 300,
    }
    responses = openai_client.send_audio_generate_http_request(request_config)
    assert responses, "no response from /v1/audio/generate"
    resp = responses[0]
    assert resp.success, f"audio generate failed: {resp.status_code} {resp.error_message}"
    assert resp.status_code == 200
