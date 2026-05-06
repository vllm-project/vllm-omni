# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for AudioX text-to-audio diffusion.

Mirrors `tests/e2e/online_serving/test_sd3_expansion.py` for image diffusion: spin up
`vllm-omni` with `--model-class-name AudioXPipeline` and validate via the standard
`send_diffusion_request` helper (now audio-aware).
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

# Tiny / random checkpoint usable in CI; override to a real bundle locally.
AUDIOX_TEST_MODEL = os.environ.get("AUDIOX_TEST_MODEL", "zhangj1an/audiox_random")
T2A_PROMPT = "A quiet living room with soft fabric rustle and gentle cat breathing."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})


def _audiox_server_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--model-class-name", "AudioXPipeline"],
            ),
            id="t2a",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _audiox_server_cases(AUDIOX_TEST_MODEL), indirect=True)
def test_audiox_t2a_online(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """AudioX text-to-audio: chat completion returns a non-empty WAV in `message.audio.data`."""
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(content_text=T2A_PROMPT),
        "extra_body": {
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
            "seed": 42,
            "audiox_task": "t2a",
            "seconds_start": 0.0,
            "seconds_total": 2.0,
        },
    }
    openai_client.send_diffusion_request(request_config)
