# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E online serving test for Stable Audio Open text-to-audio diffusion.

Stable Audio Open is served through the OpenAI-compatible
`POST /v1/audio/generate` endpoint (JSON in, binary WAV out).
"""

import os
from io import BytesIO

import pytest
import requests
import soundfile
import torch

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams
from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor

# Stable Audio Open is gated; override to a local bundle locally if needed.
STABLE_AUDIO_TEST_MODEL = os.environ.get("STABLE_AUDIO_TEST_MODEL", "stabilityai/stable-audio-open-1.0")
T2A_PROMPT = "A piano playing a gentle melody with soft room ambience."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})


def _stable_audio_server_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--watermark-config", '{"modalities":{"audio":{"algorithm":"audioseal"}}}'],
            ),
            id="t2a",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _stable_audio_server_cases(STABLE_AUDIO_TEST_MODEL), indirect=True)
def test_stable_audio_t2a_online_with_watermarking(
    omni_server: OmniServer,
    online_client,
) -> None:
    """Stable Audio Open text-to-audio: `/v1/audio/generate` returns a non-empty
    WAV.

    Uses tiny steps / short duration to keep CI light, matching the offline smoke
    test in `tests/e2e/offline_inference/test_stable_audio_expansion.py`.

    We also check to ensure that we can watermark through the server's diffusion path
    using this model.
    """
    with requests.post(
        f"{online_client.base_url}/v1/audio/generate",
        json={
            "model": omni_server.model,
            "input": T2A_PROMPT,
            "audio_length": 2.0,
            "num_inference_steps": 4,
            "guidance_scale": 7.0,
            "negative_prompt": "Low quality.",
            "seed": 42,
            "response_format": "wav",
        },
        timeout=300,
    ) as response:
        assert response.status_code == 200, response.text
        content = response.content

    audio, sample_rate = soundfile.read(BytesIO(content), dtype="float32", always_2d=True)
    samples = torch.from_numpy(audio.T.copy()).unsqueeze(0)

    watermarker = AudioSealWatermarker()
    try:
        assert watermarker.is_watermarked(AudioTensor(samples, sample_rate))
    finally:
        watermarker.close()
