# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for MOSS-SoundEffect-v2.0 text-to-audio diffusion.
"""

from io import BytesIO

import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams


def _moss_soundeffect_server_cases():
    return [
        pytest.param(
            OmniServerParams(
                model="OpenMOSS-Team/MOSS-SoundEffect-v2.0",
            ),
            marks=hardware_marks(res={"cuda": "L4"}),
        ),
    ]


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _moss_soundeffect_server_cases(), indirect=True)
def test_moss_sound_effect_v2_online(omni_server: OmniServer) -> None:
    response = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/audio/generate",
        json={
            "model": omni_server.model,
            "input": "The sound of a dog barking",
            "audio_length": 2.0,
            "num_inference_steps": 10,
            "guidance_scale": 6.0,
            "seed": 42,
            "extra_params": {"sigma_shift": 7.0},
        },
        timeout=300,
    )
    response.raise_for_status()

    audio, sample_rate = sf.read(BytesIO(response.content))
    assert sample_rate == 48000
    assert audio.ndim == 1
    assert audio.size > 0
    assert abs(len(audio) / sample_rate - 2.0) < 0.1
