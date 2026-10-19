# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK Speech API E2E coverage for the encoder -> diffusion deployment."""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VLLM_OMNI_AUK_MODEL_DIR")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skipif(not MODEL, reason="set VLLM_OMNI_AUK_MODEL_DIR to an assembled AuK checkpoint"),
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("reference", [False, True], ids=["instruct", "voice-clone"])
@pytest.mark.parametrize("stream", [False, True], ids=["wav", "streaming-pcm"])
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL or ".",
                stage_config_path=get_deploy_config_path("auk.yaml"),
                server_args=["--trust-remote-code", "--disable-log-stats"],
            ),
            id="auk",
        )
    ],
    indirect=True,
)
def test_auk_speech_api_request(omni_server, reference, stream) -> None:
    """The adapter accepts Speech API requests and returns bounded 24 kHz WAVs."""

    def generate(duration: float) -> int:
        extra = {}
        if reference:
            extra["ref_audio"] = get_asset_path("cosyvoice3/zero_shot_prompt.wav", as_data_url=True)
            # Match the shared seed-tts benchmark request envelope.
            extra["ref_text"] = "Reference transcript is not required by AuK."
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech",
            json={
                "model": omni_server.model,
                "input": "",
                "instructions": (
                    "Say the following with the same voice: 'Hello, this is an AuK speech API test.'"
                    if reference
                    else 'Generate speech based on the following description: "A clear, natural voice.". '
                    'The content to speak is: "Hello, this is an AuK speech API test.".'
                ),
                "voice": "default",
                "duration_seconds": duration,
                "seed": 7,
                "response_format": "pcm" if stream else "wav",
                "stream": stream,
                "stream_format": "audio" if stream else None,
                **extra,
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        if stream:
            waveform = np.frombuffer(response.content, dtype="<i2").astype(np.float32) / 32768
        else:
            waveform, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
            assert sample_rate == 24_000
        assert np.isfinite(waveform).all()
        assert np.max(np.abs(waveform)) > 0
        return int(np.asarray(waveform).reshape(-1).shape[0])

    assert generate(2.0) == 100 * 480
