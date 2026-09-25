# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LTX-2.5 text-to-audio smoke through the registered OpenAI endpoint."""

from __future__ import annotations

import io
import os
import wave

import numpy as np
import pytest
import requests

from tests.helpers import skip_if_gated_repo_inaccessible
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DEFAULT_MODEL = "Lightricks/LTX-2.5-Diffusers"
DEFAULT_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
MODEL = os.environ.get("VLLM_TEST_LTX25_MODEL", DEFAULT_MODEL)
MODEL_REVISION = os.environ.get(
    "VLLM_TEST_LTX25_MODEL_REVISION",
    DEFAULT_REVISION if MODEL == DEFAULT_MODEL else "",
)
SAMPLE_RATE = 48_000
EXPECTED_SAMPLE_COUNT = 96_480

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def require_ltx25_model_access() -> None:
    if not os.path.isdir(MODEL):
        skip_if_gated_repo_inaccessible(MODEL, revision=MODEL_REVISION or None, filename="model_index.json")


def _server() -> OmniServerParams:
    return OmniServerParams(
        model=MODEL,
        server_args=[
            *(["--revision", MODEL_REVISION] if MODEL_REVISION else []),
            "--model-class-name",
            "LTX2TextToAudioPipeline",
            "--enforce-eager",
            "--diffusion-attention-backend",
            "CUDNN_ATTN",
        ],
    )


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_server(), id="ltx25_t2a")], indirect=True)
def test_ltx25_text_to_audio_online(omni_server, openai_client) -> None:
    """The registered T2A pipeline returns a non-empty stereo 48 kHz WAV."""
    response = requests.post(
        f"{openai_client.base_url.rstrip('/')}/v1/audio/generate",
        json={
            "model": omni_server.model,
            "input": "A close-up recording of a concert grand piano playing a gentle melody.",
            "audio_length": 2.0,
            "num_inference_steps": 30,
            "guidance_scale": 7.0,
            "seed": 42,
            "response_format": "wav",
        },
        timeout=900,
    )
    assert response.status_code == 200, response.text[:500]
    assert response.content[:4] == b"RIFF"

    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() > 0
        assert wav.getnframes() == EXPECTED_SAMPLE_COUNT
        frames = wav.readframes(wav.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16)
    assert samples.size > 0
    assert np.isfinite(samples).all()
    assert np.any(samples != 0), "generated audio is empty"
