# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end audio watermarking test."""

from collections.abc import Generator
from io import BytesIO

import pytest
import requests
import soundfile
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=[
                "--trust-remote-code",
                "--watermark-config",
                '{"strict":true,"modalities":{"audio":{"algorithm":"audioseal"}}}',
            ],
        ),
        id="qwen3-tts-0.6b",
    )
]


@pytest.fixture(scope="module")
def audio_watermarker() -> Generator[AudioSealWatermarker, None, None]:
    """Reuse the AudioSeal models across streaming modes."""
    watermarker = AudioSealWatermarker()
    yield watermarker
    watermarker.close()


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("stream", [False, True], ids=["without stream", "with stream"])
@pytest.mark.parametrize("watermarking", [True, False], ids=["watermarked", "opted out"])
def test_tts_audio_watermarking(
    omni_server,
    online_client,
    audio_watermarker: AudioSealWatermarker,
    stream: bool,
    watermarking: bool,
) -> None:
    """Ensure TTS audio is watermarked unless the request opts out, with and without streaming."""
    payload = {
        "model": omni_server.model,
        "input": "Hello, this is a short watermark test.",
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "vivian",
        "max_new_tokens": 64,
        "stream": stream,
        "watermarking": watermarking,
    }
    if stream:
        payload["stream_format"] = "audio"

    with requests.post(
        f"{online_client.base_url}/v1/audio/speech",
        json=payload,
        stream=stream,
        timeout=300,
    ) as response:
        assert response.status_code == 200, response.text
        content = b"".join(response.iter_content(chunk_size=None)) if stream else response.content

    audio, sample_rate = soundfile.read(BytesIO(content), dtype="float32", always_2d=True)
    samples = torch.from_numpy(audio.T.copy()).unsqueeze(0)

    assert audio_watermarker.is_watermarked(AudioTensor(samples, sample_rate)) is watermarking
