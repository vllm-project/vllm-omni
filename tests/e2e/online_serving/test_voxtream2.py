# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online test for Voxtream2 via /v1/audio/speech."""

import os
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import httpx
import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

MODEL = "herimor/voxtream2"
REPO_ROOT = Path(__file__).parents[3]
VOXTREAM_ROOT = Path(os.environ.get("VOXTREAM2_ROOT", REPO_ROOT / "voxtream")).resolve()
REF_AUDIO = VOXTREAM_ROOT / "assets" / "audio" / "english_female.wav"
STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxtream2_1stage.yaml"
MIN_AUDIO_BYTES = 5000

TEST_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=str(STAGE_CONFIG),
        server_args=["--trust-remote-code", "--enforce-eager", "--disable-log-stats"],
        env_dict={"VOXTREAM2_ROOT": str(VOXTREAM_ROOT)},
        stage_init_timeout=600,
    )
]


def _verify_wav_audio(content: bytes) -> bool:
    return len(content) >= 44 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"


def _make_speech_request(host: str, port: int, text: str, ref_audio: Path) -> httpx.Response:
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "input": text,
        "ref_audio": ref_audio.as_uri(),
        "response_format": "wav",
    }
    with httpx.Client(timeout=300.0) as client:
        return client.post(url, json=payload)


@pytest.mark.skipif(not REF_AUDIO.is_file(), reason="Voxtream2 reference audio is not available")
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_voxtream2_speech_file_ref_audio(omni_server) -> None:
    response = _make_speech_request(
        host=omni_server.host,
        port=omni_server.port,
        text="This is a Voxtream2 online serving example.",
        ref_audio=REF_AUDIO,
    )

    assert response.status_code == 200, f"Request failed: {response.text}"
    assert response.headers.get("content-type") == "audio/wav"
    assert _verify_wav_audio(response.content), "Response is not valid WAV audio"
    assert len(response.content) > MIN_AUDIO_BYTES, (
        f"Audio too small ({len(response.content)} bytes), expected > {MIN_AUDIO_BYTES}"
    )
