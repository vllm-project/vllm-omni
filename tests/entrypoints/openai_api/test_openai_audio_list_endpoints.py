# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GET ``/v1/audio/voices`` and ``/v1/models`` against a running omni server (TTS model).

Migrated from ``tests/e2e/online_serving/test_voxtral_tts.py`` to colocate with other OpenAPI entrypoint tests.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import httpx
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "mistralai/Voxtral-4B-TTS-2603"
STAGE_CONFIG = get_deploy_config_path("voxtral_tts.yaml")
EXTRA_ARGS = ["--trust-remote-code", "--enforce-eager", "--disable-log-stats"]
TEST_PARAMS = [OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, server_args=EXTRA_ARGS)]


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOpenAIAudioListEndpoints:
    """HTTP surface checks for OpenAI-compatible meta routes (voices, models)."""

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_list_voices_endpoint(self, omni_server) -> None:
        """``GET /v1/audio/voices`` returns a non-empty ``voices`` list."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/voices"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)

        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert isinstance(data["voices"], list)
        assert len(data["voices"]) > 0

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_models_endpoint(self, omni_server) -> None:
        """``GET /v1/models`` returns at least one model."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/models"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
