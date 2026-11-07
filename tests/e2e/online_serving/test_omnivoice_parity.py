# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-mode full-model parity tests for OmniVoice."""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
import requests

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "k2-fsa/OmniVoice"
STAGE_CONFIG = get_deploy_config_path("omnivoice.yaml")
PROMPT = "The weather is nice today, perfect for a walk in the park."

payload = {
    "model": MODEL,
    "input": PROMPT,
    "language": "English",
    "seed": 42,
    "response_format": "wav",
    "extra_params": {"num_inference_steps": 32},
}


def _generate_without_graph(server_args: list[str]) -> bytes:
    with OmniServer(
        MODEL,
        server_args,
        use_omni=True,
        env_dict={"OMNIVOICE_CUDA_GRAPH": "0"},
    ) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/audio/speech",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        assert response.content.startswith(b"RIFF")
        return response.content


def _generate_with_graph(server_args: list[str]) -> bytes:
    with OmniServer(
        MODEL,
        server_args,
        use_omni=True,
    ) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/audio/speech",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        assert response.content.startswith(b"RIFF")
        return response.content


def _common_args() -> list[str]:
    """Return the shared B=1 server arguments for parity tests."""
    return [
        "--trust-remote-code",
        "--disable-log-stats",
        "--deploy-config",
        STAGE_CONFIG,
        "--max-num-seqs",
        "1",
    ]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_request_mode_and_step_execution_b1_parity_without_graph() -> None:
    """B=1 eager request and step modes must produce identical seeded WAV bytes."""
    common_args = _common_args()

    request_audio = _generate_without_graph(common_args)
    step_audio = _generate_without_graph([*common_args, "--step-execution", "--enforce-eager"])

    assert request_audio == step_audio


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_request_mode_and_step_execution_b1_parity_with_graph() -> None:
    """B=1 Graph request and step modes must produce identical seeded WAV bytes."""
    common_args = _common_args()

    request_audio = _generate_with_graph(common_args)
    step_audio = _generate_with_graph([*common_args, "--step-execution", "--enforce-eager"])

    assert request_audio == step_audio
