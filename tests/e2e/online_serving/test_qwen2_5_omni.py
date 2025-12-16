# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen2_5-Omni model with video input and audio output.
"""

import os

from pathlib import Path
import pytest
import subprocess
from tests.conftest import OmniServer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen2.5-Omni-7B"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")]
# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses module scope so the server starts only once for all tests.
    Multi-stage initialization can take 10-20+ minutes.
    """
    model, stage_config_path = request.param
    #with OmniServer(model, ["--stage-configs-path", stage_config_path]) as server:
    with OmniServer(model, []) as server:
        yield server

@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_audio(
    omni_server,
) -> None:
    """Test processing video+audio+image, generating audio output via OpenAI API."""
    # Create data URL for the base64 encoded video
    command = [
        "vllm-omni",
        "bench",
        "serve",
        "--model",
        omni_server.model,
        "--host",
        omni_server.host,
        "--port",
        str(omni_server.port),
        "--dataset-name",
        "random-mm",
        "--request_rate",
        "1",
        "--random-input-len",
        "32",
        "--random-range-ratio",
        "0.0",
        "--random-mm-base-items-per-request",
        "2",
        "--random-mm-num-mm-items-range-ratio",
        "0",
        "--random-mm-limit-mm-per-prompt",
        '{"image":10, "video": 1, "audio": 1}',
        "--random-mm-bucket-config",
        '{"(640,640,1)":0.5, "(0,1,1)": 0.1, "(256, 256, 2)": 0.4}',
        "--ignore-eos",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
