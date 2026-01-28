"""
Example Online tests for Qwen3-Omni model.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path

import pytest

from tests.conftest import (
    OmniServer,
)
from vllm_omni.utils import is_rocm

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2xH100-80G GPUs or AMD GPU MI325
if is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_configs = [str(Path(__file__).parent / "stage_configs" / "rocm" / "qwen3_omni_ci.yaml")]
else:
    stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    model, stage_config_path = request.param

    print(f"Starting OmniServer with model: {model}")
    print("This may take 10-20+ minutes for initialization...")

    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"]) as server:
        print("OmniServer started successfully")
        yield server
        print("OmniServer stopped")


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_multimodal_generation_001() -> None:
    pass
