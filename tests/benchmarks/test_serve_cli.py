import subprocess
from pathlib import Path

import pytest

from tests.conftest import OmniServer

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]
stage_configs = [str(Path(__file__).parent.parent / "e2e" / "stage_configs" / "qwen3_omni_ci.yaml")]

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
def test_bench_serve_chat(omni_server):
    command = [
        "vllm",
        "bench",
        "serve",
        "--omni",
        "--model",
        omni_server.model,
        "--port",
        str(omni_server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat-omni",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
