import subprocess
from pathlib import Path

import pytest

from tests.utils import hardware_test

models = ["Qwen/Qwen2.5-Omni-7B"]
stage_configs = [str(Path(__file__).parent.parent / "e2e" / "stage_configs" / "qwen2_5_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.mark.core_model
@pytest.mark.benchmark
@hardware_test(res={"cuda": "L4"}, num_cards=3)
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
