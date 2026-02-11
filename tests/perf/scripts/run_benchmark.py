import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import (
    OmniServer,
)


def load_configs(config_path: str) -> list[dict[str, Any]]:
    try:
        abs_path = Path(config_path).resolve()
        with open(abs_path, encoding="utf-8") as f:
            configs = json.load(f)

        return configs

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {str(e)}")


def create_unique_server_params(configs: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    unique_params = set()
    for config in configs:
        test_name = config["test_name"]
        model = config["server_params"]["model"]
        stage_config_name = config["server_params"]["stage_config_name"]
        stage_config_path = str(Path(__file__).parent.parent / "stage_configs" / stage_config_name)
        unique_params.add((test_name, model, stage_config_path))

    return list(unique_params)


def create_test_parameter_mapping(configs: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict]:
    mapping = {}
    for config in configs:
        test_name = config["test_name"]
        model = config["server_params"]["model"]
        stage_config_name = config["server_params"]["stage_config_name"]
        stage_config_path = str(Path(__file__).parent.parent / "stage_configs" / stage_config_name)
        server_key = (test_name, model, stage_config_path)

        mapping[server_key] = {
            "test_name": test_name,
            "model": model,
            "stage_config_path": stage_config_path,
            "benchmark_params": config["benchmark_params"],
        }

    return mapping


CONFIG_FILE_PATH = str(Path(__file__).parent.parents / "tests" / "test.json")
BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)


test_params = create_unique_server_params(BENCHMARK_CONFIGS)
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS)

_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        _, model, stage_config_path = request.param

        print(f"Starting OmniServer with model: {model}")

        with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"]) as server:
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


@pytest.fixture
def benchmark_params(request, omni_server):
    test_name, model, stage_config_path = request.node.callspec.params["omni_server"]
    server_key = (test_name, model, stage_config_path)

    if server_key not in server_to_benchmark_mapping:
        raise ValueError(f"No benchmark parameters found for server key: {server_key}")

    config_data = server_to_benchmark_mapping[server_key]
    all_params = config_data["benchmark_params"]

    param_index = request.param if hasattr(request, "param") else 0

    if param_index < len(all_params):
        return {"test_name": config_data["test_name"], "model": config_data["model"], "params": all_params[param_index]}
    else:
        raise ValueError(f"No benchmark parameters found for index {param_index}")


def run_benchmark(args: list, test_name: str, qps: float) -> Any:
    """Generate synthetic image with random values."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"result_{test_name}_{qps}_{current_dt}.json"
    if "--result-filename" in args:
        print(f"The result file will be overwritten by {result_filename}")
    command = (
        ["vllm", "bench", "serve", "--omni"]
        + args
        + [
            "--backend",
            "openai-chat-omni",
            "--endpoint",
            "/v1/chat/completions",
            "--save-result",
            "--result-filename",
            result_filename,
        ]
    )
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end=" ")

    for line in iter(process.stderr.readline, ""):
        print(line, end=" ")

    if "--result-dir" in args:
        index = args.index("--result-dir")
        result_dir = args[index + 1]
    else:
        result_dir = "./"

    with open(os.path.join(result_dir, result_filename), encoding="utf-8") as f:
        result = json.load(f)
    return result


def create_benchmark_indices():
    indices = []
    for server_key, config_data in server_to_benchmark_mapping.items():
        params_list = config_data["benchmark_params"]
        indices.extend(range(len(params_list)))
    return indices


benchmark_indices = create_benchmark_indices()


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("benchmark_params", benchmark_indices, indirect=True)
def test_performance_benchmark(omni_server, benchmark_params):
    test_name = benchmark_params["test_name"]
    model = benchmark_params["model"]
    params = benchmark_params["params"]

    host = omni_server.host
    port = omni_server.port

    print(f"Running benchmark for model: {model}")
    print(f"Benchmark parameters: {benchmark_params}")

    for qps in params.get("qps", []):
        args = [
            "--host",
            host,
            "--port",
            str(port),
            "--dataset-name",
            params.get("dataset_name", "random"),
            "--num-prompts",
            str(params.get("num_prompts", 100)),
            "--random-input-len",
            str(params.get("random_input_len", 10)),
            "--random-output-len",
            str(params.get("random_output_len", 10)),
            "--percentile-metrics",
            params.get("percentile-metrics", "ttft,tpot,itl,e2el"),
            "--request-rate",
            str(qps),
        ]

        result = run_benchmark(args=args, test_name=test_name, qps=qps)
        assert result["completed"] == params.get("num_prompts"), "Request failures exist"
