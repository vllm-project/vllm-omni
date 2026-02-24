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

from tests.conftest import OmniServer, modify_stage_config


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


def modify_stage(default_path, updates, deletes):
    kwargs = {}
    if updates is not None:
        kwargs["updates"] = updates
    if deletes is not None:
        kwargs["deletes"] = deletes
    if kwargs:
        path = modify_stage_config(default_path, **kwargs)
    else:
        path = default_path

    return path


def create_unique_server_params(configs: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    unique_params = []
    seen = set()
    for config in configs:
        test_name = config["test_name"]
        model = config["server_params"]["model"]
        stage_config_name = config["server_params"]["stage_config_name"]
        stage_config_path = str(Path(__file__).parent.parent / "stage_configs" / stage_config_name)
        delete = config["server_params"].get("delete", None)
        update = config["server_params"].get("update", None)
        stage_config_path = modify_stage(stage_config_path, update, delete)

        server_param = (test_name, model, stage_config_path)
        if server_param not in seen:
            seen.add(server_param)
            unique_params.append(server_param)

    return unique_params


def create_test_parameter_mapping(configs: list[dict[str, Any]]) -> dict[str, dict]:
    mapping = {}
    for config in configs:
        test_name = config["test_name"]
        if test_name not in mapping:
            mapping[test_name] = {
                "test_name": test_name,
                "benchmark_params": [],
            }
        mapping[test_name]["benchmark_params"].extend(config["benchmark_params"])
    return mapping


CONFIG_FILE_PATH = str(Path(__file__).parent.parent / "tests" / "test.json")
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
        test_name, model, stage_config_path = request.param

        print(f"Starting OmniServer with test: {test_name}, model: {model}")

        with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"]) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


def run_benchmark(args: list, test_name: str, flow, dataset_name: str, num_prompt) -> Any:
    """Generate synthetic image with random values."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"result_{test_name}_{dataset_name}_{flow}_{num_prompt}_{current_dt}.json"
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


def get_benchmark_params_for_server(test_name: str) -> list:
    if test_name not in server_to_benchmark_mapping:
        return []
    return server_to_benchmark_mapping[test_name]["benchmark_params"]


def create_benchmark_indices():
    indices = []
    seen = set()
    for config in BENCHMARK_CONFIGS:
        test_name = config["test_name"]
        if test_name not in seen:
            seen.add(test_name)
            params_list = get_benchmark_params_for_server(test_name)
            for idx in range(len(params_list)):
                indices.append((test_name, idx))

    return indices


benchmark_indices = create_benchmark_indices()


@pytest.fixture(params=benchmark_indices)
def benchmark_params(request, omni_server):
    """Benchmark parameters fixture with proper parametrization"""
    test_name, param_index = request.param

    if test_name != omni_server.test_name:
        pytest.skip(f"Skipping parameter for {test_name} - current server is {omni_server.test_name}")

    all_params = get_benchmark_params_for_server(test_name)

    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")

    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")

    return {"test_name": test_name, "params": all_params[param_index]}


def assert_result(result, params, num_prompt):
    assert result["completed"] == num_prompt, "Request failures exist"
    baseline_data = params.get("baseline", {})
    for metric_name, baseline_value in baseline_data.items():
        current_value = result[metric_name]
        if "throughput" in metric_name:
            assert current_value >= baseline_value, f"{metric_name}: {current_value} < {baseline_value}"
        else:
            assert current_value <= baseline_value, f"{metric_name}: {current_value} > {baseline_value}"


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("benchmark_params", benchmark_indices, indirect=True)
def test_performance_benchmark(omni_server, benchmark_params):
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    dataset_name = params.get("dataset_name", "")

    host = omni_server.host
    port = omni_server.port
    model = omni_server.model

    print(f"Running benchmark for model: {model}")
    print(f"Benchmark parameters: {benchmark_params}")

    def to_list(value, default=None):
        if value is None:
            return [] if default is None else [default]
        return [value] if not isinstance(value, (list, tuple)) else list(value)

    qps_list = to_list(params.get("request_rate"))
    num_prompt_list = to_list(params.get("num_prompts"))
    max_concurrency_list = to_list(params.get("max_concurrency"))

    max_len = max(len(qps_list), len(max_concurrency_list))
    if len(num_prompt_list) == 1 and max_len > 1:
        num_prompt_list = num_prompt_list * max_len
    elif max_len == 1 and len(num_prompt_list) > 1:
        if len(qps_list) == 1:
            qps_list = qps_list * len(num_prompt_list)
        if len(max_concurrency_list) == 1:
            max_concurrency_list = max_concurrency_list * len(num_prompt_list)
        max_len = max(len(qps_list), len(max_concurrency_list))
    elif len(num_prompt_list) != max_len and max_len > 0:
        raise ValueError("The number of prompts does not match the QPS or max_concurrency")

    args = ["--host", host, "--port", str(port)]
    exclude_keys = {"request_rate", "baseline", "num_prompts", "max_concurrency"}

    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue

        arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            json_str = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            args.extend([arg_name, json_str])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])

    # QPS test
    for qps, num_prompt in zip(qps_list, num_prompt_list):
        args = args + ["--request-rate", str(qps), "--num-prompts", str(num_prompt)]
        result = run_benchmark(
            args=args, test_name=test_name, flow=qps, dataset_name=dataset_name, num_prompt=num_prompt
        )
        assert_result(result, params, num_prompt=num_prompt)

    # concurrency test
    for concurrency, num_prompt in zip(max_concurrency_list, num_prompt_list):
        args = args + ["--max-concurrency", str(concurrency), "--num-prompts", str(num_prompt), "--request-rate", "inf"]
        result = run_benchmark(
            args=args, test_name=test_name, flow=concurrency, dataset_name=dataset_name, num_prompt=num_prompt
        )
        assert_result(result, params, num_prompt=num_prompt)
