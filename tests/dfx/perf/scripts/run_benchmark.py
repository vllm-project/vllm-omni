import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import OmniServer
from tests.dfx.conftest import (
    create_benchmark_indices,
    create_test_parameter_mapping,
    create_unique_server_params,
    get_benchmark_params_for_server,
    load_configs,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"


CONFIG_FILE_PATH = str(Path(__file__).parent.parent / "tests" / "test.json")
BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)


STAGE_CONFIGS_DIR = Path(__file__).parent.parent / "stage_configs"
test_params = create_unique_server_params(BENCHMARK_CONFIGS, STAGE_CONFIGS_DIR)
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

        server_args = ["--stage-init-timeout", "120", "--init-timeout", "900"]
        if stage_config_path:
            server_args = ["--stage-configs-path", stage_config_path] + server_args
        with OmniServer(model, server_args) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


def run_benchmark(
    args: list,
    test_name: str,
    flow,
    dataset_name: str,
    num_prompt,
) -> Any:
    """Run a single benchmark iteration and return the parsed result JSON."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"result_{test_name}_{dataset_name}_{flow}_{num_prompt}_{current_dt}.json"
    if "--result-filename" in args:
        print(f"The result file will be overwritten by {result_filename}")
    command = (
        ["vllm", "bench", "serve", "--omni"]
        + args
        + [
            "--save-result",
            "--result-dir",
            os.environ.get("BENCHMARK_DIR", "tests"),
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

    if "--result-dir" in command:
        index = command.index("--result-dir")
        result_dir = command[index + 1]
    else:
        result_dir = "./"

    with open(os.path.join(result_dir, result_filename), encoding="utf-8") as f:
        result = json.load(f)
    return result


benchmark_indices = create_benchmark_indices(BENCHMARK_CONFIGS, server_to_benchmark_mapping)


@pytest.fixture(params=benchmark_indices)
def benchmark_params(request, omni_server):
    """Benchmark parameters fixture with proper parametrization"""
    test_name, param_index = request.param

    if test_name != omni_server.test_name:
        pytest.skip(f"Skipping parameter for {test_name} - current server is {omni_server.test_name}")

    all_params = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)

    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")

    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")

    return {
        "test_name": test_name,
        "params": all_params[param_index],
    }


def _resolve_baseline_value(
    baseline_raw: Any,
    *,
    sweep_index: int | None,
    max_concurrency: Any = None,
    request_rate: Any = None,
) -> Any:
    """Pick the baseline threshold for this sweep step.

    Supported shapes per metric:
    - **Scalar** — same threshold for every concurrency / QPS.
    - **List** — aligned with ``max_concurrency`` / ``request_rate`` sweep order; use ``sweep_index``.
    - **Dict** — keyed by concurrency or rate, e.g. ``{"1": 500, "4": 800}`` (keys are strings in JSON).

    For dict lookup, ``max_concurrency`` is preferred when both are set (concurrency sweep).
    """
    if baseline_raw is None:
        # If no baseline is set, the maximum value will be used.
        return 100000
    if isinstance(baseline_raw, dict):
        if max_concurrency is not None:
            for key in (max_concurrency, str(max_concurrency)):
                if key in baseline_raw:
                    return baseline_raw[key]
        if request_rate is not None:
            for key in (request_rate, str(request_rate)):
                if key in baseline_raw:
                    return baseline_raw[key]
        raise KeyError(
            f"baseline dict has no key for max_concurrency={max_concurrency!r} "
            f"or request_rate={request_rate!r}; keys={list(baseline_raw.keys())!r}"
        )
    if isinstance(baseline_raw, (list, tuple)):
        return baseline_raw[sweep_index]
    return baseline_raw


def assert_result(
    result,
    params,
    num_prompt,
    *,
    sweep_index: int | None = None,
    max_concurrency: Any = None,
    request_rate: Any = None,
) -> None:
    assert result["completed"] == num_prompt, "Request failures exist"
    baseline_data = params.get("baseline", {})
    for metric_name, baseline_raw in baseline_data.items():
        current_value = result[metric_name]
        baseline_value = _resolve_baseline_value(
            baseline_raw,
            sweep_index=sweep_index,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
        )
        if "throughput" in metric_name:
            if current_value <= baseline_value:
                print(
                    f"ERROR: Throughput test results were below baseline: {metric_name}: {current_value} > {baseline_value}"
                )
        else:
            if current_value >= baseline_value:
                print(f"ERROR: Test results exceeded baseline: {metric_name}: {current_value} < {baseline_value}")


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

    # QPS test (sweep_index aligns with qps_list / num_prompt_list for this loop)
    for i, (qps, num_prompt) in enumerate(zip(qps_list, num_prompt_list)):
        args = args + ["--request-rate", str(qps), "--num-prompts", str(num_prompt)]
        result = run_benchmark(
            args=args,
            test_name=test_name,
            flow=qps,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
        )
        assert_result(
            result,
            params,
            num_prompt=num_prompt,
            sweep_index=i,
            request_rate=qps,
        )

    # concurrency test (sweep_index aligns with max_concurrency_list for separate thresholds per concurrency)
    for i, (concurrency, num_prompt) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        args = args + ["--max-concurrency", str(concurrency), "--num-prompts", str(num_prompt), "--request-rate", "inf"]
        result = run_benchmark(
            args=args,
            test_name=test_name,
            flow=concurrency,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
        )
        assert_result(
            result,
            params,
            num_prompt=num_prompt,
            sweep_index=i,
            max_concurrency=concurrency,
        )
