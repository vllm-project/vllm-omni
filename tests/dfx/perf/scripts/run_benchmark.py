import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    create_paired_omni_benchmark_pytest_params,
    create_test_parameter_mapping,
    get_benchmark_params_for_server,
    get_runtime_resource_label,
    is_diffusion_perf_config,
    load_benchmark_configs,
    resolve_baseline_value,
    run_benchmark,
)
from tests.helpers.runtime import OmniServer

# Compare metrics to each test JSON ``baseline`` block only when pytest is run with ``--assert-baseline``
# (registered in ``tests/dfx/conftest.py``; default: off). ``run_benchmark`` and ``_resolve_baseline_value`` are
# defined in the same module.
#
# Optional JSON field ``mark`` is applied as pytest marks via
# ``create_paired_omni_benchmark_pytest_params`` (e.g. ``"mark": [{"hardware_marks":
# {"res": {"cuda": "H100"}, "num_cards": 2}}, "full_model", "omni"]``).


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _get_config_file_from_argv() -> str | None:
    """Read ``--test-config-file`` from ``sys.argv`` at import time so parametrization can use it."""
    import sys

    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]
    return None


_PERF_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

CONFIG_FILE_PATH = _get_config_file_from_argv()
if CONFIG_FILE_PATH is None:
    _all_configs = load_benchmark_configs(config_dir=_PERF_TESTS_DIR)
    BENCHMARK_CONFIGS = [cfg for cfg in _all_configs if not is_diffusion_perf_config(cfg)]
    print(
        f"No --test-config-file: loaded {len(BENCHMARK_CONFIGS)} omni/tts case(s) from "
        f"{_PERF_TESTS_DIR}/*.json (skipped {len(_all_configs) - len(BENCHMARK_CONFIGS)} diffusion; "
        f"use -m to filter, e.g. -m tts)"
    )
else:
    BENCHMARK_CONFIGS = load_benchmark_configs(CONFIG_FILE_PATH)

DEPLOY_CONFIGS_DIR = Path(__file__).parent.parent / "deploy"
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS)
paired_benchmark_params = create_paired_omni_benchmark_pytest_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR)

_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        test_name, model, stage_config_path, stage_overrides, extra_cli_args, use_omni = request.param

        print(f"Starting OmniServer with test: {test_name}, model: {model}")

        server_args: list[str] = []
        if use_omni:
            server_args += ["--stage-init-timeout", "600", "--init-timeout", "900"]
        # --deploy-config and --stage-overrides compose at the CLI (see vllm_omni/entrypoints/utils.py):
        # deploy-config sets the base; stage-overrides are applied on top. Both can be set.
        if stage_config_path:
            server_args = ["--deploy-config", stage_config_path] + server_args
        if stage_overrides:
            server_args = ["--stage-overrides", stage_overrides] + server_args
        if extra_cli_args:
            server_args = list(extra_cli_args) + server_args
        with OmniServer(model, server_args, use_omni=use_omni) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


@pytest.fixture
def benchmark_params(request):
    """Benchmark parameters fixture; paired with ``omni_server`` via parametrization."""
    test_name, param_index = request.param

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


def assert_result(
    result,
    params,
    num_prompt,
    *,
    assert_baseline: bool,
    sweep_index: int | None = None,
    max_concurrency: Any | None = None,
    request_rate: Any | None = None,
) -> None:
    assert result["completed"] == num_prompt, "Request failures exist"
    if not assert_baseline:
        return
    baseline_data = params.get("baseline", {})
    for metric_name, baseline_raw in baseline_data.items():
        current_value = result[metric_name]
        baseline_value = resolve_baseline_value(
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


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "omni_server,benchmark_params",
    paired_benchmark_params,
    indirect=["omni_server", "benchmark_params"],
)
def test_performance_benchmark(omni_server, benchmark_params, request):
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    dataset_name = params.get("dataset_name", "")

    host = omni_server.host
    port = omni_server.port
    model = omni_server.model

    print(f"Running benchmark for model: {model}")
    print(f"Benchmark parameters: {benchmark_params}")

    assert_baseline = request.config.getoption("--assert-baseline", default=False)
    resource_label = get_runtime_resource_label()

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
    exclude_keys = {
        "request_rate",
        "baseline",
        "num_prompts",
        "max_concurrency",
        "task",
        "enabled",
        "eval_phase",
        "trust_remote_code",
    }

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

    for config in BENCHMARK_CONFIGS:
        if config.get("test_name") != test_name:
            continue
        server_params = config.get("server_params") or {}
        if server_params.get("trust_remote_code") or params.get("trust_remote_code"):
            args.append("--trust-remote-code")
        break

    # QPS / request-rate sweep
    for i, (qps, num_prompt) in enumerate(zip(qps_list, num_prompt_list)):
        args = args + ["--request-rate", str(qps), "--num-prompts", str(num_prompt)]
        result = run_benchmark(
            args=args,
            test_name=test_name,
            flow=qps,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            baseline_config=params.get("baseline"),
            sweep_index=i,
            request_rate=qps,
            max_concurrency=None,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
            resource_label=resource_label,
        )
        assert_result(
            result,
            params,
            num_prompt,
            assert_baseline=assert_baseline,
            sweep_index=i,
            request_rate=qps,
        )

    # concurrency test
    for i, (concurrency, num_prompt) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        args = args + ["--max-concurrency", str(concurrency), "--num-prompts", str(num_prompt), "--request-rate", "inf"]
        result = run_benchmark(
            args=args,
            test_name=test_name,
            flow=concurrency,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            baseline_config=params.get("baseline"),
            sweep_index=i,
            request_rate=None,
            max_concurrency=concurrency,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
            resource_label=resource_label,
        )
        assert_result(
            result,
            params,
            num_prompt,
            assert_baseline=assert_baseline,
            sweep_index=i,
            max_concurrency=concurrency,
        )
