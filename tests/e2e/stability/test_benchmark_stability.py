"""
长稳用例：先起 OmniServer，再在指定时长内按 request-rate 或 max-concurrency 跑 benchmark，
超过时长后不再发新请求，断言无失败请求。

与 perf 逻辑一致：load_configs、modify_stage、create_unique_server_params、create_test_parameter_mapping、
get_benchmark_params_for_server、create_benchmark_indices、omni_server fixture 与 perf 相同，
仅 run_benchmark（此处为 run_stability_benchmark 带时长）和测试用例不同。不修改 tests/perf。

时长可由环境变量 STABILITY_BENCHMARK_DURATION_SEC 覆盖配置中的 duration_sec（默认 300）。
"""
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import OmniServer, modify_stage_config

STABILITY_DIR = Path(__file__).resolve().parent
STAGE_CONFIGS_DIR = STABILITY_DIR / "stage_configs"
CONFIG_FILE_PATH = str(STABILITY_DIR / "stability_test.json")


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
        stage_config_path = str(STAGE_CONFIGS_DIR / stage_config_name)
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


try:
    BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)
except FileNotFoundError:
    BENCHMARK_CONFIGS = []

test_params = create_unique_server_params(BENCHMARK_CONFIGS) if BENCHMARK_CONFIGS else []
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS) if BENCHMARK_CONFIGS else {}

_omni_server_lock = threading.Lock()


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


@pytest.fixture(params=benchmark_indices)
def stability_benchmark_params(request, omni_server):
    """Benchmark parameters fixture with proper parametrization (same as perf)."""
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


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("stability_benchmark_params", benchmark_indices, indirect=True)
def test_benchmark_stability(omni_server, stability_benchmark_params):
    """在指定时长内按 request-rate 或 max-concurrency 跑 benchmark，断言无失败请求。"""
    from tests.e2e.stability.run_benchmark_duration import run_stability_benchmark

    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]
    duration_sec = int(os.environ.get("STABILITY_BENCHMARK_DURATION_SEC", params.get("duration_sec", 300)))
    request_rate = params.get("request_rate")
    max_concurrency = params.get("max_concurrency")

    bench_params = {
        k: v
        for k, v in params.items()
        if k not in ("duration_sec", "request_rate", "max_concurrency")
    }

    result = run_stability_benchmark(
        host=omni_server.host,
        port=omni_server.port,
        duration_sec=duration_sec,
        params=bench_params,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        result_dir=str(STABILITY_DIR),
    )

    assert result.get("failed", 0) == 0, f"存在失败请求: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, "未完成任何请求"
