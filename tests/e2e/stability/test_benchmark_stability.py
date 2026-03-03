"""
长稳用例：在指定时长内通过 benchmark 向 vLLM-Omni 服务持续发送请求，校验服务稳定无崩溃、请求成功率。

形式与 tests/perf/scripts/run_benchmark.py 对齐：使用 @pytest.mark.parametrize + indirect fixture，
从 stability_config.json 读取 server 与 stability_benchmark_params，多组参数组合跑长稳。
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import OmniServer, modify_stage_config

STABILITY_DIR = Path(__file__).resolve().parent
# 与 perf 对齐：stage_configs 使用 tests/perf/stage_configs
STAGE_CONFIGS_BASE = STABILITY_DIR.parent.parent / "perf" / "stage_configs"
CONFIG_FILE_PATH = str(STABILITY_DIR / "stability_config.json")


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


def modify_stage(default_path: str, updates: dict | None, deletes: dict | None) -> str:
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
        stage_config_path = str(STAGE_CONFIGS_BASE / stage_config_name)
        delete = config["server_params"].get("delete", None)
        update = config["server_params"].get("update", None)
        stage_config_path = modify_stage(stage_config_path, update, delete)

        server_param = (test_name, model, stage_config_path)
        if server_param not in seen:
            seen.add(server_param)
            unique_params.append(server_param)
    return unique_params


def create_stability_params_mapping(configs: list[dict[str, Any]]) -> dict[str, dict]:
    mapping = {}
    for config in configs:
        test_name = config["test_name"]
        if test_name not in mapping:
            mapping[test_name] = {
                "test_name": test_name,
                "stability_benchmark_params": [],
            }
        mapping[test_name]["stability_benchmark_params"].extend(
            config.get("stability_benchmark_params", [])
        )
    return mapping


STABILITY_CONFIGS = load_configs(CONFIG_FILE_PATH)
test_params = create_unique_server_params(STABILITY_CONFIGS)
server_to_stability_mapping = create_stability_params_mapping(STABILITY_CONFIGS)

_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request):
    """启动 vLLM-Omni 服务，与 run_benchmark.py 的 omni_server 形式一致。"""
    with _omni_server_lock:
        test_name, model, stage_config_path = request.param

        print(f"Starting OmniServer for stability: {test_name}, model: {model}")

        with OmniServer(
            model,
            ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"],
        ) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


def get_stability_params_for_server(test_name: str) -> list[dict]:
    if test_name not in server_to_stability_mapping:
        return []
    return server_to_stability_mapping[test_name]["stability_benchmark_params"]


def create_stability_benchmark_indices() -> list[tuple[str, int]]:
    indices = []
    seen = set()
    for config in STABILITY_CONFIGS:
        test_name = config["test_name"]
        if test_name not in seen:
            seen.add(test_name)
            params_list = get_stability_params_for_server(test_name)
            for idx in range(len(params_list)):
                indices.append((test_name, idx))
    return indices


stability_benchmark_indices = create_stability_benchmark_indices()


@pytest.fixture(params=stability_benchmark_indices)
def stability_benchmark_params(request, omni_server):
    """长稳 benchmark 参数 fixture，与 run_benchmark.py 的 benchmark_params 形式一致。"""
    test_name, param_index = request.param

    if test_name != omni_server.test_name:
        pytest.skip(
            f"Skipping stability param for {test_name} - current server is {omni_server.test_name}"
        )

    all_params = get_stability_params_for_server(test_name)

    if not all_params:
        raise ValueError(f"No stability benchmark parameters found for test: {test_name}")

    if param_index >= len(all_params):
        raise ValueError(
            f"No stability benchmark parameters found for index {param_index} in test: {test_name}"
        )

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running stability benchmark {current}/{total} for {test_name}")

    return {"test_name": test_name, "params": all_params[param_index]}


def _params_to_script_args(params: dict, duration_sec: float) -> list[str]:
    """把 stability_benchmark_params 里的一项转成 run_benchmark_duration.py 的命令行参数。
    duration_sec 由上层传入（已考虑环境变量 STABILITY_BENCHMARK_DURATION_SEC 优先）。
    """
    exclude = {"duration_sec", "baseline"}
    args = []
    args.extend(["--duration-sec", str(int(duration_sec))])

    args.extend(["--dataset-name", params.get("dataset_name", "random")])
    args.extend(["--request-rate", str(params.get("request_rate", 1.0))])
    args.extend(["--num-prompts-per-batch", str(params.get("num_prompts_per_batch", 20))])

    for key, value in params.items():
        if key in exclude or value is None:
            continue
        if key in ("dataset_name", "request_rate", "num_prompts_per_batch", "duration_sec"):
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            args.append(arg_name)
            args.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        elif not isinstance(value, bool):
            args.append(arg_name)
            args.append(str(value))
    return args


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("stability_benchmark_params", stability_benchmark_indices, indirect=True)
def test_benchmark_stability_duration(omni_server, stability_benchmark_params, tmp_path):
    """在指定时长内循环跑 benchmark，断言无失败请求、服务未崩溃。"""
    script = STABILITY_DIR / "run_benchmark_duration.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")

    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]

    duration_sec = float(
        os.environ.get("STABILITY_BENCHMARK_DURATION_SEC", params.get("duration_sec", 300))
    )
    result_dir = tmp_path / "stability_bench_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    script_args = _params_to_script_args(params, duration_sec)
    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", "python"),
        str(script),
        "--host",
        omni_server.host,
        "--port",
        str(omni_server.port),
        "--model",
        omni_server.model,
        "--result-dir",
        str(result_dir),
    ] + script_args

    proc = subprocess.run(
        cmd,
        cwd=STABILITY_DIR.parent.parent.parent,
        capture_output=True,
        text=True,
        timeout=duration_sec + 600,
    )

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
    assert proc.returncode == 0, (
        f"run_benchmark_duration.py exited with {proc.returncode}. "
        f"stdout: {proc.stdout!r} stderr: {proc.stderr!r}"
    )

    summary_path = result_dir / "stability_summary.json"
    assert summary_path.exists(), f"Summary not written: {summary_path}"

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    total_completed = summary.get("total_completed", 0)
    total_failed = summary.get("total_failed", 0)
    assert total_completed > 0, "No request completed during stability run"
    assert total_failed == 0, (
        f"Stability run had {total_failed} failed requests (completed: {total_completed})"
    )
