import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.conftest import modify_stage_config


def flatten_server_serve_args(serve_args: dict[str, Any] | None) -> list[str]:
    """Turn ``server_params.serve_args`` into CLI tokens (``--flag`` / ``--flag value``)."""
    if not serve_args:
        return []
    out: list[str] = []
    for key, value in serve_args.items():
        flag = f"--{key.lstrip('-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif isinstance(value, dict):
            out.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
        elif isinstance(value, list):
            out.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
        else:
            out.extend([flag, str(value)])
    return out


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


def create_unique_server_params(
    configs: list[dict[str, Any]],
    stage_configs_dir: Path,
) -> list[tuple[str, str, list[str]]]:
    """Return unique ``(test_name, model, omni_serve_extras)`` for OmniServer.

    ``omni_serve_extras`` is appended after ``serve`` / ``--omni`` / host / port
    (for example ``--stage-configs-path …`` and/or diffusion ``--tensor-parallel-size``).
    Timeouts are added by each test harness (perf vs stability).
    """
    unique_params = []
    seen = set()
    for config in configs:
        test_name = config["test_name"]
        server_params = config["server_params"]
        model = server_params["model"]
        stage_config_name = server_params.get("stage_config_name")
        extras: list[str] = []
        if stage_config_name:
            stage_config_path = str(stage_configs_dir / stage_config_name)
            delete = server_params.get("delete", None)
            update = server_params.get("update", None)
            stage_config_path = modify_stage(stage_config_path, update, delete)
            extras.extend(["--stage-configs-path", stage_config_path])
        extras.extend(flatten_server_serve_args(server_params.get("serve_args")))

        dedupe_key = (test_name, model, tuple(extras))
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            unique_params.append((test_name, model, extras))

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


def get_benchmark_params_for_server(test_name: str, server_to_benchmark_mapping: dict[str, dict]) -> list:
    if test_name not in server_to_benchmark_mapping:
        return []
    return server_to_benchmark_mapping[test_name]["benchmark_params"]


def create_benchmark_indices(
    benchmark_configs: list[dict[str, Any]],
    server_to_benchmark_mapping: dict[str, dict],
) -> list[tuple[str, int]]:
    indices = []
    seen = set()
    for config in benchmark_configs:
        test_name = config["test_name"]
        if test_name not in seen:
            seen.add(test_name)
            params_list = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)
            for idx in range(len(params_list)):
                indices.append((test_name, idx))

    return indices


def run_benchmark(
    args: list[str],
    test_name: str,
    flow: Any,
    dataset_name: str,
    num_prompt: int,
) -> dict[str, Any]:
    """Run one ``vllm bench serve --omni`` iteration and return parsed metrics."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"result_{test_name}_{dataset_name}_{flow}_{num_prompt}_{current_dt}.json"
    if "--result-filename" in args:
        print(f"The result file will be overwritten by {result_filename}")
    command = (
        ["vllm", "bench", "serve", "--omni"]
        + args
        + [
            "--num-warmups",
            "2",
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
