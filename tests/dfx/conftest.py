import json
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
