import json
from pathlib import Path
from typing import Any

from tests.conftest import modify_stage_config


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
) -> list[tuple[str, str, str]]:
    unique_params = []
    seen = set()
    for config in configs:
        test_name = config["test_name"]
        model = config["server_params"]["model"]
        stage_config_name = config["server_params"].get("stage_config_name")
        if stage_config_name:
            stage_config_path = str(stage_configs_dir / stage_config_name)
            delete = config["server_params"].get("delete", None)
            update = config["server_params"].get("update", None)
            stage_config_path = modify_stage(stage_config_path, update, delete)
        else:
            stage_config_path = None

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
