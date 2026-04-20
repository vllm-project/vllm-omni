import json
import re
import time
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import OmniServerParams
from tests.dfx.reliability.conftest import list_remote_process_pids_by_pattern, post_chat_completions_raw
from vllm_omni.platforms import current_omni_platform


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


def create_unique_server_params(
    configs: list[dict[str, Any]],
    stage_configs_dir: Path,
) -> list[tuple[str, str, str | None, str | None, tuple[str, ...]]]:
    unique_params = []
    seen = set()
    for config in configs:
        test_name = config["test_name"]
        server_params = config["server_params"]
        model = server_params["model"]
        stage_config_name = server_params.get("stage_config_name")
        if stage_config_name:
            stage_config_path = str(stage_configs_dir / stage_config_name)
            delete = server_params.get("delete", None)
            update = server_params.get("update", None)
            stage_config_path = modify_stage(stage_config_path, update, delete)
        else:
            stage_config_path = None

        stage_overrides = server_params.get("stage_overrides")
        stage_overrides_json = json.dumps(stage_overrides) if stage_overrides else None

        # ``extra_cli_args`` passes raw CLI flags straight through to
        # ``vllm_omni.entrypoints.cli.main serve`` — used for flags that
        # don't map to stage-level overrides, e.g. ``--async-chunk`` /
        # ``--no-async-chunk`` toggling the deploy-level async_chunk bool.
        extra_cli_args = tuple(server_params.get("extra_cli_args") or ())

        server_param = (test_name, model, stage_config_path, stage_overrides_json, extra_cli_args)
        if server_param not in seen:
            seen.add(server_param)
            unique_params.append(server_param)

    return unique_params


def configs_with_platform_stage_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve stage config for XPU vs CUDA/ROCm."""
    out: list[dict[str, Any]] = []
    for config in configs:
        config_copy = json.loads(json.dumps(config))
        if current_omni_platform.is_xpu():
            config_copy["server_params"]["stage_config_name"] = "xpu/qwen3_omni_ci.yaml"
        out.append(config_copy)
    return out


def extract_server_args_by_test_name(configs: list[dict[str, Any]]) -> dict[str, list[str] | None]:
    mapping: dict[str, list[str] | None] = {}
    for cfg in configs:
        test_name = str(cfg.get("test_name"))
        server_params = cfg.get("server_params") or {}
        raw_args = server_params.get("server_args")
        mapping[test_name] = [str(item) for item in raw_args] if isinstance(raw_args, list) else None
    return mapping


def create_reliability_omni_server_params(configs: list[dict[str, Any]], stage_configs_dir: Path) -> list[OmniServerParams]:
    adjusted_configs = configs_with_platform_stage_configs(configs)
    unique_params = create_unique_server_params(adjusted_configs, stage_configs_dir)
    server_args_by_name = extract_server_args_by_test_name(adjusted_configs)
    return [
        OmniServerParams(
            model=model,
            stage_config_path=stage_config_path,
            server_args=server_args_by_name.get(test_name),
        )
        for test_name, model, stage_config_path in unique_params
    ]


def supports_video_generation(model_name: str) -> bool:
    lower = model_name.lower()
    return any(key in lower for key in ("wan", "video", "i2v", "t2v"))


def supports_chat_generation(model_name: str) -> bool:
    return not supports_video_generation(model_name)


def parse_stage_devices(stage_config_path: str) -> str:
    text = Path(stage_config_path).read_text(encoding="utf-8")
    raw_devices: list[str] = re.findall(r"^\s*devices:\s*\"?([0-9,\s]+)\"?\s*$", text, flags=re.MULTILINE)
    devices: set[int] = set()
    for item in raw_devices:
        for token in item.split(","):
            token = token.strip()
            if token:
                devices.add(int(token))
    if not devices:
        raise ValueError(f"No runtime.devices found in stage config: {stage_config_path}")
    return ",".join(str(x) for x in sorted(devices))


def resolve_oom_device_spec(config: dict[str, Any], stage_config_path: str | None) -> str:
    explicit = config.get("device")
    if explicit is not None:
        return str(explicit)
    if not stage_config_path:
        return "0"
    return parse_stage_devices(stage_config_path)


def assert_fault_exception(exc: Exception, error_keywords: tuple[str, ...]) -> None:
    text = str(exc).lower()
    assert any(key in text for key in error_keywords), f"unexpected error under fault injection: {exc}"


def wait_chat_request_ready(host: str, port: int, model: str, timeout_sec: int = 180) -> None:
    """Poll a minimal chat request until success."""
    deadline = time.time() + timeout_sec
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "stream": False,
            "modalities": ["text"],
        }
    )
    last_error: str | None = None
    while time.time() < deadline:
        try:
            status, body = post_chat_completions_raw(host, port, payload)
            if status == 200:
                return
            last_error = f"http={status} body={body[:200]!r}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"runtime-teardown warmup request did not succeed within {timeout_sec}s: {last_error}")


def assert_no_extra_worker_processes(
    baseline_pids: set[int],
    worker_pattern: str,
    timeout_sec: int = 60,
) -> None:
    """Ensure no extra worker PIDs remain after teardown."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        current = set(list_remote_process_pids_by_pattern(worker_pattern))
        extra = current - baseline_pids
        if not extra:
            return
        time.sleep(2)
    current = set(list_remote_process_pids_by_pattern(worker_pattern))
    extra = sorted(current - baseline_pids)
    assert not extra, f"orphan worker processes remain after container teardown: {extra}"


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


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register shared CLI options for DFX benchmark suites."""
    parser.addoption(
        "--test-config-file",
        action="store",
        default=None,
        help=("Path to benchmark config JSON. Example: --test-config-file tests/dfx/perf/tests/test_tts.json"),
    )
