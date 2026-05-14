import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import list_remote_process_pids_by_pattern, post_chat_completions_raw
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import modify_stage_config
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


def modify_stage(default_path: str, updates: dict[str, Any] | None, deletes: dict[str, Any] | None) -> str:
    kwargs: dict[str, Any] = {}
    if updates is not None:
        kwargs["updates"] = updates
    if deletes is not None:
        kwargs["deletes"] = deletes
    if kwargs:
        return modify_stage_config(default_path, **kwargs)
    return default_path


def _build_serve_args(serve_args: Any) -> list[str]:
    """Convert server_params.serve_args to a flat CLI args list."""
    if serve_args is None:
        return []
    if isinstance(serve_args, list):
        return [str(item) for item in serve_args]
    if not isinstance(serve_args, dict):
        raise TypeError(f"serve_args must be dict/list/None, got {type(serve_args).__name__}")

    args: list[str] = []
    for key, value in serve_args.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            args.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
            continue
        args.extend([flag, str(value)])
    return args


def create_unique_server_params(
    configs: list[dict[str, Any]],
    stage_configs_dir: Path,
) -> list[tuple[str, str, str | None, str | None, tuple[str, ...], bool]]:
    """Return one row per unique server configuration.

    ``(test_name, model, deploy_yaml_path, stage_overrides_json, extra_cli_args, use_omni)``.

    JSON ``server_params.serve_args`` (dict/list) is expanded via ``_build_serve_args``
    and **prepended** to ``extra_cli_args`` so perf / stability ``omni_server`` fixtures
    stay identical to main while still honoring ``serve_args`` in benchmark JSON.
    """
    unique_params: list[tuple[str, str, str | None, str | None, tuple[str, ...], bool]] = []
    seen: set[tuple[str, str, str | None, str | None, tuple[str, ...], bool]] = set()
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
        serve_flat = _build_serve_args(server_params.get("serve_args"))
        raw_extra = tuple(server_params.get("extra_cli_args") or ())
        extra_cli_args = tuple(serve_flat) + raw_extra
        use_omni = bool(server_params.get("use_omni", True))

        server_param = (
            test_name,
            model,
            stage_config_path,
            stage_overrides_json,
            extra_cli_args,
            use_omni,
        )
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


def create_reliability_omni_server_params(
    configs: list[dict[str, Any]], stage_configs_dir: Path
) -> list[OmniServerParams]:
    adjusted_configs = configs_with_platform_stage_configs(configs)
    unique_params = create_unique_server_params(adjusted_configs, stage_configs_dir)
    server_args_by_name = extract_server_args_by_test_name(adjusted_configs)
    return [
        OmniServerParams(
            model=model,
            stage_config_path=stage_config_path,
            server_args=server_args_by_name.get(test_name),
            use_omni=use_omni,
        )
        for test_name, model, stage_config_path, _stage_overrides_json, _extra_cli_args, use_omni in unique_params
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
        for entry in config["benchmark_params"]:
            # Skip disabled entries
            if not entry.get("enabled", True):
                continue
            mapping[test_name]["benchmark_params"].append(entry)
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


def _safe_filename_token(value: Any | None, *, default: str = "na") -> str:
    """Make a single path segment safe for result filenames on common filesystems."""
    if value is None:
        return default
    s = str(value).strip()
    for bad in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        s = s.replace(bad, "_")
    return s if s else default


def _resolve_baseline_value(
    baseline_raw: Any,
    *,
    sweep_index: int | None,
    max_concurrency: Any = None,
    request_rate: Any = None,
) -> Any:
    """Pick the baseline threshold for this sweep step."""
    if baseline_raw is None:
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
        if sweep_index is None:
            raise ValueError("list baseline requires sweep_index")
        if not (0 <= sweep_index < len(baseline_raw)):
            raise IndexError(f"baseline list len={len(baseline_raw)} has no index {sweep_index}")
        return baseline_raw[sweep_index]
    return baseline_raw


def _baseline_thresholds_for_step(
    baseline_data: dict[str, Any],
    *,
    sweep_index: int | None = None,
    max_concurrency: Any = None,
    request_rate: Any = None,
) -> dict[str, Any]:
    """Resolve baseline config to one threshold per metric for this iteration."""
    return {
        metric_name: _resolve_baseline_value(
            baseline_raw,
            sweep_index=sweep_index,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
        )
        for metric_name, baseline_raw in baseline_data.items()
    }


def run_benchmark(
    args: list[str],
    test_name: str,
    flow: Any,
    dataset_name: str,
    num_prompt: int,
    *,
    baseline_config: dict[str, Any] | None = None,
    sweep_index: int | None = None,
    request_rate: Any | None = None,
    max_concurrency: Any | None = None,
    random_input_len: Any | None = None,
    random_output_len: Any | None = None,
) -> dict[str, Any]:
    """Run one ``vllm bench serve --omni`` iteration and return parsed metrics."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    ri = _safe_filename_token(random_input_len)
    ro = _safe_filename_token(random_output_len)
    result_filename = f"result_{test_name}_{dataset_name}_{flow}_{num_prompt}_in{ri}_out{ro}_{current_dt}.json"
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

    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to capture benchmark process output streams")

    def _forward_stream(stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                print(line, end="")
        finally:
            stream.close()

    stdout_thread = threading.Thread(target=_forward_stream, args=(process.stdout,))
    stderr_thread = threading.Thread(target=_forward_stream, args=(process.stderr,))
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    process.wait()

    if "--result-dir" in command:
        index = command.index("--result-dir")
        result_dir = command[index + 1]
    else:
        result_dir = "./"

    result_path = os.path.join(result_dir, result_filename)
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    if baseline_config:
        result["baseline"] = _baseline_thresholds_for_step(
            baseline_config,
            sweep_index=sweep_index,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
        )
    else:
        result["baseline"] = {}
    if random_input_len is not None:
        result["random_input_len"] = random_input_len
    if random_output_len is not None:
        result["random_output_len"] = random_output_len
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register shared CLI options for DFX benchmark suites."""
    parser.addoption(
        "--test-config-file",
        action="store",
        default=None,
        help=("Path to benchmark config JSON. Example: --test-config-file tests/dfx/perf/tests/test_tts.json"),
    )
