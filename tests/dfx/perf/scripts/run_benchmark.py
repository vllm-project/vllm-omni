# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    create_paired_benchmark_pytest_params,
    create_paired_omni_benchmark_pytest_params,
    create_test_parameter_mapping,
    get_benchmark_params_for_server,
    get_runtime_resource_label,
    is_diffusion_perf_config,
    load_benchmark_configs,
    run_benchmark,
)
from tests.dfx.perf.scripts.sglang_omni_server import SglangOmniServer, sglang_server_entries

# Optional JSON field ``mark`` is applied as pytest marks via
# ``create_paired_omni_benchmark_pytest_params`` (e.g. ``"mark": [{"hardware_marks":
# {"res": {"cuda": "H100"}, "num_cards": 2}}, "full_model", "omni"]``).


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OMNI_BENCHMARK_SCRIPT = _REPO_ROOT / "benchmarks" / "omni" / "omni_benchmark_serving.py"


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
_server_types = {config.get("server_type", "vllm-omni") for config in BENCHMARK_CONFIGS}
if len(_server_types) != 1:
    raise ValueError(f"A benchmark config file must use one server type, got: {sorted(_server_types)}")
SERVER_TYPE = next(iter(_server_types), "vllm-omni")
if SERVER_TYPE == "sglang-omni":

    def _sglang_marks(config: dict[str, Any]) -> list[pytest.MarkDecorator]:
        marks: list[pytest.MarkDecorator] = []
        for item in config.get("mark", []):
            if isinstance(item, str):
                marks.append(getattr(pytest.mark, item))
                continue
            hardware = item.get("hardware_marks", {})
            for platform, resource in hardware.get("res", {}).items():
                marks.extend((getattr(pytest.mark, platform), getattr(pytest.mark, resource)))
            cards = hardware.get("num_cards")
            if isinstance(cards, int):
                marks.append(getattr(pytest.mark, f"cards_{cards}"))
        return marks

    _server_entries = sglang_server_entries(BENCHMARK_CONFIGS)
    paired_benchmark_params = create_paired_benchmark_pytest_params(
        _server_entries,
        {name: get_benchmark_params_for_server(name, server_to_benchmark_mapping) for _, name in _server_entries},
        {config["test_name"]: _sglang_marks(config) for config in BENCHMARK_CONFIGS},
    )
else:
    paired_benchmark_params = create_paired_omni_benchmark_pytest_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR)

_omni_server_lock = threading.Lock()


class _SingleActiveContext:
    """Reuse one active context while its configuration key is unchanged."""

    def __init__(self) -> None:
        self._key: Any = None
        self._stack: ExitStack | None = None
        self._value: Any = None

    def acquire(self, key: Any, factory: Callable[[], AbstractContextManager[Any]]) -> Any:
        if self._stack is not None and key == self._key:
            return self._value

        self.close()
        stack = ExitStack()
        value = stack.enter_context(factory())
        self._key = key
        self._stack = stack
        self._value = value
        return value

    def close(self) -> None:
        stack = self._stack
        self._key = None
        self._stack = None
        self._value = None
        if stack is not None:
            stack.close()


@contextmanager
def _start_omni_server(server_param):
    from tests.helpers.runtime import OmniServer

    test_name, model, stage_config_path, stage_overrides, extra_cli_args, use_omni = server_param

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


@pytest.fixture(scope="module")
def omni_server_context():
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Reuse it for adjacent benchmark cases with the same server configuration.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        active_context = _SingleActiveContext()
        try:
            yield active_context
        finally:
            active_context.close()


@pytest.fixture
def omni_server(request, omni_server_context):
    if SERVER_TYPE == "sglang-omni":
        key = json.dumps(request.param, sort_keys=True)
        return omni_server_context.acquire(key, lambda: SglangOmniServer(request.param))
    return omni_server_context.acquire(request.param, lambda: _start_omni_server(request.param))


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


def _resolve_num_warmups(params: dict[str, Any], *, default: int) -> int:
    value = params.get("num_warmups")
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("num_warmups must be a non-negative integer")
    return value


def assert_result(result, params, num_prompt) -> None:
    assert result["completed"] == num_prompt, "Request failures exist"
    if params.get("dataset_name") == "omniinteract":
        summary = result.get("omniinteract")
        assert isinstance(summary, dict), "OmniInteract summary is missing"
        assert (summary.get("total"), summary.get("success"), summary.get("failed")) == (
            num_prompt,
            num_prompt,
            0,
        ), "OmniInteract requests did not all succeed"
        assert summary.get("artifacts_complete") is True, "OmniInteract artifacts are incomplete"
    baseline = params.get("baseline")
    hardware = result.get("Hardware")
    hardware_baseline = baseline.get(hardware) if isinstance(baseline, dict) and isinstance(hardware, str) else None
    if isinstance(hardware_baseline, dict) and "mean_tpot_ms" in hardware_baseline:
        num_tpot_samples = result.get("num_tpot_samples")
        mean_tpot_ms = result.get("mean_tpot_ms")
        assert isinstance(num_tpot_samples, int) and not isinstance(num_tpot_samples, bool) and num_tpot_samples > 0, (
            "TPOT baseline is configured, but no measurable TPOT samples were produced"
        )
        assert (
            isinstance(mean_tpot_ms, int | float) and not isinstance(mean_tpot_ms, bool) and math.isfinite(mean_tpot_ms)
        ), "TPOT baseline is configured, but mean_tpot_ms is not finite"
    expected_audio_turns = params.get("expected_duplex_audio_turns_per_session")
    if expected_audio_turns is not None:
        session_metrics = result.get("duplex_session_metrics")
        assert isinstance(session_metrics, list), "Duplex session metrics are missing"
        assert len(session_metrics) == num_prompt, (
            f"Expected {num_prompt} duplex session metric rows, got {len(session_metrics)}"
        )
        assert all(
            isinstance(metric, dict) and metric.get("audio_turn_count") == expected_audio_turns
            for metric in session_metrics
        ), f"Not every duplex session emitted {expected_audio_turns} audio turns"


def _run_sglang_omni_benchmark(
    *,
    args: list[str],
    test_name: str,
    flow: Any,
    dataset_name: str,
    num_prompt: int,
    **_: Any,
) -> dict[str, Any]:
    """Drive SGLang-Omni without requiring a ``vllm`` console script on PATH."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = Path(os.environ.get("BENCHMARK_DIR", "tests/dfx/perf/results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"result_{test_name}_{dataset_name}_{flow}_{num_prompt}_{timestamp}.json"
    command = [
        sys.executable,
        str(_OMNI_BENCHMARK_SCRIPT),
        "--request-backend",
        "sglang_omni",
        "--output-file",
        str(result_path),
        *args,
    ]
    subprocess.run(command, cwd=_REPO_ROOT, check=True)
    with result_path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def _dispatch_run_benchmark(**kwargs: Any) -> dict[str, Any]:
    if SERVER_TYPE == "sglang-omni":
        return _run_sglang_omni_benchmark(**kwargs)
    return run_benchmark(**kwargs)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "omni_server,benchmark_params",
    paired_benchmark_params,
    indirect=["omni_server", "benchmark_params"],
)
def test_performance_benchmark(omni_server, benchmark_params):
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    dataset_name = params.get("dataset_name", "")

    host = omni_server.host
    port = omni_server.port
    model = omni_server.model

    print(f"Running benchmark for model: {model}")
    print(f"Benchmark parameters: {benchmark_params}")

    resource_label = omni_server.resource_label if SERVER_TYPE == "sglang-omni" else get_runtime_resource_label()

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

    args = ["--host", host, "--port", str(port), "--model", model]
    exclude_keys = {
        "request_rate",
        "baseline",
        "num_prompts",
        "max_concurrency",
        "num_warmups",
        "task",
        "enabled",
        "eval_phase",
        "trust_remote_code",
        "expected_duplex_audio_turns_per_session",
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
    for sweep_index, (qps, num_prompt) in enumerate(zip(qps_list, num_prompt_list)):
        args = args + ["--request-rate", str(qps), "--num-prompts", str(num_prompt)]
        result = _dispatch_run_benchmark(
            args=args,
            test_name=test_name,
            flow=qps,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            baseline_config=params.get("baseline"),
            sweep_index=sweep_index,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
            resource_label=resource_label,
            num_warmups=_resolve_num_warmups(params, default=2),
        )
        assert_result(result, params, num_prompt)

    # concurrency test
    for sweep_index, (concurrency, num_prompt) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        args = args + ["--max-concurrency", str(concurrency), "--num-prompts", str(num_prompt), "--request-rate", "inf"]
        result = _dispatch_run_benchmark(
            args=args,
            test_name=test_name,
            flow=concurrency,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            baseline_config=params.get("baseline"),
            sweep_index=sweep_index,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
            resource_label=resource_label,
            num_warmups=_resolve_num_warmups(params, default=max(2, int(concurrency))),
        )
        assert_result(result, params, num_prompt)
