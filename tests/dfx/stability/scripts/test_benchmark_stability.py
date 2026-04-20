"""
Stability test cases: start OmniServer first, then run benchmark traffic with either
`request-rate` or `max-concurrency` for a fixed duration. No new requests are sent
after the duration is reached, and the test asserts that there are no failed requests.

The overall flow matches the perf logic: `load_configs`, `modify_stage`,
`create_unique_server_params`, `create_test_parameter_mapping`,
`get_benchmark_params_for_server`, `create_benchmark_indices`, and the
`omni_server` fixture are aligned with perf. Only the benchmark execution
(`run_stability_benchmark`, which is duration-based here) and the test cases differ.

All test-specific parameters, such as `duration_sec`, `request_rate` /
`max_concurrency`, and `num_prompts_per_batch`, are configured in
`tests/dfx/stability/tests/test.json` and are no longer overridden
through environment variables.
"""

import json
import os
import threading
import time
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
from tests.dfx.perf.scripts.run_benchmark import run_benchmark

STABILITY_DIR = Path(__file__).resolve().parent.parent
DEPLOY_CONFIGS_DIR = STABILITY_DIR / "deploy"
CONFIG_FILE_PATH = str(STABILITY_DIR / "tests" / "test.json")
DEFAULT_NUM_PROMPTS_PER_BATCH = 20


try:
    BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)
except FileNotFoundError:
    BENCHMARK_CONFIGS = []

test_params = create_unique_server_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR) if BENCHMARK_CONFIGS else []
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS) if BENCHMARK_CONFIGS else {}

_omni_server_lock = threading.Lock()


benchmark_indices = create_benchmark_indices(BENCHMARK_CONFIGS, server_to_benchmark_mapping)


def _build_base_args(params: dict[str, Any], host: str, port: int) -> list[str]:
    exclude = {
        "request_rate",
        "max_concurrency",
        "num_prompts",
        "baseline",
        "duration_sec",
        "num_prompts_per_batch",
    }
    args = ["--host", host, "--port", str(port)]
    for key, value in params.items():
        if key in exclude or value is None:
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            args.extend([arg_name, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])
    return args


def _run_one_benchmark_batch(
    host: str,
    port: int,
    params: dict[str, Any],
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    result_dir: str,
    batch_index: int,
) -> dict[str, Any]:
    base = _build_base_args(params, host, port)
    if request_rate is not None:
        args = base + ["--request-rate", str(request_rate), "--num-prompts", str(num_prompts)]
        flow = request_rate
    else:
        args = base + [
            "--max-concurrency",
            str(max_concurrency),
            "--num-prompts",
            str(num_prompts),
            "--request-rate",
            "inf",
        ]
        flow = max_concurrency

    dataset_name = params.get("dataset_name", "random")
    old_benchmark_dir = os.environ.get("BENCHMARK_DIR")
    try:
        os.environ["BENCHMARK_DIR"] = result_dir
        result = run_benchmark(
            args=args,
            test_name="stability",
            flow=flow,
            dataset_name=dataset_name,
            num_prompt=num_prompts,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
        )
        return result
    except (FileNotFoundError, OSError) as e:
        # Surface batch failure so the stability test does not false-pass when
        # run_benchmark fails before writing JSON (e.g. command not found).
        return {
            "completed": 0,
            "failed": 1,
            "duration": 0.0,
            "errors": [f"Benchmark batch failed: {type(e).__name__}: {e}"],
        }
    finally:
        if old_benchmark_dir is not None:
            os.environ["BENCHMARK_DIR"] = old_benchmark_dir
        elif "BENCHMARK_DIR" in os.environ:
            os.environ.pop("BENCHMARK_DIR")


def _merge_batch_results(batch_results: list[dict[str, Any]], total_duration_sec: float) -> dict[str, Any]:
    if not batch_results:
        return {"completed": 0, "failed": 0, "duration": total_duration_sec, "errors": []}

    completed = sum(r.get("completed", 0) for r in batch_results)
    failed = sum(r.get("failed", 0) for r in batch_results)
    merged: dict[str, Any] = {
        "completed": completed,
        "failed": failed,
        "duration": total_duration_sec,
        "errors": [],
    }
    for r in batch_results:
        merged["errors"].extend(r.get("errors") or [])
    return merged


def _print_merged_report(result: dict[str, Any]) -> None:
    """Print the final summary: successful requests, failed requests, and total duration only."""
    fmt = "{:<40} {:<10}"
    fmt_float = "{:<40} {:<10.2f}"
    completed = result.get("completed", 0)
    failed = result.get("failed", 0)
    duration = float(result.get("duration", 0.0) or 0.0)
    print("\n============ Stability Benchmark Summary ============")
    print(fmt.format("Successful requests:", completed))
    print(fmt.format("Failed requests:", failed))
    print(fmt_float.format("Total duration (s):", duration))
    print("==================================================\n")


def run_stability_benchmark(
    host: str,
    port: int,
    duration_sec: int | float,
    params: dict[str, Any],
    *,
    request_rate: float | None = None,
    max_concurrency: int | None = None,
    result_filename: str | None = None,
    result_dir: str = "./",
    num_prompts_per_batch: int = DEFAULT_NUM_PROMPTS_PER_BATCH,
) -> dict[str, Any]:
    if (request_rate is None) == (max_concurrency is None):
        raise ValueError("Exactly one of request_rate or max_concurrency must be specified")

    start_time = time.perf_counter()
    batch_results: list[dict[str, Any]] = []
    batch_index = 0

    while True:
        if (time.perf_counter() - start_time) >= duration_sec:
            break
        result = _run_one_benchmark_batch(
            host=host,
            port=port,
            params=params,
            num_prompts=num_prompts_per_batch,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            result_dir=result_dir,
            batch_index=batch_index,
        )
        batch_results.append(result)
        batch_index += 1
        if (time.perf_counter() - start_time) >= duration_sec:
            break

    total_duration = time.perf_counter() - start_time
    merged = _merge_batch_results(batch_results, total_duration)
    _print_merged_report(merged)

    if result_filename and result_dir:
        result_path = Path(result_dir) / result_filename
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    return merged


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        test_name, model, stage_config_path, stage_overrides, extra_cli_args = request.param

        print(f"Starting OmniServer with test: {test_name}, model: {model}")

        server_args = ["--stage-init-timeout", "120"]
        # --deploy-config and --stage-overrides compose at the CLI (see vllm_omni/entrypoints/utils.py):
        # deploy-config sets the base; stage-overrides are applied on top. Both can be set.
        if stage_config_path:
            server_args = ["--deploy-config", stage_config_path] + server_args
        if stage_overrides:
            server_args = ["--stage-overrides", stage_overrides] + server_args
        if extra_cli_args:
            server_args = list(extra_cli_args) + server_args
        with OmniServer(model, server_args) as server:
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

    all_params = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)

    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")

    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")

    return {"test_name": test_name, "params": all_params[param_index]}


@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("stability_benchmark_params", benchmark_indices, indirect=True)
def test_benchmark_stability(omni_server, stability_benchmark_params):
    """Run the benchmark for a fixed duration using request-rate or max-concurrency and assert zero failed requests."""
    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]
    duration_sec = params.get("duration_sec", 300)
    num_prompts_per_batch = params.get("num_prompts_per_batch", DEFAULT_NUM_PROMPTS_PER_BATCH)
    request_rate = params.get("request_rate")
    max_concurrency = params.get("max_concurrency")

    bench_params = {
        k: v
        for k, v in params.items()
        if k not in ("duration_sec", "request_rate", "max_concurrency", "num_prompts_per_batch")
    }

    result = run_stability_benchmark(
        host=omni_server.host,
        port=omni_server.port,
        duration_sec=duration_sec,
        params=bench_params,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        result_dir=str(STABILITY_DIR),
        num_prompts_per_batch=num_prompts_per_batch,
    )

    assert result.get("failed", 0) == 0, f"[{test_name}] Failed requests detected: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, f"[{test_name}] No requests completed"
