"""
Qwen3-Omni stability: OmniServer + ``vllm bench serve --omni`` for a fixed duration.

Configuration: ``tests/dfx/stability/tests/test_qwen3_omni.json``.
"""

from __future__ import annotations

import json
import os
import threading
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
from tests.dfx.stability.conftest import run_stability_benchmark_loop

STABILITY_DIR = Path(__file__).resolve().parent.parent
STAGE_CONFIGS_DIR = STABILITY_DIR / "stage_configs"
CONFIG_FILE_PATH = str(STABILITY_DIR / "tests" / "test_qwen3_omni.json")
DEFAULT_NUM_PROMPTS_PER_BATCH = 20

BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)
test_params = create_unique_server_params(BENCHMARK_CONFIGS, STAGE_CONFIGS_DIR)
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS)
benchmark_indices = create_benchmark_indices(BENCHMARK_CONFIGS, server_to_benchmark_mapping)

_omni_server_lock = threading.Lock()


def _normalize_bench_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    completed = int(raw.get("completed", raw.get("completed_requests", 0) or 0))
    failed = int(raw.get("failed", raw.get("failed_requests", 0) or 0))
    duration = float(raw.get("duration", 0.0) or 0.0)
    errors = list(raw.get("errors") or [])
    if failed and not errors:
        errors = [f"{failed} benchmark request(s) failed"]
    return {"completed": completed, "failed": failed, "duration": duration, "errors": errors}


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


def _run_one_vllm_bench_batch(
    host: str,
    port: int,
    _model: str,
    params: dict[str, Any],
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    result_dir: str,
    _batch_index: int,
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
        )
        return _normalize_bench_metrics(result)
    except (FileNotFoundError, OSError) as e:
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


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server with Qwen3-Omni stage config."""
    with _omni_server_lock:
        test_name, model, serve_extras = request.param
        print(f"Starting OmniServer with test: {test_name}, model: {model}")
        server_args = list(serve_extras) + ["--stage-init-timeout", "120"]
        with OmniServer(model, server_args) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")
        print("OmniServer stopped")


@pytest.fixture(params=benchmark_indices)
def stability_benchmark_params(request, omni_server):
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
def test_stability_qwen3_omni(omni_server, stability_benchmark_params):
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

    result = run_stability_benchmark_loop(
        host=omni_server.host,
        port=omni_server.port,
        model=omni_server.model,
        duration_sec=duration_sec,
        params=bench_params,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        result_dir=str(STABILITY_DIR),
        num_prompts_per_batch=num_prompts_per_batch,
        run_one_batch=_run_one_vllm_bench_batch,
    )

    assert result.get("failed", 0) == 0, f"[{test_name}] Failed requests detected: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, f"[{test_name}] No requests completed"
