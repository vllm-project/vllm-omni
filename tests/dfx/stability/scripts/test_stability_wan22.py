"""
Wan2.2 T2V stability: OmniServer (diffusion) + ``diffusion_benchmark_serving.py`` / ``v1/videos``.

Configuration: ``tests/dfx/stability/tests/test_wan22.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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
from tests.dfx.stability.conftest import run_stability_benchmark_loop

STABILITY_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STABILITY_DIR.parent.parent.parent
STAGE_CONFIGS_DIR = STABILITY_DIR / "stage_configs"
CONFIG_FILE_PATH = str(STABILITY_DIR / "tests" / "test_wan22.json")
DEFAULT_NUM_PROMPTS_PER_BATCH = 20
DIFFUSION_BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"

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


def _build_diffusion_cmd(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    output_path: Path,
) -> list[str]:
    skip_keys = {
        "request_rate",
        "max_concurrency",
        "num_prompts",
        "baseline",
        "duration_sec",
        "num_prompts_per_batch",
    }
    cmd: list[str] = [
        sys.executable,
        "-u",
        str(DIFFUSION_BENCHMARK_SCRIPT),
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--output-file",
        str(output_path),
    ]
    for key, value in params.items():
        if key in skip_keys or value is None:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool) and value:
            cmd.append(flag)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(value)])

    cmd.extend(["--num-prompts", str(num_prompts)])
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])
    else:
        cmd.extend(
            [
                "--max-concurrency",
                str(max_concurrency),
                "--request-rate",
                "inf",
            ]
        )
    return cmd


def _run_one_diffusion_batch(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    _result_dir: str,
    _batch_index: int,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="stability_diffusion_", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = _build_diffusion_cmd(host, port, model, params, num_prompts, request_rate, max_concurrency, out_path)
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        if proc.returncode != 0:
            return {
                "completed": 0,
                "failed": 1,
                "duration": 0.0,
                "errors": [f"diffusion_benchmark_serving.py exited {proc.returncode}"],
            }
        if not out_path.is_file():
            return {
                "completed": 0,
                "failed": 1,
                "duration": 0.0,
                "errors": [f"Missing benchmark output: {out_path}"],
            }
        with open(out_path, encoding="utf-8") as f:
            metrics = json.load(f)
        return _normalize_bench_metrics(metrics)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        return {
            "completed": 0,
            "failed": 1,
            "duration": 0.0,
            "errors": [f"Diffusion batch failed: {type(e).__name__}: {e}"],
        }
    finally:
        out_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni diffusion server (no multi-stage YAML)."""
    with _omni_server_lock:
        test_name, model, serve_extras = request.param
        print(f"Starting OmniServer with test: {test_name}, model: {model}")
        server_args = list(serve_extras) + ["--stage-init-timeout", "600", "--init-timeout", "900"]
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
def test_stability_wan22(omni_server, stability_benchmark_params):
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
        run_one_batch=_run_one_diffusion_batch,
    )

    assert result.get("failed", 0) == 0, f"[{test_name}] Failed requests detected: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, f"[{test_name}] No requests completed"
