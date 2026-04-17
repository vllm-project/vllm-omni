"""
Stability-specific conftest: when pytest is executed under this directory,
resource monitoring is started before each test and finalized after each test,
so each stability test case gets its own HTML report (one report per case).
No need to wrap pytest with `bash resource_monitor.sh run -- pytest ...`.

Also exposes duration-based stability benchmark helpers (merge/summary/loop)
used by ``test_stability_*.py`` under ``scripts/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import OmniServer
from tests.dfx.conftest import get_benchmark_params_for_server, run_benchmark

STABILITY_DIR = Path(__file__).resolve().parent
RESOURCE_MONITOR_SCRIPT = STABILITY_DIR / "scripts" / "resource_monitor.sh"
REPO_ROOT = STABILITY_DIR.parent.parent.parent
DEFAULT_STABILITY_SERVER_TIMEOUT_ARGS = ["--stage-init-timeout", "600", "--init-timeout", "900"]

RunOneBatchFn = Callable[
    [str, int, str, dict[str, Any], int, float | None, int | None, str, int],
    dict[str, Any],
]

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


def _build_diffusion_cmd(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    output_path: Path,
    diffusion_benchmark_script: Path,
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
        str(diffusion_benchmark_script),
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
        cmd.extend(["--max-concurrency", str(max_concurrency), "--request-rate", "inf"])
    return cmd


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
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
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
    diffusion_benchmark_script = Path(REPO_ROOT / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="stability_diffusion_", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = _build_diffusion_cmd(
            host,
            port,
            model,
            params,
            num_prompts,
            request_rate,
            max_concurrency,
            out_path,
            diffusion_benchmark_script,
        )
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
def omni_server(request: pytest.FixtureRequest):
    """Start OmniServer for stability tests, with per-module timeout override."""
    timeout_args = getattr(request.module, "STABILITY_SERVER_TIMEOUT_ARGS", DEFAULT_STABILITY_SERVER_TIMEOUT_ARGS)
    with _omni_server_lock:
        param = request.param
        if len(param) == 4:
            test_name, model, stage_config_path, serve_args = param
        else:
            test_name, model, stage_config_path = param
            serve_args = []
        print(f"Starting OmniServer with test: {test_name}, model: {model}")
        server_args = list(serve_args) + list(timeout_args)
        if stage_config_path:
            server_args = ["--stage-configs-path", stage_config_path] + server_args
        with OmniServer(model, server_args) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")
        print("OmniServer stopped")


@pytest.fixture
def stability_benchmark_params(request: pytest.FixtureRequest, omni_server):
    test_name, param_index = request.param
    if test_name != omni_server.test_name:
        pytest.skip(f"Skipping parameter for {test_name} - current server is {omni_server.test_name}")

    server_to_benchmark_mapping = getattr(request.module, "server_to_benchmark_mapping", None)
    if server_to_benchmark_mapping is None:
        raise ValueError("server_to_benchmark_mapping must be defined in the test module")

    all_params = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)
    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")
    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": all_params[param_index]}


def merge_batch_results(batch_results: list[dict[str, Any]], total_duration_sec: float) -> dict[str, Any]:
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


def print_merged_report(result: dict[str, Any]) -> None:
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


def run_stability_benchmark_loop(
    host: str,
    port: int,
    model: str,
    duration_sec: int | float,
    params: dict[str, Any],
    *,
    request_rate: float | None,
    max_concurrency: int | None,
    result_dir: str,
    num_prompts_per_batch: int,
    run_one_batch: RunOneBatchFn,
    result_filename: str | None = None,
) -> dict[str, Any]:
    if (request_rate is None) == (max_concurrency is None):
        raise ValueError("Exactly one of request_rate or max_concurrency must be specified")

    start_time = time.perf_counter()
    batch_results: list[dict[str, Any]] = []
    batch_index = 0

    while True:
        if (time.perf_counter() - start_time) >= duration_sec:
            break
        result = run_one_batch(
            host,
            port,
            model,
            params,
            num_prompts_per_batch,
            request_rate,
            max_concurrency,
            result_dir,
            batch_index,
        )
        batch_results.append(result)
        batch_index += 1
        if (time.perf_counter() - start_time) >= duration_sec:
            break

    total_duration = time.perf_counter() - start_time
    merged = merge_batch_results(batch_results, total_duration)
    print_merged_report(merged)

    if result_filename and result_dir:
        result_path = Path(result_dir) / result_filename
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    return merged


def _start_resource_monitor():
    """Start `resource_monitor.sh start` in the background and return `Popen` or `None`."""
    if not RESOURCE_MONITOR_SCRIPT.is_file():
        return None
    try:
        proc = subprocess.Popen(
            ["bash", str(RESOURCE_MONITOR_SCRIPT), "start", "--backend", "gpu"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=2)
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                if stderr.strip():
                    sys.stderr.write(f"[Stability] Resource monitor failed to start: {stderr.strip()}\n")
                return None
        except subprocess.TimeoutExpired:
            pass
        return proc
    except (FileNotFoundError, OSError):
        return None


def _get_monitor_data_root() -> Path:
    data_root = os.environ.get("RESOURCE_MONITOR_DATA_ROOT") or os.environ.get("GPU_MONITOR_DATA_ROOT")
    if data_root:
        return Path(data_root)
    return STABILITY_DIR / "gpu_monitor_data"


def _wait_for_run_dir(timeout_sec: int = 10) -> Path | None:
    data_root = _get_monitor_data_root()
    run_id_file = data_root / "current_run_id"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if run_id_file.is_file():
            run_id = run_id_file.read_text(encoding="utf-8").strip()
            if run_id:
                run_dir = data_root / run_id
                if run_dir.is_dir():
                    return run_dir
        time.sleep(0.5)
    return None


def _report_latest_gpu_samples(stop_event: threading.Event) -> None:
    """Periodically print the latest sampled GPU line."""
    log_interval = int(
        os.environ.get("RESOURCE_MONITOR_LOG_INTERVAL") or os.environ.get("GPU_MONITOR_LOG_INTERVAL") or "15"
    )
    log_interval = max(log_interval, 1)
    last_line = ""

    time.sleep(min(log_interval, 5))
    while not stop_event.wait(log_interval):
        run_dir = _wait_for_run_dir(timeout_sec=1)
        if run_dir is None:
            continue
        csv_file = run_dir / "gpu_metrics.csv"
        if not csv_file.is_file():
            continue
        try:
            lines = csv_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if len(lines) <= 1:
            continue
        latest = lines[-1].strip()
        if latest and latest != last_line:
            last_line = latest
            sys.stderr.write(f"[GPU] {latest}\n")


def _finalize_resource_monitor() -> str | None:
    """
    Run `resource_monitor.sh finalize` for the current run and generate the report.
    Returns the bundle dir path (for this test case's report) if successful, else None.
    """
    if not RESOURCE_MONITOR_SCRIPT.is_file():
        return None
    try:
        result = subprocess.run(
            ["bash", str(RESOURCE_MONITOR_SCRIPT), "finalize", "--backend", "gpu"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return None
        for line in (result.stdout or "").splitlines():
            if line.startswith("GPU_MONITOR_BUNDLE_DIR=") or line.startswith("RESOURCE_MONITOR_BUNDLE_DIR="):
                _, _, value = line.partition("=")
                return value.strip() if value else None
        return None
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


@pytest.fixture(autouse=True)
def stability_resource_monitor_per_test(request: pytest.FixtureRequest):
    """
    For each test under this directory: start GPU monitor before the test,
    then finalize after the test so this case gets its own report.html.
    """
    proc = _start_resource_monitor()
    stop_event = threading.Event()
    reporter: threading.Thread | None = None

    if proc is not None:
        reporter = threading.Thread(
            target=_report_latest_gpu_samples,
            args=(stop_event,),
            name="stability-resource-monitor-reporter",
            daemon=True,
        )
        reporter.start()
        run_dir = _wait_for_run_dir(timeout_sec=5)
        node_name = request.node.name
        if run_dir is not None:
            sys.stderr.write(f"[Stability] Resource monitor started for test: {node_name} | run dir: {run_dir}\n")
        else:
            sys.stderr.write(f"[Stability] Resource monitor started for test: {node_name} (run dir not ready yet)\n")

    yield

    # Teardown: stop reporter, stop monitor, finalize → one HTML per test
    if proc is not None:
        stop_event.set()
        if reporter is not None and reporter.is_alive():
            reporter.join(timeout=2)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        bundle_dir = _finalize_resource_monitor()
        node_name = request.node.name
        if bundle_dir:
            sys.stderr.write(f"[Stability] Report for test «{node_name}»: {bundle_dir}/report.html\n")
        else:
            sys.stderr.write(f"[Stability] Finalize skipped or failed for test «{node_name}»\n")
