"""Stability helpers for resource monitoring and benchmark execution."""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.dfx.conftest import run_benchmark

STABILITY_DIR = Path(__file__).resolve().parent
RESOURCE_MONITOR_SCRIPT = STABILITY_DIR / "scripts" / "resource_monitor.sh"
REPO_ROOT = STABILITY_DIR.parent.parent.parent
_BUCKET_KEY_PATTERN = re.compile(r"^\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)$")

RunOneBatchFn = Callable[
    [str, int, str, dict[str, Any], int, float | None, int | None, str, int],
    dict[str, Any],
]


def start_resource_monitor():
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


def get_monitor_data_root() -> Path:
    data_root = os.environ.get("RESOURCE_MONITOR_DATA_ROOT") or os.environ.get("GPU_MONITOR_DATA_ROOT")
    if data_root:
        return Path(data_root)
    return STABILITY_DIR / "gpu_monitor_data"


def wait_for_run_dir(timeout_sec: int = 10) -> Path | None:
    data_root = get_monitor_data_root()
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


def report_latest_gpu_samples(stop_event: threading.Event) -> None:
    """Periodically print the latest sampled GPU line."""
    log_interval = int(
        os.environ.get("RESOURCE_MONITOR_LOG_INTERVAL") or os.environ.get("GPU_MONITOR_LOG_INTERVAL") or "15"
    )
    log_interval = max(log_interval, 1)
    last_line = ""

    time.sleep(min(log_interval, 5))
    while not stop_event.wait(log_interval):
        run_dir = wait_for_run_dir(timeout_sec=1)
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


def finalize_resource_monitor() -> str | None:
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


def _sample_int_from_range_spec(value: Any, rng: random.Random) -> Any:
    """Resolve one value that may be scalar or range spec into an int."""
    if isinstance(value, int):
        return value

    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        low, high = int(value[0]), int(value[1])
        if low > high:
            low, high = high, low
        return rng.randint(low, high)

    if isinstance(value, dict) and {"min", "max"} <= set(value):
        low, high = int(value["min"]), int(value["max"])
        if low > high:
            low, high = high, low
        return rng.randint(low, high)

    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return int(raw)
        if "-" in raw:
            parts = [p.strip() for p in raw.split("-", 1)]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                low, high = int(parts[0]), int(parts[1])
                if low > high:
                    low, high = high, low
                return rng.randint(low, high)

    return value


def _sample_bucket_key(raw_key: str, rng: random.Random) -> str:
    """Sample bucket tuple keys that use range syntax, e.g. ``(128-512, 128-512, 1)``."""
    match = _BUCKET_KEY_PATTERN.match(raw_key.strip())
    if not match:
        return raw_key

    sampled_parts: list[int] = []
    for token in match.groups():
        sampled = _sample_int_from_range_spec(token.strip(), rng)
        if not isinstance(sampled, int):
            return raw_key
        sampled_parts.append(sampled)

    # For video buckets (height>0 and num_frames>1), enforce even H/W to avoid
    # ffmpeg yuv420p encoding/decoding failures ("Could not open video stream").
    if sampled_parts[0] > 0 and sampled_parts[2] > 1:
        sampled_parts[0] = max(2, sampled_parts[0] - (sampled_parts[0] % 2))
        sampled_parts[1] = max(2, sampled_parts[1] - (sampled_parts[1] % 2))

    return f"({sampled_parts[0]}, {sampled_parts[1]}, {sampled_parts[2]})"


def _sample_stability_batch_params(params: dict[str, Any], batch_index: int) -> dict[str, Any]:
    """Materialize per-batch random values for configured range fields."""
    sampled = dict(params)
    rng = random.Random(time.time_ns() + batch_index)

    for field_name in (
        "random_input_len",
        "random_output_len",
        "random_mm_base_items_per_request",
        "width",
        "height",
    ):
        if field_name in sampled:
            sampled[field_name] = _sample_int_from_range_spec(sampled[field_name], rng)

    bucket_config = sampled.get("random_mm_bucket_config")
    if isinstance(bucket_config, dict):
        sampled_bucket_config: dict[str, float] = {}
        for raw_key, probability in bucket_config.items():
            sampled_key = _sample_bucket_key(str(raw_key), rng)
            sampled_bucket_config[sampled_key] = sampled_bucket_config.get(sampled_key, 0.0) + float(probability)
        sampled["random_mm_bucket_config"] = sampled_bucket_config

    return sampled


def _run_one_vllm_bench_batch(
    host: str,
    port: int,
    _model: str,
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

    # Print the exact per-batch benchmark CLI (randomized params are already materialized).
    preview_cmd = ["vllm", "bench", "serve", "--omni", *args]
    print(f"\n[Stability][Batch {batch_index}] Benchmark command:")
    print(shlex.join(preview_cmd))

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
    except (FileNotFoundError, OSError) as exc:
        return {
            "completed": 0,
            "failed": 1,
            "duration": 0.0,
            "errors": [f"Benchmark batch failed: {type(exc).__name__}: {exc}"],
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
        with open(out_path, encoding="utf-8") as file:
            metrics = json.load(file)
        return _normalize_bench_metrics(metrics)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        return {
            "completed": 0,
            "failed": 1,
            "duration": 0.0,
            "errors": [f"Diffusion batch failed: {type(exc).__name__}: {exc}"],
        }
    finally:
        out_path.unlink(missing_ok=True)


def merge_batch_results(batch_results: list[dict[str, Any]], total_duration_sec: float) -> dict[str, Any]:
    if not batch_results:
        return {"completed": 0, "failed": 0, "duration": total_duration_sec, "errors": []}

    completed = sum(result.get("completed", 0) for result in batch_results)
    failed = sum(result.get("failed", 0) for result in batch_results)
    merged: dict[str, Any] = {
        "completed": completed,
        "failed": failed,
        "duration": total_duration_sec,
        "errors": [],
    }
    for result in batch_results:
        merged["errors"].extend(result.get("errors") or [])
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
        sampled_params = _sample_stability_batch_params(params, batch_index)
        result = run_one_batch(
            host,
            port,
            model,
            sampled_params,
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
        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(merged, file, indent=2, ensure_ascii=False)

    return merged
