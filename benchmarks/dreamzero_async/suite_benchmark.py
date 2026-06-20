#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_BENCHMARK = REPO_ROOT / "benchmarks" / "dreamzero_async" / "live_benchmark.py"
DEFAULT_CONFIG = REPO_ROOT / "benchmarks" / "dreamzero_async" / "suite_config.example.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _run_capture(command: list[str], *, cwd: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except OSError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": repr(exc),
        }


def collect_environment(*, python: str, cwd: Path) -> dict[str, Any]:
    runtime = _run_capture(
        [
            python,
            "-c",
            (
                "import json, torch, vllm; "
                "print(json.dumps({'torch': torch.__version__, 'vllm': vllm.__version__}))"
            ),
        ],
        cwd=cwd,
    )
    runtime_payload: dict[str, Any] = {}
    if runtime["returncode"] == 0 and runtime["stdout"]:
        runtime_payload = json.loads(runtime["stdout"].splitlines()[-1])

    gpu = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        cwd=cwd,
    )
    git_commit = _run_capture(["git", "rev-parse", "HEAD"], cwd=cwd)
    git_branch = _run_capture(["git", "branch", "--show-current"], cwd=cwd)
    return {
        "captured_at_s": round(time.time(), 6),
        "platform": platform.platform(),
        "python": python,
        "python_version": sys.version,
        "runtime": runtime_payload,
        "runtime_probe": runtime,
        "gpu": gpu["stdout"].splitlines() if gpu["returncode"] == 0 else [],
        "gpu_probe": gpu,
        "git": {
            "branch": git_branch["stdout"],
            "commit": git_commit["stdout"],
        },
    }


def wait_for_health(url: str, *, timeout_s: float, interval_s: float, proc: subprocess.Popen[bytes] | None) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server exited before health was ready with code {proc.returncode}: {last_error}")
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = repr(exc)
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for health endpoint {url}: {last_error}")


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30.0)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.wait(timeout=30.0)


def _benchmark_command(config: dict[str, Any], variant: dict[str, Any], output_dir: Path) -> list[str]:
    bench = config["benchmark"]
    server = config["server"]
    command = [
        config.get("python", sys.executable),
        str(LIVE_BENCHMARK),
        "--host",
        server.get("host", "127.0.0.1"),
        "--port",
        str(server.get("port", 8000)),
        "--sync-path",
        bench.get("sync_path", "/v1/realtime/robot/openpi"),
        "--async-path",
        bench.get("async_path", "/v1/realtime/robot/dreamzero-async"),
        "--video-dir",
        bench["video_dir"],
        "--num-chunks",
        str(bench.get("num_chunks", 15)),
        "--control-hz",
        str(bench.get("control_hz", 15.0)),
        "--warmups",
        str(bench.get("warmups", 1)),
        "--repeats",
        str(bench.get("repeats", 3)),
        "--order",
        bench.get("order", "sync-first"),
        "--bootstrap-timeout-s",
        str(bench.get("bootstrap_timeout_s", 300.0)),
        "--chunk-timeout-s",
        str(bench.get("chunk_timeout_s", 120.0)),
        "--output-dir",
        str(output_dir),
    ]
    if bench.get("repeat_last_observation", False):
        command.append("--repeat-last-observation")
    if bench.get("realtime", False):
        command.append("--realtime")
    if bench.get("require_valid_speedup", False):
        command.append("--require-valid-speedup")
    if bench.get("sync_python"):
        command.extend(["--sync-python", bench["sync_python"]])
    if bench.get("async_python"):
        command.extend(["--async-python", bench["async_python"]])
    if variant.get("python"):
        command.extend(["--python", variant["python"]])
    return command


def run_variant(config: dict[str, Any], variant: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    server = config["server"]
    cwd = Path(variant.get("cwd", config.get("cwd", REPO_ROOT))).resolve()
    variant_dir = output_dir / variant["name"]
    logs_dir = variant_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = logs_dir / "server.log"

    env = os.environ.copy()
    env.update(config.get("env", {}))
    env.update(variant.get("env", {}))
    command = variant["command"]
    started_s = time.monotonic()
    with server_log_path.open("wb") as server_log:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            health_url = server.get("health_url") or (
                f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 8000)}"
                f"{server.get('health_path', '/health')}"
            )
            wait_for_health(
                health_url,
                timeout_s=float(server.get("startup_timeout_s", 900.0)),
                interval_s=float(server.get("health_interval_s", 5.0)),
                proc=proc,
            )
            ready_s = time.monotonic()
            bench_command = _benchmark_command(config, variant, variant_dir / "benchmark")
            bench_started_s = time.monotonic()
            bench = subprocess.run(bench_command, cwd=str(cwd), capture_output=True, text=True, check=False)
            bench_finished_s = time.monotonic()
        finally:
            _stop_process(proc)
    stopped_s = time.monotonic()

    (logs_dir / "benchmark_stdout.log").write_text(bench.stdout, encoding="utf-8")
    (logs_dir / "benchmark_stderr.log").write_text(bench.stderr, encoding="utf-8")
    if bench.returncode != 0:
        raise RuntimeError(f"Benchmark failed for {variant['name']} with exit code {bench.returncode}")

    summary = _load_json(variant_dir / "benchmark" / "summary.json")
    result = {
        "name": variant["name"],
        "description": variant.get("description", ""),
        "command": command,
        "env": variant.get("env", {}),
        "benchmark_command": bench_command,
        "timing": {
            "server_startup_s": _round(ready_s - started_s),
            "benchmark_wall_s": _round(bench_finished_s - bench_started_s),
            "total_wall_s": _round(stopped_s - started_s),
        },
        "summary": summary["summary"],
        "config": summary["config"],
        "artifact_dir": str(variant_dir),
    }
    _write_json(variant_dir / "variant_summary.json", result)
    return result


def summarize_benchmark_suite(
    variants: list[dict[str, Any]],
    *,
    baseline_name: str | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = next((item for item in variants if item["name"] == baseline_name), variants[0] if variants else None)
    baseline_async = None if baseline is None else float(baseline["summary"]["async_time_mean_s"])
    baseline_sync = None if baseline is None else float(baseline["summary"]["sync_time_mean_s"])
    rows = []
    for item in variants:
        summary = item["summary"]
        sync_mean = float(summary["sync_time_mean_s"])
        async_mean = float(summary["async_time_mean_s"])
        rows.append(
            {
                "name": item["name"],
                "description": item.get("description", ""),
                "sync_mean_s": _round(sync_mean),
                "async_mean_s": _round(async_mean),
                "time_saved_mean_s": summary["time_saved_mean_s"],
                "async_vs_sync_speedup": summary["speedup_mean"],
                "async_vs_baseline_async": _round(baseline_async / async_mean) if baseline_async and async_mean else None,
                "sync_vs_baseline_sync": _round(baseline_sync / sync_mean) if baseline_sync and sync_mean else None,
                "underruns": summary["async_underrun_total"],
                "server_errors": summary["async_server_error_total"],
                "measured_repeats": summary["measured_repeats"],
                "valid_speedup_repeats": summary.get("valid_speedup_repeats", 0),
                "invalid_speedup_repeats": summary.get("invalid_speedup_repeats", summary["measured_repeats"]),
                "speedup_claim_valid": summary.get("speedup_claim_valid", False),
            }
        )
    return {
        "environment": environment or {},
        "baseline": baseline["name"] if baseline else None,
        "variants": variants,
        "comparison": rows,
    }


def write_comparison_table(path: Path, suite: dict[str, Any]) -> None:
    lines = [
        "| Variant | Sync mean (s) | Async mean (s) | Raw async vs sync | Claim valid | Async vs baseline async | Underruns | Errors |",
        "| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for row in suite["comparison"]:
        async_vs_baseline = row["async_vs_baseline_async"]
        async_vs_baseline_text = "n/a" if async_vs_baseline is None else f"{async_vs_baseline:.3f}x"
        claim_text = "yes" if row["speedup_claim_valid"] else "no"
        lines.append(
            f"| {row['name']} | {row['sync_mean_s']:.3f} | {row['async_mean_s']:.3f} | "
            f"{row['async_vs_sync_speedup']:.3f}x | {claim_text} | {async_vs_baseline_text} | "
            f"{row['underruns']} | {row['server_errors']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_config(config: dict[str, Any]) -> None:
    if "server" not in config:
        raise ValueError("suite config must contain `server`")
    if "benchmark" not in config:
        raise ValueError("suite config must contain `benchmark`")
    if not config.get("variants"):
        raise ValueError("suite config must contain at least one variant")
    for variant in config["variants"]:
        if not variant.get("name"):
            raise ValueError("each variant must have a name")
        if not isinstance(variant.get("command"), list) or not variant["command"]:
            raise ValueError(f"variant {variant.get('name')!r} must have a non-empty command list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DreamZero async benchmark suite across server variants.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "dreamzero_async" / "suite")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write planned commands only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_json(args.config)
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(python=config.get("python", sys.executable), cwd=Path(config.get("cwd", REPO_ROOT)))
    planned = {
        "config": config,
        "environment": environment,
        "benchmark_script": str(LIVE_BENCHMARK),
    }
    _write_json(args.output_dir / "plan.json", planned)
    if args.dry_run:
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 0

    variant_results = [run_variant(config, variant, output_dir=args.output_dir) for variant in config["variants"]]
    suite = summarize_benchmark_suite(
        variant_results,
        baseline_name=args.baseline or config.get("baseline"),
        environment=environment,
    )
    _write_json(args.output_dir / "summary.json", suite)
    write_comparison_table(args.output_dir / "result_table.md", suite)
    print(json.dumps(suite["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
