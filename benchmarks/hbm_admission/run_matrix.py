#!/usr/bin/env python3
"""Run the single-GPU before/after HBM-admission performance matrix.

The runner is deliberately resumable: each completed case owns a benchmark
JSON, GPU telemetry CSV, client log, and a status JSON. Re-running with the
same --output-dir skips completed cases and rebuilds summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BEFORE_REPO = Path("/tmp/vllm-omni-before")
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_smoke"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
PORT = 8000


@dataclass(frozen=True)
class Group:
    name: str
    code: str
    deploy_name: str
    stage_limits_gb: tuple[float, float] | None


GROUPS = {
    "A": Group("A", "before", "no_limit", None),
    "B": Group("B", "after", "no_limit", None),
    "C1": Group("C1", "after", "medium", (8.0, 6.0)),
    "C2": Group("C2", "after", "tight", (4.0, 3.0)),
}

ROUND_ORDERS = (
    ("A", "B", "C1", "C2"),
    ("C2", "C1", "B", "A"),
    ("B", "A", "C2", "C1"),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare_deploy_configs(output_dir: Path) -> dict[str, Path]:
    source = yaml.safe_load(DEFAULT_DEPLOY.read_text())
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for group in GROUPS.values():
        path = config_dir / f"{group.deploy_name}.yaml"
        config = json.loads(json.dumps(source))
        for stage in config["stages"]:
            stage.pop("hbm_limit_gb", None)
        if group.stage_limits_gb is not None:
            by_id = {stage["stage_id"]: stage for stage in config["stages"]}
            by_id[0]["hbm_limit_gb"] = group.stage_limits_gb[0]
            by_id[1]["hbm_limit_gb"] = group.stage_limits_gb[1]
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        result[group.deploy_name] = path
    return result


def wait_for_health(proc: subprocess.Popen, timeout_s: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{PORT}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"server exited with {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy in {timeout_s}s; see {log_path}")


def stop_process_group(proc: subprocess.Popen | None, timeout_s: int = 60) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def wait_for_gpu_release(threshold_mib: int = 1024, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(output.strip().splitlines()[0]) <= threshold_mib:
            return
        time.sleep(2)
    raise TimeoutError("GPU memory did not return below the release threshold")


def start_server(
    repo: Path,
    model: str,
    deploy: Path,
    log_path: Path,
    startup_timeout_s: int,
) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    command = [
        "python",
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--deploy-config",
        str(deploy),
        "--log-stats",
    ]
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        wait_for_health(proc, startup_timeout_s, log_path)
    except Exception:
        stop_process_group(proc)
        log_file.close()
        raise
    return proc, log_file


def start_gpu_monitor(path: Path) -> tuple[subprocess.Popen, Any]:
    output = path.open("w")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-lms",
        "200",
    ]
    return (
        subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True),
        output,
    )


def parse_gpu_metrics(path: Path) -> dict[str, Any]:
    memory: list[float] = []
    utilization: list[float] = []
    power: list[float] = []
    temperature: list[float] = []
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if len(row) < 5:
                continue
            try:
                memory.append(float(row[1].strip()))
                utilization.append(float(row[2].strip()))
                power.append(float(row[3].strip()))
                temperature.append(float(row[4].strip()))
            except ValueError:
                continue
    if not memory:
        return {"samples": 0}
    return {
        "samples": len(memory),
        "memory_used_mib": {
            "min": min(memory),
            "mean": statistics.fmean(memory),
            "peak": max(memory),
        },
        "gpu_utilization_percent": {
            "mean": statistics.fmean(utilization),
            "peak": max(utilization),
        },
        "power_draw_w": {
            "mean": statistics.fmean(power),
            "peak": max(power),
        },
        "temperature_c": {"mean": statistics.fmean(temperature), "peak": max(temperature)},
    }


def run_case(
    *,
    repo: Path,
    model: str,
    output_dir: Path,
    group: Group,
    round_number: int,
    concurrency: int,
    num_prompts: int,
    num_warmups: int,
) -> dict[str, Any]:
    case_id = f"round_{round_number}_{group.name}_c{concurrency}"
    case_dir = output_dir / "cases" / case_id
    status_path = case_dir / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "completed":
            return previous

    case_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json = case_dir / "benchmark.json"
    client_log = case_dir / "client.log"
    gpu_csv = case_dir / "gpu.csv"
    status: dict[str, Any] = {
        "case_id": case_id,
        "group": asdict(group),
        "round": round_number,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "num_warmups": num_warmups,
        "started_at": utc_now(),
        "status": "running",
    }
    write_json(status_path, status)

    monitor, monitor_file = start_gpu_monitor(gpu_csv)
    command = [
        "vllm",
        "bench",
        "serve",
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--model",
        model,
        "--backend",
        "openai-audio-speech",
        "--endpoint",
        "/v1/audio/speech",
        "--dataset-name",
        "seed-tts-text",
        "--dataset-path",
        str(DEFAULT_DATASET),
        "--seed-tts-locale",
        "en",
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        str(num_warmups),
        "--extra-body",
        '{"voice":"Vivian","language":"English","task_type":"CustomVoice"}',
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--percentile-metrics",
        "ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun",
        "--save-result",
        "--result-dir",
        str(case_dir),
        "--result-filename",
        benchmark_json.name,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    try:
        with client_log.open("w") as output:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600,
            )
        status["client_exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        status["client_exit_code"] = None
        status["error"] = "benchmark timed out after 3600 seconds"
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=5)
        monitor_file.close()

    status["gpu"] = parse_gpu_metrics(gpu_csv)
    if benchmark_json.exists():
        status["benchmark"] = json.loads(benchmark_json.read_text())
    if status.get("client_exit_code") == 0 and benchmark_json.exists():
        status["status"] = "completed"
    else:
        status["status"] = "failed"
        status.setdefault("error", f"benchmark exited with {status.get('client_exit_code')}")
    status["finished_at"] = utc_now()
    write_json(status_path, status)
    return status


def aggregate(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for path in sorted((output_dir / "cases").glob("*/status.json")):
        cases.append(json.loads(path.read_text()))
    summary: dict[str, Any] = {
        "metadata": metadata,
        "cases": cases,
        "aggregates": {},
        "comparisons_percent": {},
        "server_events": {},
    }
    metric_names = (
        "request_throughput",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p99_e2el_ms",
        "mean_audio_ttfp_ms",
        "median_audio_ttfp_ms",
        "p99_audio_ttfp_ms",
        "mean_audio_rtf",
        "median_audio_rtf",
        "p99_audio_rtf",
        "mean_audio_underrun_s",
        "p99_audio_underrun_s",
    )
    for group_name in GROUPS:
        for concurrency in metadata["concurrencies"]:
            selected = [
                case
                for case in cases
                if case.get("status") == "completed"
                and case["group"]["name"] == group_name
                and case["concurrency"] == concurrency
            ]
            if not selected:
                continue
            item: dict[str, Any] = {"runs": len(selected)}
            for metric in metric_names:
                values = [case["benchmark"].get(metric) for case in selected]
                values = [float(value) for value in values if isinstance(value, (int, float))]
                if values:
                    item[metric] = {
                        "mean": statistics.fmean(values),
                        "min": min(values),
                        "max": max(values),
                    }
            peaks = [
                case.get("gpu", {}).get("memory_used_mib", {}).get("peak")
                for case in selected
            ]
            peaks = [float(value) for value in peaks if isinstance(value, (int, float))]
            if peaks:
                item["peak_memory_used_mib"] = {
                    "mean": statistics.fmean(peaks),
                    "min": min(peaks),
                    "max": max(peaks),
                }
            completed = [case["benchmark"].get("completed", 0) for case in selected]
            failed = [case["benchmark"].get("failed", 0) for case in selected]
            item["completed_requests"] = sum(completed)
            item["failed_requests"] = sum(failed)
            summary["aggregates"][f"{group_name}_c{concurrency}"] = item

    event_patterns = {
        "oom_mentions": re.compile(r"out of memory|cuda oom", re.IGNORECASE),
        "preemption_mentions": re.compile(r"preempt", re.IGNORECASE),
        "fatal_mentions": re.compile(r"fatal|traceback", re.IGNORECASE),
    }
    for log_path in sorted((output_dir / "server_logs").glob("*.log")):
        contents = log_path.read_text(errors="replace")
        summary["server_events"][log_path.name] = {
            name: len(pattern.findall(contents))
            for name, pattern in event_patterns.items()
        }

    comparisons = (("B", "A"), ("C1", "B"), ("C2", "B"))
    comparison_metrics = (
        "request_throughput",
        "p99_e2el_ms",
        "p99_audio_ttfp_ms",
        "median_audio_rtf",
        "peak_memory_used_mib",
    )
    for new_group, baseline_group in comparisons:
        for concurrency in metadata["concurrencies"]:
            new = summary["aggregates"].get(f"{new_group}_c{concurrency}")
            baseline = summary["aggregates"].get(f"{baseline_group}_c{concurrency}")
            if new is None or baseline is None:
                continue
            deltas: dict[str, float] = {}
            for metric in comparison_metrics:
                new_value = new.get(metric, {}).get("mean")
                baseline_value = baseline.get(metric, {}).get("mean")
                if new_value is not None and baseline_value:
                    deltas[metric] = (new_value / baseline_value - 1.0) * 100.0
            summary["comparisons_percent"][
                f"{new_group}_vs_{baseline_group}_c{concurrency}"
            ] = deltas
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--before-repo", type=Path, default=DEFAULT_BEFORE_REPO)
    parser.add_argument("--after-repo", type=Path, default=REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        metadata = json.loads((args.output_dir / "metadata.json").read_text())
        aggregate(args.output_dir, metadata)
        return 0
    deploys = prepare_deploy_configs(args.output_dir)
    if args.smoke:
        rounds = 1
        concurrencies = [1]
        num_prompts = 2
        num_warmups = 1
        orders = (("B", "C2"),)
    else:
        rounds = args.rounds
        concurrencies = args.concurrencies
        num_prompts = args.num_prompts
        num_warmups = args.num_warmups
        orders = ROUND_ORDERS
        if rounds > len(orders):
            raise ValueError(f"at most {len(orders)} counterbalanced rounds are defined")

    metadata = {
        "created_at": utc_now(),
        "model": args.model,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "before_repo": str(args.before_repo),
        "after_repo": str(args.after_repo),
        "groups": {name: asdict(group) for name, group in GROUPS.items()},
        "rounds": rounds,
        "concurrencies": concurrencies,
        "num_prompts_per_case": num_prompts,
        "num_warmups_per_case": num_warmups,
    }
    write_json(args.output_dir / "metadata.json", metadata)

    for round_index in range(rounds):
        for group_name in orders[round_index]:
            group = GROUPS[group_name]
            repo = args.before_repo if group.code == "before" else args.after_repo
            deploy = deploys[group.deploy_name]
            server_log = args.output_dir / "server_logs" / f"round_{round_index + 1}_{group_name}.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            wait_for_gpu_release()
            server = None
            server_file = None
            try:
                server, server_file = start_server(
                    repo,
                    args.model,
                    deploy,
                    server_log,
                    args.startup_timeout,
                )
                for concurrency in concurrencies:
                    result = run_case(
                        repo=repo,
                        model=args.model,
                        output_dir=args.output_dir,
                        group=group,
                        round_number=round_index + 1,
                        concurrency=concurrency,
                        num_prompts=num_prompts,
                        num_warmups=num_warmups,
                    )
                    aggregate(args.output_dir, metadata)
                    if result["status"] != "completed":
                        raise RuntimeError(f"case {result['case_id']} failed: {result.get('error')}")
            finally:
                stop_process_group(server)
                if server_file is not None:
                    server_file.close()
                wait_for_gpu_release()

    aggregate(args.output_dir, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
