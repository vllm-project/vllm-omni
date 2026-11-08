# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate complete AMD measurements; do not apply CUDA performance baselines."""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

CASES = {"test_qwen_image_single_device", "test_qwen_image_single_device_step_execution"}
METRICS = ("throughput_qps", "latency_mean", "peak_memory_mb_max", "peak_memory_mb_mean")


def resolve_checkout_head() -> str | None:
    checkout_path = os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH")
    if not checkout_path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", checkout_path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Some CPU suites execute a source-only image without repository
        # metadata. BUILDKITE_COMMIT remains the authoritative build revision.
        return None


def workload_key(name: str, params: dict) -> str:
    # The runner adds an infinite request rate to concurrency sweeps. Baselines
    # are reference metadata, not workload inputs or AMD qualification evidence.
    if params.get("request-rate", "inf") != "inf":
        raise ValueError("Concurrency sweeps must use the unchanged infinite request rate")
    workload = {key: value for key, value in params.items() if key not in {"baseline", "request-rate"}}
    return json.dumps([name, workload], sort_keys=True)


def expected_workloads(configs: list[dict]) -> dict[str, dict]:
    expected = {}
    for config in configs:
        if config["test_name"] not in CASES:
            continue
        for params in config["benchmark_params"]:
            prompts = params["num-prompts"]
            concurrency = params["max-concurrency"]
            prompts = prompts if isinstance(prompts, list) else [prompts]
            concurrency = concurrency if isinstance(concurrency, list) else [concurrency]
            if len(prompts) != len(concurrency):
                raise ValueError("Qwen-Image request counts and concurrency must pair exactly")
            for count, limit in zip(prompts, concurrency):
                run = {**params, "num-prompts": count, "max-concurrency": limit}
                key = workload_key(config["test_name"], run)
                if key in expected:
                    raise ValueError(f"Duplicate configured workload: {key}")
                expected[key] = config["server_params"]
    if len(expected) != 8:
        raise ValueError(f"Expected eight configured measurements, found {len(expected)}")
    return expected


def validate_records(records: list[dict], configs: list[dict]) -> list[dict]:
    expected = expected_workloads(configs)
    seen = set()
    measurements = []
    for record in records:
        params = record["benchmark_params"]
        key = workload_key(record["test_name"], params)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate workload: {key}")
        seen.add(key)
        if record["server_params"] != expected[key]:
            raise ValueError("Server settings differ from the configured workload")
        if record["Hardware"] not in {"MI300", "MI325"}:
            raise ValueError(f"Expected AMD measurements, got {record['Hardware']!r}")
        if type(record["completed"]) is not int or record["completed"] != params["num-prompts"]:
            raise ValueError("Incomplete benchmark requests")
        if type(record["failed"]) is not int or record["failed"] != 0:
            raise ValueError("Failed benchmark requests")
        for metric in METRICS:
            value = record.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing or invalid {metric}: {value!r}")
        measurements.append(
            {
                "test_name": record["test_name"],
                "hardware": record["Hardware"],
                "benchmark_params": {k: v for k, v in params.items() if k != "baseline"},
                "completed": record["completed"],
                "failed": record["failed"],
                **{metric: record[metric] for metric in METRICS},
            }
        )
    if seen != set(expected):
        raise ValueError(f"Expected eight measurements, found {len(seen)}")
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("summary_file", type=Path)
    args = parser.parse_args()
    files = sorted(args.result_dir.glob("diffusion_result_*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one fresh result aggregate, found {len(files)}")
    records = json.loads(files[0].read_text(encoding="utf-8"))
    configs = json.loads(args.config_file.read_text(encoding="utf-8"))
    measurements = validate_records(records, configs)
    summary = {
        "mode": "measurement-only",
        "amd_baseline": None,
        "performance_regression_gate": False,
        "buildkite_commit": os.environ["BUILDKITE_COMMIT"],
        "checkout_head": resolve_checkout_head(),
        "units": {"throughput_qps": "requests/s", "latency_mean": "s", "peak_memory_mb": "MB"},
        "source_result": files[0].name,
        "measurements": measurements,
    }
    args.summary_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validated {len(measurements)} AMD measurements; performance thresholds remain unqualified.")


if __name__ == "__main__":
    main()
