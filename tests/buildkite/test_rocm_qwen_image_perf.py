# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for measurement completeness, not AMD performance qualification."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SCRIPT = Path(".buildkite/amd/scripts/validate-qwen-image-perf.py")
CONFIG = Path("tests/dfx/perf/tests/test_qwen_image_vllm_omni.json")
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT))
validate_records = SCRIPT_GLOBALS["validate_records"]
resolve_checkout_head = SCRIPT_GLOBALS["resolve_checkout_head"]


@pytest.fixture
def measurements() -> tuple[list[dict], list[dict]]:
    configs = json.loads(CONFIG.read_text())
    settings = {config["test_name"]: config["server_params"] for config in configs}
    # Explicit workload contract: two baseline runs, two sequential step runs,
    # then four step-execution concurrency runs. Do not derive this from the
    # validator's expansion logic, which is itself under test.
    baseline = "test_qwen_image_single_device"
    step = "test_qwen_image_single_device_step_execution"
    runs = [
        (baseline, 512, 20, 10, 1, False),
        (baseline, 1536, 35, 10, 1, False),
        (step, 512, 20, 10, 1, False),
        (step, 1536, 35, 10, 1, False),
        (step, 512, 20, 20, 1, True),
        (step, 512, 20, 40, 2, True),
        (step, 512, 20, 80, 4, True),
        (step, 512, 20, 160, 8, True),
    ]
    records = []
    for name, size, steps, requests, concurrency, warmup in runs:
        params = {
            "name": f"{size}x{size}_steps{steps}" + ("_high_concurrency" if warmup else ""),
            "dataset": "random",
            "task": "t2i",
            "width": size,
            "height": size,
            "num-inference-steps": steps,
            "num-prompts": requests,
            "max-concurrency": concurrency,
            "request-rate": "inf",
            "enable-negative-prompt": True,
        }
        if warmup:
            params.update({"warmup-requests": 8, "warmup-concurrency": 8, "warmup-num-inference-steps": 20})
        records.append(
            {
                "test_name": name,
                "server_params": settings[name],
                "benchmark_params": params,
                "Hardware": "MI300",
                "completed": requests,
                "failed": 0,
                "throughput_qps": 0.1,
                "latency_mean": 10.0,
                "peak_memory_mb_max": 60000.0,
                "peak_memory_mb_mean": 59000.0,
            }
        )
    return records, configs


def test_complete_measurements_are_published(tmp_path: Path, measurements, monkeypatch):
    records, _ = measurements
    (tmp_path / "diffusion_result_fixture.json").write_text(json.dumps(records))
    summary = tmp_path / "summary.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=CI Fixture",
            "-c",
            "user.email=ci@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setenv("BUILDKITE_COMMIT", "a" * 40)
    monkeypatch.setenv("BUILDKITE_BUILD_CHECKOUT_PATH", str(checkout))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), str(CONFIG), str(summary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(summary.read_text())
    assert report["mode"] == "measurement-only"
    assert report["amd_baseline"] is None
    assert report["performance_regression_gate"] is False
    assert report["buildkite_commit"] == "a" * 40
    assert len(report["checkout_head"]) == 40
    assert [run["completed"] for run in report["measurements"]] == [10, 10, 10, 10, 20, 40, 80, 160]


def test_checkout_head_is_optional_without_git(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BUILDKITE_BUILD_CHECKOUT_PATH", str(tmp_path))
    assert resolve_checkout_head() is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed", 9),
        ("completed", True),
        ("failed", 1),
        ("Hardware", "H100"),
        ("throughput_qps", None),
        ("latency_mean", float("nan")),
        ("latency_mean", float("inf")),
        ("peak_memory_mb_mean", 0),
        ("peak_memory_mb_max", None),
        ("server_params", {}),
    ],
)
def test_invalid_measurement_fails(measurements, field, value):
    records, configs = measurements
    records[0][field] = value
    with pytest.raises(ValueError):
        validate_records(records, configs)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "resolution", "concurrency", "warmup", "request-rate"])
def test_workload_drift_fails(measurements, mutation):
    records, configs = measurements
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records[-1] = records[0]
    else:
        field = {
            "resolution": "width",
            "concurrency": "max-concurrency",
            "warmup": "warmup-requests",
            "request-rate": "request-rate",
        }[mutation]
        records[-1]["benchmark_params"][field] = 1
    with pytest.raises(ValueError):
        validate_records(records, configs)


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_stale_aggregate_fails(tmp_path: Path, count: int):
    for index in range(count):
        (tmp_path / f"diffusion_result_{index}.json").write_text("[]")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), str(CONFIG), str(tmp_path / "summary.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "summary.json").exists()
