# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify resource accounting, estimate scope, and utilisation through the CLI."""

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).resolve().parents[2] / ".claude/skills/omni-perf-ceiling/scripts/ceiling_report.py"
END = dt.datetime(2026, 11, 11, 12, 0, 10)


@pytest.fixture
def ceiling():
    spec = importlib.util.spec_from_file_location("ceiling_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate(ceiling, *, compute=(), decode=(), fixed=(), concurrency=2, efficiency=None):
    """Use unit resource rates so the expected schedules are explicit."""
    peaks = {"copy_bandwidth_gbs": 2**30 / 1e9, "matmul_tflops": {"fp32": 1, "bf16": 1}}
    cell = {
        "concurrency": concurrency,
        "mean_audio_duration_s": 5,
        "request_audio_s": [1, 9],
        "aggregate_rtf": 10,
        "mean_audio_rtf": 10,
    }
    specs = {"compute": compute, "decode": decode, "fixed": fixed, "fill": [], "launch": []}
    return ceiling.physical_bounds(cell, peaks, specs, 1, efficiency)


def test_shared_resources_can_overlap(ceiling):
    cell = estimate(ceiling, compute=["compute@0=1,fp32", "memory@1=0,fp32,1"])
    # Two pipeline stages can overlap 1 s of compute with 1 s of memory.
    assert cell["rtf_bound"] == pytest.approx(2)
    assert cell["g_bound_total"] == pytest.approx(1)


def test_shared_same_resource_adds(ceiling):
    cell = estimate(ceiling, compute=["a@0=1,fp32", "b@1=1,fp32"])
    assert cell["rtf_bound"] == pytest.approx(4)


def test_serial_stage_rows_still_add(ceiling):
    cell = estimate(ceiling, compute=["stage@0=1,fp32", "stage@0=0,fp32,1"])
    assert cell["rtf_bound"] == pytest.approx(4)


def test_distinct_compute_modes_are_not_assumed_serial(ceiling):
    cell = estimate(ceiling, compute=["a@0=1,fp32", "b@1=1,bf16"])
    assert cell["rtf_bound"] == pytest.approx(2)


def test_fixed_work_joins_matching_resource(ceiling):
    cell = estimate(ceiling, compute=["compute@0=1,fp32"], fixed=["prefill@0=0,fp32,10"])
    # Fixed memory = 10 / batch / mean_audio = 1 s/audio-s; overlaps compute.
    shared = cell["bound_floors"]["shared device"]
    assert shared["fixed_s"] / 5 + shared["rate"] == pytest.approx(2)


def test_fixed_compute_adds_to_streaming_compute(ceiling):
    cell = estimate(ceiling, compute=["compute@0=1,fp32"], fixed=["prefill@0=5,fp32"])
    shared = cell["bound_floors"]["shared device"]
    assert shared["fixed_s"] / 5 + shared["rate"] == pytest.approx(4)


def test_efficiency_projection_preserves_resource_overlap(ceiling):
    cell = estimate(
        ceiling,
        compute=["compute@0=1,fp32", "memory@1=0,fp32,1"],
        efficiency={"compute": 0.5, "memory": 0.5},
    )
    assert cell["efficiency_projection"]["rtf"] == pytest.approx(4)


@pytest.mark.parametrize("concurrency", [1, 64])
def test_workload_average_cannot_bound_mean_rtf(ceiling, concurrency):
    # d=[1,9], tokens=[1,81], 1 ms/token: mean RTF=.005, aggregate=.0082.
    cell = estimate(ceiling, decode=["thinker@0=0,8.2,0,1,fp32"], concurrency=concurrency)
    assert "rtf_bound_mean" not in cell
    assert cell["mean_bound_unavailable"]
    assert all(not floor["per_request"] for floor in cell["bound_floors"].values())


def test_estimates_are_not_physical_bounds(ceiling):
    cell = estimate(ceiling, compute=["compute@0=1,fp32"])
    assert cell["bound_kind"] == "calibrated_estimate"


@pytest.fixture(params=["cuda", "npu"])
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


def run_report(
    tmp_path: Path,
    backend: str,
    stage_samples: list[list[tuple[float, int | str]]],
    duration: float = 10.0,
    bound_args: tuple[str, ...] = (),
    postprocess_s: float = 0.0,
    has_measured_end: bool = True,
) -> tuple[dict, str]:
    """Sample times are seconds before the benchmark's end."""
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "date": (END + dt.timedelta(seconds=postprocess_s)).strftime("%Y%m%d-%H%M%S"),
                **({"benchmark_end_time": END.isoformat()} if has_measured_end else {}),
                "duration": duration,
                "max_concurrency": 1,
                "completed": 1,
                "failed": 0,
                "audio_throughput": 2.0,
                "mean_audio_duration_s": 2.0,
                "mean_e2el_ms": 1000.0,
                "mean_audio_rtf": 0.5,
                "median_audio_rtf": 0.5,
                "request_audio_metrics": [{"audio_duration_s": 2.0, "audio_latency_s": 1.0}],
            }
        )
    )
    ledger = tmp_path / "ledger.json"
    args = [sys.executable, str(SCRIPT), "--cell", f"cell={result}::{tmp_path / 'stage0.csv'}"]
    if bound_args:
        peaks = tmp_path / "peaks.json"
        peaks.write_text(json.dumps({"copy_bandwidth_gbs": 2**30 / 1e9, "matmul_tflops": {"fp32": 1}}))
        args += ["--peaks", str(peaks), *bound_args]
    for stage, samples in enumerate(stage_samples):
        util = tmp_path / f"stage{stage}.csv"
        header = "timestamp, utilization.gpu [%]" if backend == "cuda" else "2026/11/11 12:00:00, AI Core(%)"
        rows = [header]
        for seconds_before_end, value in samples:
            stamp = (END - dt.timedelta(seconds=seconds_before_end)).strftime("%Y/%m/%d %H:%M:%S.%f")
            rows.append(f"{stamp}, {value}")
        util.write_text("\n".join(rows) + "\n")
        args += ["--stage-util", f"cell={stage}:{util}"]
    proc = subprocess.run(args + ["--json", str(ledger)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(ledger.read_text())[0], proc.stdout


@pytest.mark.parametrize("coverage", ["complete", "partial", "legacy", "invalid"])
def test_audio_completion_scope_and_coverage(ceiling, tmp_path, coverage):
    # T_audio = 1 + .5 * duration; the request remains open for another 2 s.
    rows = [
        {"audio_duration_s": 2.0, "audio_latency_s": 2.0},
        {"audio_duration_s": 6.0, "audio_latency_s": 4.0},
    ]
    result = {
        "max_concurrency": 1,
        "completed": 2,
        "failed": 0,
        "duration": 10.0,
        "audio_throughput": 0.8,
        "mean_audio_duration_s": 4.0,
        "mean_e2el_ms": 5000.0,
        "mean_audio_rtf": (1 + 4 / 6) / 2,
        "median_audio_rtf": (1 + 4 / 6) / 2,
        # Old/partial stage snapshots must not supply the fit's population.
        "request_stage_metrics": [{"2": {"audio_duration_s": 2.0}}, None],
    }
    if coverage != "legacy":
        result["request_audio_metrics"] = rows
    if coverage == "partial":
        result["request_audio_metrics"] = [rows[0], None]
    elif coverage == "invalid":
        rows[1]["audio_latency_s"] = float("nan")
    path = tmp_path / "audio.json"
    path.write_text(json.dumps(result))
    cell = ceiling.analyse_cell("audio", str(path), "AI Core(%)")
    assert cell["request_e2el_rtf"] == pytest.approx(1.25)
    assert cell["little_residual"] == pytest.approx(0)
    if coverage == "complete":
        assert cell["aggregate_rtf"] == pytest.approx(0.75)
        assert cell["audio_coverage"] == 1
        assert cell["measured_fixed_s"] == pytest.approx(1)
        assert cell["measured_rate"] == pytest.approx(0.5)
    else:
        assert cell["aggregate_rtf"] is None
        assert cell["audio_coverage"] == (0 if coverage == "legacy" else 0.5)
        assert "measured_fixed_s" not in cell
        assert "aggregate audio RTF unavailable" in ceiling.render([cell], [])


def test_postprocessing_does_not_shift_utilisation_window(tmp_path: Path, backend: str) -> None:
    cell, report = run_report(tmp_path, backend, [[(5, 80), (-55, 10)]], postprocess_s=60)

    assert cell["window"] == ["2026-11-11T12:00:02", "2026-11-11T12:00:08"]
    assert cell["gpu_busy_share"] == cell["stage_device_share"]["0"] == 0.8
    assert cell["gpu_samples"] == 1
    assert "could not be trimmed" not in report


def test_legacy_date_does_not_establish_measurement_window(tmp_path: Path, backend: str) -> None:
    cell, report = run_report(tmp_path, backend, [[(5, 80), (-55, 10)]], postprocess_s=60, has_measured_end=False)

    assert cell["window"] is None
    assert cell["gpu_samples"] == 2
    assert cell["util_trimmed"] is False
    assert "whole sample file, not the measured window" in report


def test_calibrated_estimate_cli(tmp_path: Path, backend: str) -> None:
    cell, report = run_report(
        tmp_path,
        backend,
        [[(5, 50)]],
        bound_args=("--compute-bound", "a@0=1,fp32", "--compute-bound", "b@1=0,fp32,1"),
    )
    assert cell["rtf_bound"] == pytest.approx(1)
    assert cell["bound_kind"] == "calibrated_estimate"
    assert "rtf_bound_mean" not in cell
    assert "Mean RTF estimate unavailable" in report
    assert "Calibrated estimates, not strict physical bounds" in report
    assert "unattributed gap" in report
    assert "empirical calibration" in report  # Above the measured .5, not a physical impossibility.


def test_zero_stage_utilisation(tmp_path: Path, backend: str) -> None:
    cell, report = run_report(tmp_path, backend, [[(5, 0)], [(5, 0)]])

    assert cell["stage_device_share"] == {"0": 0.0, "1": 0.0}
    assert cell["stage_device_g"] == {"0": 0.0, "1": 0.0}
    assert cell["stage_device_samples"] == {"0": 1, "1": 1}
    assert cell["gpu_busy_share"] == 0.0
    assert cell["g_total"] == cell["rtf_floor_no_idle"] == 0.0
    assert "stage 0: busy 0.00, g 0.0000" in report
    assert "X bound unavailable" in report
    assert "binding stage" not in report
    assert "no utilisation samples" not in report


@pytest.mark.parametrize("duration", [0.5, 3.0, 4.0])
def test_short_run_keeps_in_window_samples(tmp_path: Path, backend: str, duration: float) -> None:
    samples: list[tuple[float, int | str]] = [(duration + 1, 100), (duration / 2, 50), (-1, 100)]
    cell, report = run_report(tmp_path, backend, [samples], duration=duration)

    assert cell["window"] == [(END - dt.timedelta(seconds=duration)).isoformat(), END.isoformat()]
    assert cell["gpu_samples"] == 1
    assert cell["stage_device_samples"] == {"0": 1}
    assert cell["stage_util_trimmed"] == {"0": True}
    assert cell["stage_device_share"] == {"0": 0.5}
    assert cell["stage_device_g"] == {"0": 0.25}
    assert "binding stage 0; X bound by it: 4.00 audio-s/s" in report


@pytest.mark.parametrize("samples", [[], [(5, "N/A")], [(20, 50)]], ids=["empty", "unavailable", "outside-window"])
def test_missing_stage_utilisation(tmp_path: Path, backend: str, samples: list[tuple[float, int | str]]) -> None:
    cell, report = run_report(tmp_path, backend, [samples])

    assert cell["stage_device_share"] == {"0": None}
    assert cell["stage_device_g"] == {"0": None}
    assert cell["stage_device_samples"] == {"0": 0}
    assert cell["gpu_busy_share"] is None
    assert cell["gpu_samples"] == 0
    assert "g_total" not in cell
    assert "rtf_floor_no_idle" not in cell
    assert "stage 0: busy -, g -" in report
    assert "binding stage" not in report
    assert "WARNING cell: no utilisation samples" in report
    assert "WARNING cell: stage 0 has no utilisation samples" in report


def test_positive_stage_with_zero_and_missing_stages(tmp_path: Path, backend: str) -> None:
    cell, report = run_report(tmp_path, backend, [[(5, 0)], [], [(1, 100), (5, 50), (9, 100)]])

    assert cell["stage_device_share"] == {"0": 0.0, "1": None, "2": 0.5}
    assert cell["stage_device_g"] == {"0": 0.0, "1": None, "2": 0.25}
    assert cell["stage_device_samples"] == {"0": 1, "1": 0, "2": 1}
    assert cell["window"] == ["2026-11-11T12:00:02", "2026-11-11T12:00:08"]
    assert "binding stage 2; X bound by it: 4.00 audio-s/s" in report
    assert "WARNING cell: stage 1 has no utilisation samples" in report
