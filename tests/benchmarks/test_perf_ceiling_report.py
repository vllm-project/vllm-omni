# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise utilisation edge cases through the ceiling-report CLI."""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).resolve().parents[2] / ".claude/skills/omni-perf-ceiling/scripts/ceiling_report.py"
END = dt.datetime(2026, 11, 11, 12, 0, 10)


@pytest.fixture(params=["cuda", "npu"])
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


def run_report(
    tmp_path: Path,
    backend: str,
    stage_samples: list[list[tuple[float, int | str]]],
    duration: float = 10.0,
) -> tuple[dict, str]:
    """Sample times are seconds before the benchmark's end."""
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "date": END.strftime("%Y%m%d-%H%M%S"),
                "duration": duration,
                "max_concurrency": 1,
                "completed": 1,
                "failed": 0,
                "audio_throughput": 2.0,
                "mean_audio_duration_s": 2.0,
                "mean_e2el_ms": 1000.0,
                "mean_audio_rtf": 0.5,
                "median_audio_rtf": 0.5,
            }
        )
    )
    ledger = tmp_path / "ledger.json"
    args = [sys.executable, str(SCRIPT), "--cell", f"cell={result}::{tmp_path / 'stage0.csv'}"]
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
