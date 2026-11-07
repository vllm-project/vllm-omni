# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]

_PATH = Path(__file__).resolve().parents[3] / "benchmarks/ar_diffusion/benchmark_e2e.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_e2e", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_metrics_keep_physis_seconds_and_use_actual_frames():
    durations = {"physis.dit": 2.0, "physis.chunk_0000": 0.5, "stage_0_gen_ms": 3000.0}
    run = benchmark.request_metrics(81, 4.0, durations)
    assert run["e2e_fps"] == 20.25
    assert run["dit_fps"] == 40.5
    assert run["dit_ms"] == 2000.0
    assert run["stage_durations"] == durations
    assert run["physis_stage_durations_seconds"] == {
        "physis.dit": 2.0,
        "physis.chunk_0000": 0.5,
    }
    assert benchmark.request_metrics(81, 4.0, {})["dit_fps"] is None
    assert benchmark.request_metrics(81, 4.0, None)["stage_durations"] == {}


def test_summary_uses_total_frames_over_total_time():
    runs = [benchmark.request_metrics(80, seconds, {}) for seconds in (2.0, 4.0)]
    summary = benchmark.summarize(runs)
    assert summary["e2e_seconds_median"] == 3.0
    assert summary["e2e_fps_median"] == 30.0
    assert summary["e2e_fps_aggregate"] == pytest.approx(160 / 6)
    assert "dit_fps_median" not in summary


@pytest.mark.parametrize("frames", [81, 121, 162])
def test_frame_count_reads_postprocessed_video_axis(frames):
    assert benchmark.video_frame_count((1, frames, 480, 832, 3)) == frames


@pytest.mark.parametrize("shape", [(81, 480, 832, 3), (1, 3, 81, 480, 832), (2, 81, 480, 832, 3)])
def test_frame_count_rejects_other_layouts(shape):
    with pytest.raises(ValueError, match="Expected one RGB video"):
        benchmark.video_frame_count(shape)


@pytest.mark.parametrize(
    "flag,value",
    [("--frames", "0"), ("--repeats", "0"), ("--warmup", "-1"), ("--fps", "nan")],
)
def test_cli_rejects_invalid_workload(flag, value):
    argv = [
        "--model",
        "model",
        "--class-name",
        "Pipeline",
        "--deploy-config",
        "deploy.yaml",
        "--frames",
        "81",
        "--chunks",
        "19",
        "--fps",
        "16",
        "--prompt",
        "test",
        "--output-json",
        "result.json",
    ]
    with pytest.raises(SystemExit):
        benchmark.parse_args([*argv, flag, value])
