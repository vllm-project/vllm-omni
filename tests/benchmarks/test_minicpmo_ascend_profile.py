# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

from benchmarks.competition.minicpmo_ascend.profile import service_root
from benchmarks.competition.minicpmo_ascend.profile_analysis import (
    analyze_trace_root,
    compare_reports,
)
from benchmarks.competition.minicpmo_ascend.profile_config import build_profile_config

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _write_csv(path: Path, fieldnames: list[str], values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def test_profile_config_targets_only_selected_stages(tmp_path: Path) -> None:
    config = {"pipeline": "minicpmo_4_5", "stages": [{"stage_id": 0}, {"stage_id": 1}, {"stage_id": 2}]}
    result = build_profile_config(config, trace_dir=tmp_path / "traces", stages={1, 2})

    assert "profiler_config" not in result["stages"][0]
    assert result["stages"][1]["profiler_config"]["profiler"] == "torch"
    assert result["stages"][2]["profiler_config"]["torch_profiler_dir"] == str((tmp_path / "traces").resolve())


def test_profile_config_rejects_unknown_stage(tmp_path: Path) -> None:
    config = {"stages": [{"stage_id": 0}]}
    try:
        build_profile_config(config, trace_dir=tmp_path, stages={2})
    except ValueError as exc:
        assert "not in deployment config" in str(exc)
    else:
        raise AssertionError("unknown profile stage was accepted")


def test_service_root_normalizes_v1_suffix() -> None:
    assert service_root("http://localhost:8099/v1/") == "http://localhost:8099"
    assert service_root("http://localhost:8099") == "http://localhost:8099"


def test_profile_analysis_aggregates_exported_csv(tmp_path: Path) -> None:
    output = tmp_path / "stage2" / "mindstudio_profiler_output"
    _write_csv(
        output / "op_statistic.csv",
        ["OP Type", "Count", "Total Time(us)"],
        [
            {"OP Type": "MatMul", "Count": 2, "Total Time(us)": 300},
            {"OP Type": "Scatter", "Count": 1, "Total Time(us)": 200},
        ],
    )
    _write_csv(
        output / "api_statistic.csv",
        ["API Name", "Count", "Time(us)"],
        [{"API Name": "aclrtSynchronizeStream", "Count": 2, "Time(us)": 100}],
    )
    _write_csv(
        output / "kernel_details.csv",
        ["Name", "Duration(us)"],
        [
            {"Name": "matmul_kernel", "Duration(us)": 120},
            {"Name": "tiny_kernel", "Duration(us)": 20},
        ],
    )
    _write_csv(
        output / "operator_details.csv",
        ["Name", "Host Self Duration(us)", "Device Self Duration(us)"],
        [{"Name": "aten::mm", "Host Self Duration(us)": 10, "Device Self Duration(us)": 120}],
    )

    result = analyze_trace_root(tmp_path)

    assert result["operators"]["calls"] == 3
    assert result["operators"]["total_us"] == 500
    assert result["apis"]["top"][0]["name"] == "aclrtSynchronizeStream"
    assert result["kernels"]["calls"] == 2
    assert result["kernels"]["small_le_50us_ratio"] == 0.5
    assert result["torch_operators"]["device_self_total_us"] == 120

    comparison = compare_reports(result, {**result, "kernels": {**result["kernels"], "total_us": 100}})
    assert comparison["compatible"] is True
    assert comparison["metrics"]["kernel_time_us"]["delta"] == -40


def test_profile_comparison_rejects_mismatched_stage_scope() -> None:
    common = {
        "trace_root": "/trace",
        "operators": {"total_us": 1, "calls": 1},
        "apis": {"total_us": 1, "calls": 1},
        "kernels": {"total_us": 1, "calls": 1, "small_le_50us_ratio": 0.0},
    }
    result = compare_reports(
        {**common, "capture": {"profile_stages": [1]}},
        {**common, "capture": {"profile_stages": [2]}},
    )

    assert result["compatible"] is False
    assert result["mismatches"][0]["field"] == "profile_stages"
