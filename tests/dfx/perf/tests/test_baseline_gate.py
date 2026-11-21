# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression direction and 10% baseline tolerance for perf runners."""

import json
import math
from pathlib import Path

import pytest

from tests.dfx.conftest import (
    assert_baseline_within_tolerance,
    metric_lower_is_better,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_all_configured_baseline_metrics_have_a_direction() -> None:
    metric_names: set[str] = set()
    for config_path in sorted(Path(__file__).parent.glob("*.json")):
        configs = json.loads(config_path.read_text(encoding="utf-8"))
        for config in configs:
            for params in config.get("benchmark_params", []):
                for bucket in params.get("baseline", {}).values():
                    metric_names.update(bucket)
    assert metric_names, "No configured baseline metrics found"
    for name in sorted(metric_names):
        assert isinstance(metric_lower_is_better(name), bool), name


def test_unknown_metric_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="no regression direction"):
        metric_lower_is_better("custom_score")


def test_conflicting_metric_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        metric_lower_is_better("latency_throughput")


@pytest.mark.parametrize("hardware", ["H100", "910B", "910B4"])
def test_exact_10_percent_boundary_passes(hardware: str) -> None:
    measured = {
        "request_throughput": 0.9,
        "mean_e2el_ms": 110.0,
        "mean_video_ttfc_ms": 110.0,
        "mean_video_steady_interval_ms": 110.0,
        "p95_video_steady_interval_ms": 110.0,
        "min_video_fps": 0.9,
    }
    baseline = {
        hardware: {
            "request_throughput": 1.0,
            "mean_e2el_ms": 100.0,
            "mean_video_ttfc_ms": 100.0,
            "mean_video_steady_interval_ms": 100.0,
            "p95_video_steady_interval_ms": 100.0,
            "min_video_fps": 1.0,
        },
        "A2": {"request_throughput": 2.0},
    }
    assert_baseline_within_tolerance(measured, baseline, hardware)


@pytest.mark.parametrize("metric", ["request_throughput", "min_video_fps"])
@pytest.mark.parametrize("hardware,bucket", [("H100", "H100"), ("910B", "A2"), ("910B4", "A2")])
def test_throughput_drop_beyond_10_percent_fails(metric: str, hardware: str, bucket: str) -> None:
    measured = {"Hardware": hardware, metric: 0.89}
    baseline = {bucket: {metric: 1.0}}
    assert_baseline_within_tolerance({metric: 1.1}, baseline, hardware)
    assert_baseline_within_tolerance({metric: 0.91}, baseline, hardware)
    with pytest.raises(AssertionError, match="below baseline 1.0"):
        assert_baseline_within_tolerance(measured, baseline, hardware)
    assert measured == {"Hardware": hardware, metric: 0.89}


@pytest.mark.parametrize(
    "metric", ["latency_mean", "mean_video_ttfc_ms", "mean_video_steady_interval_ms", "p95_video_steady_interval_ms"]
)
def test_latency_increase_beyond_10_percent_fails_and_improvement_passes(metric: str) -> None:
    baseline = {"A3": {metric: 10.0, "peak_memory_mb_mean": 1000.0}}
    assert_baseline_within_tolerance(
        {metric: 8.0, "peak_memory_mb_mean": 900.0},
        baseline,
        "A3",
    )
    with pytest.raises(AssertionError, match=metric):
        assert_baseline_within_tolerance(
            {metric: 11.01, "peak_memory_mb_mean": 1000.0},
            baseline,
            "A3",
        )


@pytest.mark.parametrize(
    "hardware,bucket", [("B200", "H100"), (None, "H100"), ("", "H100"), ("910B", "H100"), ("910B4", "A3")]
)
def test_unmatched_hardware_warns_and_skips(
    hardware: str | None, bucket: str, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = {bucket: {"throughput_qps": 1.0}}
    assert_baseline_within_tolerance({"throughput_qps": 0.01}, baseline, hardware)
    output = capsys.readouterr().out
    assert "[baseline-check] WARNING SKIP: No matching baseline" in output
    assert repr(hardware) in output
    assert f"available hardware: ['{bucket}']" in output
    assert any(
        record.levelname == "WARNING" and "[baseline-check] SKIP: No matching baseline" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("baseline", [None, {}])
def test_empty_baseline_warns_and_skips(
    baseline: dict[str, object] | None, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_baseline_within_tolerance({}, baseline, "H100")
    assert "[baseline-check] WARNING SKIP: No matching baseline" in capsys.readouterr().out
    assert any(
        record.levelname == "WARNING" and "available hardware: []" in record.message for record in caplog.records
    )


def test_missing_nonfinite_negative_and_unresolved_baselines_fail() -> None:
    with pytest.raises(AssertionError, match="not a finite number"):
        assert_baseline_within_tolerance({"mean_ttft_ms": math.nan}, {"H100": {"mean_ttft_ms": 10.0}}, "H100")
    with pytest.raises(AssertionError, match="not a finite number"):
        assert_baseline_within_tolerance({}, {"H100": {"mean_ttft_ms": 10.0}}, "H100")
    with pytest.raises(AssertionError, match="negative baseline"):
        assert_baseline_within_tolerance({"mean_ttft_ms": 1.0}, {"H100": {"mean_ttft_ms": -1.0}}, "H100")
    with pytest.raises(AssertionError, match="sweep list"):
        assert_baseline_within_tolerance({"mean_ttft_ms": 1.0}, {"H100": {"mean_ttft_ms": [1.0, 2.0]}}, "H100")


@pytest.mark.parametrize("metric", ["mean_e2el_ms", "request_throughput"])
def test_negative_measurement_is_rejected(metric: str) -> None:
    with pytest.raises(AssertionError, match="negative measured value"):
        assert_baseline_within_tolerance({metric: -5.0}, {"H100": {metric: 100.0}}, "H100")


def test_zero_baseline_rejects_only_the_bad_direction() -> None:
    assert_baseline_within_tolerance({"mean_e2el_ms": 0.0}, {"H100": {"mean_e2el_ms": 0.0}}, "H100")
    with pytest.raises(AssertionError, match="mean_e2el_ms"):
        assert_baseline_within_tolerance({"mean_e2el_ms": 0.1}, {"H100": {"mean_e2el_ms": 0.0}}, "H100")
    assert_baseline_within_tolerance({"request_throughput": 0.0}, {"H100": {"request_throughput": 0.0}}, "H100")
    with pytest.raises(AssertionError, match="request_throughput"):
        assert_baseline_within_tolerance(
            {"request_throughput": -0.1},
            {"H100": {"request_throughput": 0.0}},
            "H100",
        )


def test_omni_assert_result_uses_resolved_baseline_only_when_enabled() -> None:
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    result = {
        "completed": 1,
        "Hardware": "H100",
        "request_throughput": 0.7,
        "baseline": {"H100": {"request_throughput": 1.0}},
    }
    # The config still holds the whole sweep. The gate must use the resolved scalar.
    params = {"baseline": {"H100": {"request_throughput": [1.0, 9.0]}}}
    assert_result(result, params, 1)
    with pytest.raises(AssertionError, match="request_throughput"):
        assert_result(result, params, 1, check_baseline=True)
    result.pop("baseline")
    assert_result(result, params, 1, check_baseline=True)


def test_diffusion_assert_result_checks_named_hardware_only_when_enabled() -> None:
    from tests.dfx.perf.scripts.run_diffusion_benchmark import assert_result

    result = {
        "completed": 1,
        "throughput_qps": 0.5,
        "latency_mean": 1.0,
        "baseline": {"H100": {"throughput_qps": 1.0, "latency_mean": 1.0}},
    }
    assert_result(result, 1)
    assert_result(result, 1, check_baseline=True, hardware="B200")
    with pytest.raises(AssertionError, match="throughput_qps"):
        assert_result(result, 1, check_baseline=True, hardware="H100")
