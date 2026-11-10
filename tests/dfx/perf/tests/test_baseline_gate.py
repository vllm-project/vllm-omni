# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression direction and 20% baseline tolerance for perf runners."""

import math

import pytest

from tests.dfx.perf.baseline_gate import (
    BASELINE_REGRESSION_TOLERANCE,
    assert_baseline_within_tolerance,
    metric_lower_is_better,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LOWER_METRICS = (
    "latency_mean",
    "latency_p99",
    "mean_audio_rtf",
    "mean_audio_ttfp_ms",
    "mean_e2el_ms",
    "mean_peak_memory_mb",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "median_audio_rtf",
    "median_audio_ttfp_ms",
    "peak_memory_mb_max",
    "peak_memory_mb_mean",
)
_HIGHER_METRICS = ("audio_throughput", "request_throughput", "throughput_qps")


@pytest.mark.parametrize("name", _LOWER_METRICS)
def test_lower_is_better_metrics(name: str) -> None:
    assert metric_lower_is_better(name) is True


@pytest.mark.parametrize("name", _HIGHER_METRICS)
def test_higher_is_better_metrics(name: str) -> None:
    assert metric_lower_is_better(name) is False


def test_unknown_metric_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="no regression direction"):
        metric_lower_is_better("custom_score")


def test_conflicting_metric_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        metric_lower_is_better("latency_throughput")


def test_exact_20_percent_boundary_passes() -> None:
    tolerance = BASELINE_REGRESSION_TOLERANCE
    measured = {
        "request_throughput": 1.0 * (1 - tolerance),
        "mean_e2el_ms": 100.0 * (1 + tolerance),
    }
    baseline = {"H100": {"request_throughput": 1.0, "mean_e2el_ms": 100.0}}
    assert_baseline_within_tolerance(measured, baseline, "H100")


def test_throughput_drop_beyond_20_percent_fails() -> None:
    measured = {"request_throughput": 0.79}
    baseline = {"H100": {"request_throughput": 1.0}}
    with pytest.raises(AssertionError, match="below baseline 1.0"):
        assert_baseline_within_tolerance(measured, baseline, "H100")


def test_latency_increase_beyond_20_percent_fails_and_improvement_passes() -> None:
    baseline = {"A3": {"latency_mean": 10.0, "peak_memory_mb_mean": 1000.0}}
    assert_baseline_within_tolerance(
        {"latency_mean": 8.0, "peak_memory_mb_mean": 900.0},
        baseline,
        "A3",
    )
    with pytest.raises(AssertionError, match="latency_mean"):
        assert_baseline_within_tolerance(
            {"latency_mean": 12.01, "peak_memory_mb_mean": 1000.0},
            baseline,
            "A3",
        )


def test_unmatched_hardware_and_empty_baseline_are_not_gated() -> None:
    measured = {"throughput_qps": 0.01}
    baseline = {"H100": {"throughput_qps": 1.0}}
    assert_baseline_within_tolerance(measured, baseline, "B200")
    assert_baseline_within_tolerance(measured, baseline, None)
    assert_baseline_within_tolerance(measured, {}, "H100")
    assert_baseline_within_tolerance(measured, None, "H100")


def test_missing_nonfinite_negative_and_unresolved_baselines_fail() -> None:
    with pytest.raises(AssertionError, match="not a finite number"):
        assert_baseline_within_tolerance({"mean_ttft_ms": math.nan}, {"H100": {"mean_ttft_ms": 10.0}}, "H100")
    with pytest.raises(AssertionError, match="not a finite number"):
        assert_baseline_within_tolerance({}, {"H100": {"mean_ttft_ms": 10.0}}, "H100")
    with pytest.raises(AssertionError, match="negative baseline"):
        assert_baseline_within_tolerance({"mean_ttft_ms": 1.0}, {"H100": {"mean_ttft_ms": -1.0}}, "H100")
    with pytest.raises(AssertionError, match="sweep list"):
        assert_baseline_within_tolerance({"mean_ttft_ms": 1.0}, {"H100": {"mean_ttft_ms": [1.0, 2.0]}}, "H100")


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
