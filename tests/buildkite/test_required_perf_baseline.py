# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Execute DFX baseline logic on CPU without importing the serving runtime."""

import ast
import json
import logging
import math
import typing
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.helpers.mark import get_hardware_mark_list

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen3_tts_h100_throughput_gates_stable_metrics_only():
    config = json.loads(Path("tests/dfx/perf/tests/test_tts.json").read_text(encoding="utf-8"))
    qwen3_tts = {
        case["test_name"]: case
        for case in config
        if case["test_name"]
        in {
            "test_qwen3_tts_base",
            "test_qwen3_tts_customvoice",
            "test_qwen3_tts_voicedesign",
        }
    }
    assert set(qwen3_tts) == {
        "test_qwen3_tts_base",
        "test_qwen3_tts_customvoice",
        "test_qwen3_tts_voicedesign",
    }

    stable_throughput_metrics = {
        "median_audio_rtf",
        "audio_throughput",
        "request_throughput",
        "mean_e2el_ms",
        "mean_audio_rtf",
    }
    ttfp_metrics = {"median_audio_ttfp_ms", "mean_audio_ttfp_ms"}
    for case in qwen3_tts.values():
        for phase in case["benchmark_params"]:
            h100 = phase.get("baseline", {}).get("H100")
            if not isinstance(h100, dict):
                continue
            if phase.get("eval_phase") == "throughput":
                assert stable_throughput_metrics <= set(h100)
                assert ttfp_metrics.isdisjoint(h100)
            elif phase.get("eval_phase") in {"latency", "stress"}:
                assert ttfp_metrics <= set(h100)


@pytest.fixture
def baseline_functions():
    path = Path("tests/dfx/conftest.py")
    names = {
        "BASELINE_REGRESSION_TOLERANCE",
        "PerformanceBaselineRegressionError",
        "_LOWER_IS_BETTER_TOKENS",
        "_HIGHER_IS_BETTER_TOKENS",
        "metric_lower_is_better",
        "_is_finite_number",
        "_worse_ratio",
        "_regressed",
        "assert_baseline_within_tolerance",
        "is_hardware_nested_baseline",
        "resolve_baseline_value",
        "resolve_baseline_for_sweep",
        "pytest_addoption",
    }
    nodes: list[ast.stmt] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    namespace = {
        "__name__": __name__,
        "math": math,
        "logging": logging,
        "Mapping": Mapping,
        "Any": typing.Any,
        "TypeGuard": typing.TypeGuard,
        "pytest": pytest,
        "get_hardware_mark_list": get_hardware_mark_list,
    }
    exec(compile(ast.Module(body=[*nodes], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("hardware", [None, "", "MI325", "MI300"])
@pytest.mark.parametrize("baseline", [None, {}, {"H100": {"mean_ttft_ms": 10.0}}])
def test_required_baseline_rejects_missing_runtime_calibration(baseline_functions, baseline, hardware):
    check = baseline_functions["assert_baseline_within_tolerance"]
    check({}, baseline, hardware)  # Preserve --assert-baseline compatibility.
    with pytest.raises(AssertionError, match="No matching baseline"):
        check({}, baseline, hardware, required=True)


@pytest.mark.parametrize(
    "metric,boundary,beyond", [("mean_ttft_ms", 110.0, 110.01), ("request_throughput", 90.0, 89.99)]
)
def test_required_baseline_keeps_cuda_tolerance(baseline_functions, metric, boundary, beyond):
    check = baseline_functions["assert_baseline_within_tolerance"]
    baseline = {"MI300": {metric: 100.0}}
    check({metric: boundary}, baseline, "MI300", required=True)
    with pytest.raises(AssertionError, match="performance baseline regression"):
        check({metric: beyond}, baseline, "MI300", required=True)
    for invalid in (None, float("nan"), -1.0):
        with pytest.raises(AssertionError):
            check({metric: invalid}, baseline, "MI300", required=True)


def test_mi300_baselines_resolve_with_registered_hardware_and_preserve_cuda(baseline_functions):
    resolve = baseline_functions["resolve_baseline_for_sweep"]
    baseline = {"MI300": {"mean_ttft_ms": [1.0, 2.0]}, "H100": {"mean_ttft_ms": [3.0, 4.0]}}
    assert resolve(baseline, sweep_index=1) == {"MI300": {"mean_ttft_ms": 2.0}, "H100": {"mean_ttft_ms": 4.0}}
    with pytest.raises(IndexError):
        resolve(baseline, sweep_index=2)
    with pytest.raises(ValueError, match="Unknown hardware"):
        resolve({"invented_GPU": {"mean_ttft_ms": 1.0}}, sweep_index=0)


@pytest.mark.parametrize("runner", ["run_benchmark.py", "run_diffusion_benchmark.py"])
def test_require_flag_enables_gate_without_assert_flag(baseline_functions, runner):
    path = Path("tests/dfx/perf/scripts") / runner
    function = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "assert_result"
    )
    namespace = dict(baseline_functions)
    namespace.update(
        {
            "_assert_fixed_stage_workloads": lambda *_: None,
            "assert_duplex_artifact_publication": lambda *_: None,
        }
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    args: tuple[object, ...] = (1,) if runner.startswith("run_diffusion") else ({}, 1)
    result = {"completed": 1, "Hardware": "MI300", "mean_ttft_ms": 100.0}
    with pytest.raises(AssertionError, match="No matching baseline"):
        namespace["assert_result"](result, *args, require_baseline=True)
    result["baseline"] = {"MI300": {"mean_ttft_ms": 100.0}}
    namespace["assert_result"](result, *args, require_baseline=True)
    result["mean_ttft_ms"] = 111.0
    with pytest.raises(AssertionError, match="performance baseline regression"):
        namespace["assert_result"](result, *args, require_baseline=True)


def test_require_baseline_option_is_opt_in(baseline_functions):
    parser = pytest.Parser()
    baseline_functions["pytest_addoption"](parser)
    assert parser.parse([]).require_baseline is False
    options = parser.parse(["--require-baseline"])
    assert options.require_baseline is True
    assert options.assert_baseline is False
