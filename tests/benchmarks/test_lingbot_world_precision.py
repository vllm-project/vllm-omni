# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Precision gates must fail on incomplete, misaligned, or nonfinite samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lingbot_world.precision import compare_tensor_samples, save_tensor_sample

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pair(tmp_path: Path, expected, actual, *, reference_inputs=None, candidate_inputs=None):
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    inputs = {"hidden_states": np.array([1, 2], dtype=np.float32), "timestep": np.array([500])}
    save_tensor_sample(reference, inputs=reference_inputs or inputs, outputs={"noise_pred": expected})
    save_tensor_sample(candidate, inputs=candidate_inputs or inputs, outputs={"noise_pred": actual})
    return reference, candidate


def test_explicit_precision_gates_and_json_report(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.array([1.0, 2.0]), np.array([1.01, 2.0]))
    result = compare_tensor_samples(reference, candidate, atol=0.02, rtol=0, min_cosine=0.99, max_relative_l2=0.01)
    assert result["status"] == "passed"
    output = result["outputs"]["noise_pred"]
    assert output["max_abs"] == pytest.approx(0.01)
    assert output["mean_abs"] == pytest.approx(0.005)
    assert output["rmse"] == pytest.approx(0.01 / np.sqrt(2))
    assert output["relative_l2"] == pytest.approx(0.01 / np.sqrt(5))
    assert output["equal"] is False
    json.dumps(result, allow_nan=False)
    failed = compare_tensor_samples(reference, candidate, atol=0.001, rtol=0)
    assert failed["status"] == "failed"
    assert failed["outputs"]["noise_pred"]["outside_tolerance_count"] == 1


@pytest.mark.parametrize("change", ["values", "dtype", "shape", "cache", "nonfinite", "signed_zero"])
def test_frozen_input_mismatch_cannot_pass_with_loose_output_tolerance(tmp_path: Path, change: str) -> None:
    inputs = {"hidden_states": np.array([1, 2], dtype=np.float32)}
    candidate_inputs = {key: value.copy() for key, value in inputs.items()}
    if change == "values":
        candidate_inputs["hidden_states"][0] = 3
    elif change == "dtype":
        candidate_inputs["hidden_states"] = candidate_inputs["hidden_states"].astype(np.float64)
    elif change == "shape":
        candidate_inputs["hidden_states"] = candidate_inputs["hidden_states"].reshape(1, 2)
    elif change == "cache":
        candidate_inputs["kv_cache"] = np.zeros(2)
    elif change == "nonfinite":
        candidate_inputs["hidden_states"][0] = np.nan
    else:
        inputs["hidden_states"][0] = 0.0
        candidate_inputs["hidden_states"][0] = -0.0
    reference, candidate = _pair(
        tmp_path, np.ones(2), np.ones(2), reference_inputs=inputs, candidate_inputs=candidate_inputs
    )
    result = compare_tensor_samples(reference, candidate, atol=100, rtol=100)
    assert result["status"] == "failed"
    assert result["inputs_match"] is False
    assert result["outputs"] == {}


@pytest.mark.parametrize("actual", [np.array([1.0, np.nan]), np.array([1.0, np.inf]), np.ones((1, 2))])
def test_nonfinite_or_differently_shaped_outputs_fail(tmp_path: Path, actual: np.ndarray) -> None:
    reference, candidate = _pair(tmp_path, np.ones(2), actual)
    result = compare_tensor_samples(reference, candidate, atol=100, rtol=100)
    assert result["status"] == "failed"
    assert result["outputs"]["noise_pred"]["error"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("actual, cosine, relative_l2", [([0, 0], 1.0, 0.0), ([1, 0], 0.0, None)])
def test_zero_reference_norm_does_not_hide_nonzero_candidate(
    tmp_path: Path, actual: list[int], cosine: float, relative_l2: float | None
) -> None:
    reference, candidate = _pair(tmp_path, np.zeros(2), np.array(actual))
    result = compare_tensor_samples(reference, candidate, atol=10, rtol=0, min_cosine=0.99, max_relative_l2=1.0)
    row = result["outputs"]["noise_pred"]
    assert row["cosine"] == cosine
    assert row["relative_l2"] == relative_l2
    assert result["status"] == ("passed" if actual == [0, 0] else "failed")
    json.dumps(result, allow_nan=False)


def test_relative_tolerance_uses_reference_values(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.array([100.0, 0.0]), np.array([101.0, 0.01]))
    result = compare_tensor_samples(reference, candidate, atol=0, rtol=0.02)
    assert result["status"] == "failed"
    assert result["outputs"]["noise_pred"]["outside_tolerance_count"] == 1


def test_context_mismatch_and_incomplete_sample_fail(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.ones(2), np.ones(2))
    save_tensor_sample(
        candidate,
        inputs={"hidden_states": np.ones(2)},
        outputs={"noise_pred": np.ones(2)},
        metadata={"context": {"chunk": 159, "step": 3}},
    )
    result = compare_tensor_samples(reference, candidate, atol=0, rtol=0)
    assert result["status"] == "failed"
    assert "context differs" in result["errors"][0]
    with candidate.open("wb") as handle:
        np.savez(handle, noise_pred=np.ones(2))
    result = compare_tensor_samples(reference, candidate, atol=0, rtol=0)
    assert result["status"] == "failed"
    assert "missing metadata" in result["errors"][0]


def test_missing_output_names_are_not_silently_truncated(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.ones(2), np.ones(2))
    with np.load(reference, allow_pickle=False) as sample:
        frozen = {key: sample[key] for key in sample.files}
    metadata = json.loads(str(frozen["__metadata__"].item()))
    frozen["output/kv"] = np.ones(2)
    metadata["dtypes"]["output/kv"] = "float64"
    frozen["__metadata__"] = np.asarray(json.dumps(metadata))
    with reference.open("wb") as handle:
        np.savez(handle, **frozen)
    result = compare_tensor_samples(reference, candidate, atol=0, rtol=0)
    assert result["status"] == "failed"
    assert "output/kv" in result["errors"][0]


@pytest.mark.parametrize("atol,rtol", [(-1, 0), (0, -1), (float("inf"), 0), (0, float("nan"))])
def test_invalid_tolerances_are_rejected(tmp_path: Path, atol: float, rtol: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_tensor_samples(tmp_path / "missing", tmp_path / "missing", atol=atol, rtol=rtol)


def test_metric_overflow_is_a_failure_with_valid_json(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.array([1e308]), np.array([-1e308]))
    result = compare_tensor_samples(reference, candidate, atol=0, rtol=0)
    assert result["status"] == "failed"
    assert "overflow" in result["outputs"]["noise_pred"]["error"]
    json.dumps(result, allow_nan=False)


def test_large_integer_difference_is_not_rounded_into_a_pass(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.array([2**53], dtype=np.int64), np.array([2**53 + 1], dtype=np.int64))
    result = compare_tensor_samples(reference, candidate, atol=0, rtol=0)
    assert result["status"] == "failed"
    assert result["outputs"]["noise_pred"]["max_abs"] == 1.0


def test_small_nonzero_norms_do_not_hide_opposite_outputs(tmp_path: Path) -> None:
    reference, candidate = _pair(tmp_path, np.array([1e-200]), np.array([-1e-200]))
    result = compare_tensor_samples(reference, candidate, atol=1, rtol=0, min_cosine=0.99, max_relative_l2=0.01)
    row = result["outputs"]["noise_pred"]
    assert result["status"] == "failed"
    assert row["cosine"] == pytest.approx(-1)
    assert row["relative_l2"] == pytest.approx(2)
    assert row["rmse"] == pytest.approx(2e-200, abs=0)
