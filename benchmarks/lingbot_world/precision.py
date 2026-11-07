# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture fixed-input replay samples and compare numerical precision on CPU.

Samples contain named input and output arrays rather than pickled model objects.
Include the complete cache/history state in the inputs when replaying a stateful
operation. Independently generated autoregressive latents are not replay samples.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_KEY = "__metadata__"
_SCHEMA_VERSION = 1


def _array(value: Any) -> tuple[np.ndarray, str]:
    original_dtype = str(getattr(value, "dtype", ""))
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            value = value.numpy()
        except TypeError:
            # NumPy does not represent PyTorch BF16/FP8 tensors directly.
            value = value.float().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "biuf" or result.size == 0:
        raise ValueError("Replay tensors must be nonempty real numeric arrays.")
    return np.array(result, copy=True), original_dtype or str(result.dtype)


def save_tensor_sample(
    path: Path,
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a replay sample; PyTorch tensors are detached and copied to CPU.

    ``inputs`` must describe the full operation input, including timestep,
    conditioning, and KV/VAE cache state. Use ``metadata`` for chunk/step and
    model/config provenance, with an optional identical ``context`` in both arms.
    """
    if not inputs or not outputs:
        raise ValueError("A replay sample requires both inputs and outputs.")
    arrays: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    for group, values in (("input", inputs), ("output", outputs)):
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Replay tensor names must be nonempty strings.")
            key = f"{group}/{name}"
            arrays[key], dtypes[key] = _array(value)
    document = {"schema_version": _SCHEMA_VERSION, "dtypes": dtypes, "metadata": dict(metadata or {})}
    arrays[_METADATA_KEY] = np.asarray(json.dumps(document, allow_nan=False, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _read_sample(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as sample:
        if _METADATA_KEY not in sample.files:
            raise ValueError(f"{path}: not a fixed-input replay sample (missing metadata).")
        document = json.loads(str(sample[_METADATA_KEY].item()))
        if not isinstance(document, dict) or document.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported replay sample schema.")
        arrays = {key: sample[key] for key in sample.files if key != _METADATA_KEY}
    if not isinstance(document.get("metadata"), dict) or not isinstance(document.get("dtypes"), dict):
        raise ValueError(f"{path}: invalid replay metadata.")
    if set(document["dtypes"]) != set(arrays):
        raise ValueError(f"{path}: replay dtype metadata does not cover every tensor.")
    if not any(key.startswith("input/") for key in arrays) or not any(key.startswith("output/") for key in arrays):
        raise ValueError(f"{path}: replay inputs or outputs are missing.")
    for key, value in arrays.items():
        if not key.startswith(("input/", "output/")) or not key.split("/", 1)[1]:
            raise ValueError(f"{path}: unknown replay tensor name {key!r}.")
        if value.dtype.kind not in "biuf" or value.size == 0:
            raise ValueError(f"{path}: {key} is not a nonempty real numeric tensor.")
    return arrays, document


def _tensor_metrics(reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    # Extended precision retains int64 differences; scaled reductions preserve
    # tiny nonzero norms and avoid squaring large values before normalization.
    expected = reference.astype(np.longdouble, copy=False).reshape(-1)
    actual = candidate.astype(np.longdouble, copy=False).reshape(-1)
    diff = actual - expected
    absolute = np.abs(diff)
    reference_scale = np.max(np.abs(expected))
    candidate_scale = np.max(np.abs(actual))
    diff_scale = np.max(absolute)
    reference_unit = expected / reference_scale if reference_scale else expected
    candidate_unit = actual / candidate_scale if candidate_scale else actual
    diff_unit = diff / diff_scale if diff_scale else diff
    reference_unit_norm = np.sqrt(np.sum(reference_unit * reference_unit))
    candidate_unit_norm = np.sqrt(np.sum(candidate_unit * candidate_unit))
    diff_unit_norm = np.sqrt(np.sum(diff_unit * diff_unit))
    if reference_scale == 0 or candidate_scale == 0:
        cosine = 1.0 if reference_scale == candidate_scale == 0 else 0.0
    else:
        cosine = float(
            np.clip(np.sum(reference_unit * candidate_unit) / (reference_unit_norm * candidate_unit_norm), -1, 1)
        )
    # A nonzero error against an all-zero reference has undefined relative L2.
    relative_l2 = (
        float((diff_scale / reference_scale) * (diff_unit_norm / reference_unit_norm))
        if reference_scale
        else (0.0 if diff_scale == 0 else None)
    )
    within = absolute <= atol + rtol * np.abs(expected)
    return {
        "max_abs": float(absolute.max()),
        "mean_abs": float(diff_scale * np.mean(np.abs(diff_unit))),
        "rmse": float(diff_scale * np.sqrt(np.mean(diff_unit * diff_unit))),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "allclose": bool(within.all()),
        "outside_tolerance_count": int(np.count_nonzero(~within)),
        "element_count": int(expected.size),
        "equal": bool(np.array_equal(expected, actual)),
        "bitwise_equal": bool(reference.dtype == candidate.dtype and reference.tobytes() == candidate.tobytes()),
    }


def compare_tensor_samples(
    reference: Path,
    candidate: Path,
    *,
    atol: float,
    rtol: float,
    min_cosine: float | None = None,
    max_relative_l2: float | None = None,
) -> dict[str, Any]:
    """Check full input alignment and gate each output with explicit tolerances.

    Zero-reference relative L2 is ``null`` for a nonzero candidate and fails a
    relative-L2 gate. Nonfinite values always fail, even when both arms match.
    """
    for name, value in (("atol", atol), ("rtol", rtol), ("max_relative_l2", max_relative_l2)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")
    if min_cosine is not None and (not math.isfinite(min_cosine) or not -1 <= min_cosine <= 1):
        raise ValueError("min_cosine must be finite and between -1 and 1.")
    report: dict[str, Any] = {
        "scope": "fixed_input_replay",
        "reference": str(reference),
        "candidate": str(candidate),
        "status": "failed",
        "thresholds": {"atol": atol, "rtol": rtol, "min_cosine": min_cosine, "max_relative_l2": max_relative_l2},
        "inputs_match": False,
        "outputs": {},
        "errors": [],
    }
    try:
        expected, reference_document = _read_sample(Path(reference))
        actual, candidate_document = _read_sample(Path(candidate))
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["errors"].append(str(error))
        return report
    report["reference_metadata"] = reference_document["metadata"]
    report["candidate_metadata"] = candidate_document["metadata"]
    if reference_document["metadata"].get("context") != candidate_document["metadata"].get("context"):
        report["errors"].append("Replay context differs (chunk, step, or workload identity).")
    if set(expected) != set(actual):
        report["errors"].append(
            f"Tensor names differ; missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}."
        )
        return report
    for name in sorted(key for key in expected if key.startswith("input/")):
        left, right = expected[name], actual[name]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or reference_document["dtypes"][name] != candidate_document["dtypes"][name]
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or not np.array_equal(left, right)
            or left.tobytes() != right.tobytes()
        ):
            report["errors"].append(f"Frozen input differs or is nonfinite: {name}.")
    report["inputs_match"] = not report["errors"]
    if not report["inputs_match"]:
        return report
    for name in sorted(key for key in expected if key.startswith("output/")):
        left, right = expected[name], actual[name]
        row: dict[str, Any] = {
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
            "reference_dtype": reference_document["dtypes"][name],
            "candidate_dtype": candidate_document["dtypes"][name],
            "status": "failed",
        }
        report["outputs"][name.removeprefix("output/")] = row
        if left.shape != right.shape:
            row["error"] = "Output shapes differ."
            continue
        row["finite"] = bool(np.isfinite(left).all() and np.isfinite(right).all())
        if not row["finite"]:
            row["error"] = "Output contains NaN or infinity."
            continue
        exact_integer_limit = 2 ** (np.finfo(np.longdouble).nmant + 1)
        if any(
            array.dtype.kind in "iu" and max(abs(int(array.min())), abs(int(array.max()))) > exact_integer_limit
            for array in (left, right)
        ):
            row["error"] = "Integer output exceeds this platform's exact metric representation."
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            row.update(_tensor_metrics(left, right, atol=atol, rtol=rtol))
        invalid_metrics = [key for key, value in row.items() if isinstance(value, float) and not math.isfinite(value)]
        if invalid_metrics:
            row["error"] = f"Numerical metric overflow: {', '.join(invalid_metrics)}."
            for key in invalid_metrics:
                row[key] = None
            continue
        if row["reference_dtype"] != row["candidate_dtype"]:
            row["bitwise_equal"] = False
        passed = row["allclose"]
        if min_cosine is not None:
            passed = passed and row["cosine"] >= min_cosine
        if max_relative_l2 is not None:
            passed = passed and row["relative_l2"] is not None and row["relative_l2"] <= max_relative_l2
        row["status"] = "passed" if passed else "failed"
    if all(row["status"] == "passed" for row in report["outputs"].values()):
        report["status"] = "passed"
    return report
