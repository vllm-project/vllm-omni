# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture fixed-input replay samples and record numerical measurements on CPU.

Samples contain named input and output arrays rather than pickled model objects.
Include the complete cache/history state in the inputs when replaying a stateful
operation. Independently generated autoregressive latents are not replay samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_KEY = "__metadata__"
_SCHEMA_VERSION = 1
_SAMPLE_COUNT = 64


def _sample_indices(size: int) -> list[int]:
    count = min(_SAMPLE_COUNT, size)
    return [0] if count == 1 else [i * (size - 1) // (count - 1) for i in range(count)]


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


def describe_tensor_sample(path: Path) -> dict[str, Any]:
    """Record finite replay arrays without claiming an unmeasured reference error."""
    arrays, document = _read_sample(path)
    measurements = {}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Replay tensor contains NaN or infinity: {name}")
        numeric = value.astype(np.longdouble, copy=False)
        statistics = {
            "minimum": float(numeric.min()),
            "maximum": float(numeric.max()),
            "mean": float(numeric.mean()),
            "std": float(numeric.std()),
        }
        if any(not np.isfinite(number) for number in statistics.values()):
            raise ValueError(f"Replay descriptive statistics overflow: {name}")
        indices = _sample_indices(value.size)
        measurements[name] = {
            "shape": list(value.shape),
            "dtype": document["dtypes"][name],
            "stored_dtype": str(value.dtype),
            "element_count": int(value.size),
            "finite": True,
            **statistics,
            "samples": {
                "indices": indices,
                "values": value.reshape(-1)[indices].tolist(),
            },
        }
        if name.startswith("input/"):
            measurements[name]["sha256"] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    return {"metadata": document["metadata"], "tensors": measurements}


def compare_tensor_sample(description: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare recorded output points after checking the complete replay input.

    This is a sampled regression check, not full-tensor or official-model
    equivalence. Source revisions describe the recording, not the candidate.
    """
    if not isinstance(baseline, Mapping) or baseline.get("scope") != "fixed_input_vae_regression":
        raise ValueError("A recorded fixed-input VAE precision baseline is required")
    source = baseline.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Precision baseline source is missing")
    for key, length in (("revision", 40), ("model_revision", 40), ("artifact_sha256", 64), ("model_assets_sha256", 64)):
        value = source.get(key)
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid precision baseline source {key}")
    context = baseline.get("context")
    if not isinstance(context, Mapping) or not context or description["metadata"].get("context") != context:
        raise ValueError("Precision replay context differs from the baseline")
    expected_tensors = baseline.get("tensors")
    if not isinstance(expected_tensors, Mapping) or set(expected_tensors) != set(description["tensors"]):
        raise ValueError("Precision baseline must cover every replay tensor")
    tolerances: dict[str, float] = {}
    for name in ("atol", "rtol"):
        value = baseline.get(name)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("Precision tolerances must be finite and nonnegative")
        tolerances[name] = float(value)
    violations = []
    differences = {}
    compared = 0
    for name, actual in description["tensors"].items():
        expected = expected_tensors[name]
        if not isinstance(expected, Mapping) or any(
            actual[field] != expected.get(field) for field in ("shape", "dtype", "stored_dtype")
        ):
            raise ValueError(f"Precision tensor shape/dtype differs: {name}")
        if name.startswith("input/"):
            if actual["sha256"] != expected.get("sha256"):
                raise ValueError(f"Precision replay input differs: {name}")
            continue
        samples = expected.get("samples")
        indices = _sample_indices(actual["element_count"])
        if (
            not isinstance(samples, Mapping)
            or samples.get("indices") != indices
            or any(type(index) is not int for index in samples["indices"])
            or actual["samples"]["indices"] != indices
            or not isinstance(samples.get("values"), list)
            or len(samples["values"]) != len(indices)
            or len(actual["samples"]["values"]) != len(indices)
        ):
            raise ValueError(f"Precision sample positions/count differ: {name}")
        values = samples["values"]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"Precision baseline contains nonfinite/nonnumeric values: {name}")
        differences[name] = 0.0
        for index, value, reference in zip(indices, actual["samples"]["values"], values, strict=True):
            error = abs(value - reference)
            differences[name] = max(differences[name], error)
            compared += 1
            if not math.isfinite(value) or error > tolerances["atol"] + tolerances["rtol"] * abs(reference):
                violations.append(f"{name}[{index}]: {value} != {reference} ({error=})")
    if not compared:
        raise ValueError("Precision baseline has no output samples")
    return {"passed": not violations, "sample_count": compared, "max_abs": differences, "violations": violations}
