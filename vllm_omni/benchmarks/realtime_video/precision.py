# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture fixed-input replay samples and record numerical measurements on CPU.

Samples contain named input and output arrays rather than pickled model objects.
Include the complete cache/history state in the inputs when replaying a stateful
operation. Independently generated autoregressive latents are not replay samples.
"""

from __future__ import annotations

import json
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
        measurements[name] = {
            "shape": list(value.shape),
            "dtype": document["dtypes"][name],
            "stored_dtype": str(value.dtype),
            "element_count": int(value.size),
            "finite": True,
            **statistics,
        }
    return {"metadata": document["metadata"], "tensors": measurements}
