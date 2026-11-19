# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-input replay recording and sampled numerical regression contracts."""

import copy
import hashlib
import json

import numpy as np
import pytest

from vllm_omni.benchmarks.realtime_video.precision import (
    compare_tensor_sample,
    describe_tensor_sample,
    save_tensor_sample,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_capture_format_records_native_dtype_shape_and_actual_statistics(tmp_path):
    import torch

    from tests.e2e.accuracy.lingbot_world.capture_precision import load_capture_format

    precision = load_capture_format()
    path = tmp_path / "capture.npz"
    precision.save_tensor_sample(
        path,
        inputs={"latent": torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)},
        outputs={"decoded": np.array([-2.0, 2.0])},
        metadata={"context": {"operation": "wan_vae_decode", "dtype": "bfloat16", "seed": 42}},
    )
    report = describe_tensor_sample(path)
    assert report["metadata"]["context"]["operation"] == "wan_vae_decode"
    assert report["tensors"]["input/latent"] == {
        "shape": [1, 2],
        "dtype": "torch.bfloat16",
        "stored_dtype": "float32",
        "element_count": 2,
        "finite": True,
        "minimum": 1.0,
        "maximum": 3.0,
        "mean": 2.0,
        "std": 1.0,
        "samples": {"indices": [0, 1], "values": [1.0, 3.0]},
        "sha256": hashlib.sha256(np.array([[1.0, 3.0]], dtype=np.float32).tobytes()).hexdigest(),
    }
    assert report["tensors"]["output/decoded"]["std"] == 2.0
    assert not {"passed", "max_abs", "psnr", "ssim"} & report.keys()
    json.dumps(report, allow_nan=False)


@pytest.fixture
def sampled_replay(tmp_path):
    inputs = np.arange(130, dtype=np.float32)
    outputs = np.linspace(-1, 1, 130, dtype=np.float32)
    context = {"operation": "wan_vae_decode", "dtype": "bfloat16", "seed": 42}
    path = tmp_path / "sampled.npz"
    save_tensor_sample(path, inputs={"latent": inputs}, outputs={"decoded": outputs}, metadata={"context": context})
    indices = np.linspace(0, 129, 64, dtype=np.int64).tolist()
    baseline = {
        "scope": "fixed_input_vae_regression",
        "source": {
            "revision": "a" * 40,
            "model_revision": "b" * 40,
            "artifact_sha256": "c" * 64,
            "model_assets_sha256": "d" * 64,
        },
        "context": context,
        "atol": 0.04,
        "rtol": 0.0,
        "tensors": {
            "input/latent": {
                "shape": [130],
                "dtype": "float32",
                "stored_dtype": "float32",
                "sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
            },
            "output/decoded": {
                "shape": [130],
                "dtype": "float32",
                "stored_dtype": "float32",
                "samples": {"indices": indices, "values": outputs[indices].tolist()},
            },
        },
    }
    return path, baseline


def test_recorded_points_pass_and_output_drift_fails(sampled_replay):
    path, baseline = sampled_replay
    actual = describe_tensor_sample(path)
    assert (
        actual["tensors"]["output/decoded"]["samples"]["indices"]
        == baseline["tensors"]["output/decoded"]["samples"]["indices"]
    )
    comparison = compare_tensor_sample(actual, baseline)
    assert comparison == {"passed": True, "sample_count": 64, "max_abs": {"output/decoded": 0.0}, "violations": []}
    actual["tensors"]["output/decoded"]["samples"]["values"][0] += 0.05
    comparison = compare_tensor_sample(actual, baseline)
    assert not comparison["passed"] and len(comparison["violations"]) == 1


def test_full_input_hash_detects_change_outside_sampled_positions(sampled_replay):
    path, baseline = sampled_replay
    with np.load(path, allow_pickle=False) as sample:
        arrays = {key: sample[key] for key in sample.files}
    assert 1 not in baseline["tensors"]["output/decoded"]["samples"]["indices"]
    arrays["input/latent"][1] += 1
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(ValueError, match="input differs"):
        compare_tensor_sample(describe_tensor_sample(path), baseline)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "empty",
        "source",
        "context",
        "tensor",
        "shape",
        "dtype",
        "indices",
        "truncated",
        "nonfinite",
        "tolerance",
    ],
)
def test_invalid_sample_baseline_cannot_pass(sampled_replay, mutation):
    path, original = sampled_replay
    baseline = copy.deepcopy(original)
    output = baseline["tensors"]["output/decoded"]
    if mutation == "missing":
        baseline = None
    elif mutation == "empty":
        baseline["tensors"] = {}
    elif mutation == "source":
        baseline["source"]["revision"] = "not-a-revision"
    elif mutation == "context":
        baseline["context"]["seed"] = 43
    elif mutation == "tensor":
        del baseline["tensors"]["input/latent"]
    elif mutation == "shape":
        output["shape"] = [129]
    elif mutation == "dtype":
        output["dtype"] = "float16"
    elif mutation == "indices":
        output["samples"]["indices"][0] = 1
    elif mutation == "truncated":
        output["samples"]["values"].pop()
    elif mutation == "nonfinite":
        output["samples"]["values"][0] = float("nan")
    else:
        baseline["atol"] = float("inf")
    with pytest.raises(ValueError):
        compare_tensor_sample(describe_tensor_sample(path), baseline)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("group", ["inputs", "outputs"])
def test_nonfinite_recorded_arrays_fail(tmp_path, value, group):
    path = tmp_path / "capture.npz"
    tensors = {"inputs": {"latent": np.ones(2)}, "outputs": {"decoded": np.ones(2)}}
    tensors[group]["latent" if group == "inputs" else "decoded"][0] = value
    save_tensor_sample(path, **tensors)
    with pytest.raises(ValueError, match="NaN or infinity"):
        describe_tensor_sample(path)


@pytest.mark.parametrize("mutation", ["metadata", "schema", "dtype_coverage", "input", "output", "name"])
def test_incomplete_replay_contract_cannot_be_recorded(tmp_path, mutation):
    path = tmp_path / "capture.npz"
    save_tensor_sample(path, inputs={"latent": np.ones(2)}, outputs={"decoded": np.ones(2)})
    with np.load(path, allow_pickle=False) as sample:
        arrays = {name: sample[name] for name in sample.files}
    metadata = json.loads(str(arrays["__metadata__"].item()))
    if mutation == "metadata":
        del arrays["__metadata__"]
    elif mutation == "schema":
        metadata["schema_version"] = 999
    elif mutation == "dtype_coverage":
        metadata["dtypes"] = {}
    elif mutation in ("input", "output"):
        key = "input/latent" if mutation == "input" else "output/decoded"
        del arrays[key]
        del metadata["dtypes"][key]
    else:
        arrays["unknown"] = arrays.pop("input/latent")
        metadata["dtypes"]["unknown"] = metadata["dtypes"].pop("input/latent")
    if mutation != "metadata":
        arrays["__metadata__"] = np.asarray(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(ValueError):
        describe_tensor_sample(path)


@pytest.mark.parametrize("value", [np.array([]), np.array(["text"]), np.array([1j])])
def test_capture_rejects_empty_or_non_real_arrays(tmp_path, value):
    with pytest.raises(ValueError, match="nonempty real numeric"):
        save_tensor_sample(tmp_path / "capture.npz", inputs={"latent": value}, outputs={"decoded": np.ones(2)})
