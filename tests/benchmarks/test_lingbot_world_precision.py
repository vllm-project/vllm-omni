# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw numerical recording validates finite samples without a reference verdict."""

import json

import numpy as np
import pytest

from vllm_omni.benchmarks.realtime_video.precision import describe_tensor_sample, save_tensor_sample

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
    }
    assert report["tensors"]["output/decoded"]["std"] == 2.0
    assert not {"passed", "max_abs", "psnr", "ssim"} & report.keys()
    json.dumps(report, allow_nan=False)


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
