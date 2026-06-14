# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_add_forward_compat_kwargs():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "vllm_omni" / "worker_v2" / "forward_compat.py"
    spec = importlib.util.spec_from_file_location("omni_forward_compat", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_forward_compat_kwargs


def test_forward_compat_kwargs_include_sampling_metadata_and_sampler():
    add_forward_compat_kwargs = _load_add_forward_compat_kwargs()
    model_inputs = {"input_ids": object()}
    sampling_metadata = object()
    sampler = object()
    input_batch = SimpleNamespace(
        sampling_metadata=sampling_metadata,
        logits_indices=[3, 7],
    )

    add_forward_compat_kwargs(model_inputs, input_batch, sampler)

    assert model_inputs["sampling_metadata"] is sampling_metadata
    assert model_inputs["logits_index"] == [3, 7]
    assert model_inputs["sampler"] is sampler


def test_forward_compat_kwargs_do_not_clobber_explicit_values():
    add_forward_compat_kwargs = _load_add_forward_compat_kwargs()
    model_inputs = {
        "sampling_metadata": "explicit-metadata",
        "logits_index": "explicit-logits",
        "sampler": "explicit-sampler",
    }
    input_batch = SimpleNamespace(
        sampling_metadata="batch-metadata",
        logits_indices="batch-logits",
    )

    add_forward_compat_kwargs(model_inputs, input_batch, sampler="runner-sampler")

    assert model_inputs["sampling_metadata"] == "explicit-metadata"
    assert model_inputs["logits_index"] == "explicit-logits"
    assert model_inputs["sampler"] == "explicit-sampler"


if __name__ == "__main__":
    test_forward_compat_kwargs_include_sampling_metadata_and_sampler()
    test_forward_compat_kwargs_do_not_clobber_explicit_values()
