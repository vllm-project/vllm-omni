# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-final pipeline outputs must reach vLLM's existing PP sender unchanged."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.parallel]


def test_nonfinal_forward_bypasses_final_output_conversion(monkeypatch):
    intermediate = IntermediateTensors({"hidden_states": torch.ones(4, 8), "residual": torch.ones(4, 8)})
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: intermediate)

    def final_only_converter(*args, **kwargs):
        pytest.fail("A non-final rank must not convert intermediate tensors to OmniOutput")

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(make_omni_output=final_only_converter, have_multimodal_outputs=True)
    runner._build_model_kwargs_extra = lambda: {}
    result = runner._model_forward()
    assert result is intermediate
    hidden, multimodal = runner.extract_multimodal_outputs(result)
    assert hidden is intermediate
    assert multimodal == {}


def test_final_tensor_still_uses_model_converter(monkeypatch):
    tensor = torch.ones(4, 8)
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: tensor)
    converted = object()
    calls = []

    def convert(value, **kwargs):
        calls.append(value)
        return converted

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(make_omni_output=convert)
    runner._build_model_kwargs_extra = lambda: {}
    assert runner._model_forward() is converted
    assert calls == [tensor]
