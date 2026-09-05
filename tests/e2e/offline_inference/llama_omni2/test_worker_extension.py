# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
from types import SimpleNamespace

import pytest
import torch

from tests.e2e.offline_inference.llama_omni2.worker_extension import (
    LlamaOmni2ValidationWorkerExtension,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_validation_worker_reports_only_local_qwen2_parameter_shapes(
    monkeypatch,
):
    model = torch.nn.Module()
    model.qkv = torch.nn.Parameter(torch.empty(576, 896))
    model.o = torch.nn.Parameter(torch.empty(896, 448))
    model.gate_up = torch.nn.Parameter(torch.empty(4864, 896))
    model.down = torch.nn.Parameter(torch.empty(896, 2432))
    names = {
        "qkv": "model.language_model.model.layers.0.self_attn.qkv_proj.weight",
        "o": "model.language_model.model.layers.0.self_attn.o_proj.weight",
        "gate_up": "model.language_model.model.layers.0.mlp.gate_up_proj.weight",
        "down": "model.language_model.model.layers.0.mlp.down_proj.weight",
    }
    model.named_parameters = lambda: (
        (names[name], parameter)
        for name, parameter in (
            ("qkv", model.qkv),
            ("o", model.o),
            ("gate_up", model.gate_up),
            ("down", model.down),
        )
    )
    worker = LlamaOmni2ValidationWorkerExtension()
    worker.model_runner = SimpleNamespace(model=model)
    monkeypatch.setattr(
        "tests.e2e.offline_inference.llama_omni2.worker_extension.get_tensor_model_parallel_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "tests.e2e.offline_inference.llama_omni2.worker_extension.get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    result = worker.llama_omni2_parameter_shapes()

    assert result["tp_rank"] == 1
    assert result["tp_world_size"] == 2
    assert result["parameters"]["language_model.model.layers.0.self_attn.qkv_proj.weight"]["shape"] == [576, 896]
    assert result["parameters"]["language_model.model.layers.0.mlp.gate_up_proj.weight"]["shape"] == [4864, 896]
    expected_digest = hashlib.sha256(model.qkv.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert result["parameters"]["language_model.model.layers.0.self_attn.qkv_proj.weight"]["sha256"] == expected_digest
