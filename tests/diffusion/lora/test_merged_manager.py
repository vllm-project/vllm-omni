# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest, TensorLoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[(1, 0), (2, 0), (2, 1)])
def pipeline(request, monkeypatch):
    size, rank = request.param
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    with set_current_vllm_config(VllmConfig()):
        pipeline = torch.nn.Module()
        pipeline.transformer = torch.nn.Module()
        pipeline.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        pipeline.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        with torch.no_grad():
            pipeline.transformer.gate_up.weight.fill_(0.25)
        yield pipeline, size, rank


def _request(adapter_id=1):
    generator = torch.Generator().manual_seed(19)
    return TensorLoRARequest(
        "policy",
        adapter_id,
        "tensor",
        lora_tensors={
            "base_model.model.transformer.gate.lora_A.weight": torch.randn(2, 4, generator=generator),
            "base_model.model.transformer.gate.lora_B.weight": torch.randn(6, 2, generator=generator),
        },
        peft_config={"r": 2, "lora_alpha": 3, "target_modules": ["gate"]},
    )


def test_native_merge_reuses_tp_packing_without_replacing_live_modules(pipeline):
    model, size, rank = pipeline
    layer = model.transformer.gate_up
    base = layer.weight.detach().clone()
    pointer = layer.weight.data_ptr()
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    request = _request()
    original_tensors = {name: tensor.clone() for name, tensor in request.lora_tensors.items()}
    a, b = request.lora_tensors.values()
    delta = (b @ a) * 1.5
    expected = base.float()
    rows = 6 // size
    expected[:rows] += delta[rank * rows : (rank + 1) * rows]
    for _ in range(3):
        manager.merge_adapter(request)
        manager.set_request_adapter(None)
        manager.set_active_adapter(None)
        assert model.transformer.gate_up is layer
        assert layer.weight.data_ptr() == pointer
        torch.testing.assert_close(layer.weight, expected.bfloat16(), rtol=0, atol=0)
    for name, tensor in request.lora_tensors.items():
        torch.testing.assert_close(tensor, original_tensors[name], rtol=0, atol=0)
    assert manager.list_adapters() == [1]
    assert manager.remove_adapter(1)
    torch.testing.assert_close(layer.weight, base, rtol=0, atol=0)


def test_invalid_replacement_preserves_policy_and_identity(pipeline):
    model, _, _ = pipeline
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    manager.merge_adapter(_request())
    previous = model.transformer.gate_up.weight.detach().clone()
    broken = _request(2)
    broken.lora_tensors.pop(next(iter(broken.lora_tensors)))
    with pytest.raises(ValueError, match="Missing A/B"):
        manager.merge_adapter(broken)
    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)
    assert manager.list_adapters() == [1]
    with pytest.raises(ValueError, match="global policy"):
        manager.set_request_adapter(LoRARequest("runtime", 3, "unused"))
    with pytest.raises(ValueError, match="weight-update session"):
        manager.add_adapter(_request())


@pytest.mark.parametrize("factor", ["lora_A", "lora_B"])
def test_undersized_factors_preserve_previous_policy(pipeline, factor):
    model, _, _ = pipeline
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    manager.merge_adapter(_request())
    layer = model.transformer.gate_up
    previous = layer.weight.detach().clone()
    pointer = layer.weight.data_ptr()
    broken = _request(2)
    name = f"base_model.model.transformer.gate.{factor}.weight"
    tensor = broken.lora_tensors[name]
    broken.lora_tensors[name] = tensor[:, :-1] if factor == "lora_A" else tensor[:-1, :]
    with pytest.raises(ValueError, match="LoRA factor shapes"):
        manager.merge_adapter(broken)
    torch.testing.assert_close(layer.weight, previous, rtol=0, atol=0)
    assert layer.weight.data_ptr() == pointer
    assert manager.list_adapters() == [1]


def test_merged_config_reaches_stage_overrides():
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    overrides = build_stage_runtime_overrides(stage_id=0, cli_overrides={"lora_application": "merged"})
    assert overrides["lora_application"] == "merged"
