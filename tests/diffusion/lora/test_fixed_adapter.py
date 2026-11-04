# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Pipeline(torch.nn.Module):
    diffusion_lora_fixed_adapter = True
    diffusion_lora_scale_in_epilogue = True

    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.qkv_proj = linear.QKVParallelLinear(8, 2, 4, bias=False, params_dtype=torch.bfloat16)
        self.transformer.fc1 = linear.MergedColumnParallelLinear(8, [10, 10], bias=False, params_dtype=torch.bfloat16)
        generator = torch.Generator().manual_seed(71)
        self.originals = dict(self.transformer.named_children())
        weights = {}
        for name, module in self.originals.items():
            module.weight.data.copy_(torch.randn(module.weight.shape, generator=generator))
            weights[f"transformer.{name}"] = LoRALayerWeights(
                f"transformer.{name}",
                8,
                11,
                torch.randn((8, 8), generator=generator).bfloat16(),
                torch.randn((module.output_size, 8), generator=generator).bfloat16(),
            )
        self.adapter = LoRAModel(1, 8, weights)
        self.config = PEFTHelper.from_dict({"r": 8, "lora_alpha": 11, "target_modules": ["qkv_proj", "fc1"]})
        self.loads = 0
        self.events = []
        self.manager = None
        self.bank = None
        self.fail_bind = False

    def _load_diffusion_lora_adapter(self, *, lora_request, **kwargs):
        self.loads += 1
        return self.adapter.clone(lora_request.lora_int_id), self.config

    def _get_diffusion_lora_packed_modules(self, name, module):
        return [name.rsplit(".", 1)[-1]]

    def _bind_diffusion_lora_manager(self, manager):
        self.events.append("bind" if manager is not None else "unbind")
        self.manager = manager
        if manager is not None and self.fail_bind:
            raise RuntimeError("binding failed")

    def _create_diffusion_merged_lora(self, *, lora_model, modules, scale):
        assert scale == 1.0
        if self.bank is not None:
            assert all(layer.base_layer is self.originals[name.rsplit(".", 1)[-1]] for name, layer in modules.items())
        return self.bank


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: lambda _layer, x, weight, bias=None: torch.nn.functional.linear(x, weight, bias),
    )
    with set_current_vllm_config(VllmConfig()):
        yield _Pipeline()


def _manager(pipeline, tmp_path):
    return DiffusionLoRAManager(pipeline, torch.device("cpu"), torch.bfloat16, lora_path=str(tmp_path), lora_scale=0.0)


@pytest.mark.parametrize("projection", ["qkv_proj", "fc1"])
def test_fused_binding_preserves_native_alpha_and_request_defaults(pipeline, tmp_path, projection):
    manager = _manager(pipeline, tmp_path)
    layer = getattr(pipeline.transformer, projection)
    weights = pipeline.adapter.loras[f"transformer.{projection}"]
    assert layer.n_slices == 1
    torch.testing.assert_close(layer.lora_b_stacked[0][0, 0], weights.lora_b, rtol=0, atol=0)
    assert weights.scaling == 11 / 8  # optimize() must not mutate the shared normalized payload.
    x = torch.randn(7, 8, generator=torch.Generator().manual_seed(53)).bfloat16()
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x, layer.base_layer.weight)
        expected.addmm_(x @ weights.lora_a.T, weights.lora_b.T, alpha=11 / 8)
        pointers = (layer.lora_a_stacked[0].data_ptr(), layer.lora_b_stacked[0].data_ptr())
        for request, scale in ((None, 0.0), (LoRARequest("ignored", 999, "/does/not/exist"), 9.0)):
            manager.set_request_adapter(request, scale)
            torch.testing.assert_close(layer(x)[0], expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="teacher failure"), manager.suspended():
            with manager.suspended():
                torch.testing.assert_close(layer(x)[0], torch.nn.functional.linear(x, layer.base_layer.weight))
            assert manager._active_adapter_id is None
            raise RuntimeError("teacher failure")
        torch.testing.assert_close(layer(x)[0], expected, rtol=0, atol=0)
    assert pointers == (layer.lora_a_stacked[0].data_ptr(), layer.lora_b_stacked[0].data_ptr())
    assert pipeline.loads == 1
    assert not manager.remove_adapter(manager._active_adapter_id)
    manager.close()
    assert getattr(pipeline.transformer, projection) is pipeline.originals[projection]
    assert pipeline.events == ["bind", "unbind"]
    manager.close()
    assert pipeline.events == ["bind", "unbind"]


class _Bank:
    def __init__(self, pipeline, *, fail_activate=False):
        self.pipeline = pipeline
        self.base = {name: module.weight for name, module in pipeline.originals.items()}
        self.student = {name: torch.nn.Parameter(weight + 0.25) for name, weight in self.base.items()}
        self.fail_activate = fail_activate
        self.closed = 0

    def activate(self):
        for name, module in self.pipeline.originals.items():
            module.weight = self.student[name]
            if self.fail_activate:
                raise RuntimeError("student installation failed")

    def suspend(self):
        for name, module in self.pipeline.originals.items():
            module.weight = self.base[name]

    def close(self):
        self.pipeline.events.append("bank-close")
        self.suspend()
        self.closed += 1


def test_merged_bank_switches_storage_without_adding_sidecar(pipeline, tmp_path):
    pipeline.bank = bank = _Bank(pipeline)
    manager = _manager(pipeline, tmp_path)
    layer = pipeline.transformer.qkv_proj
    x = torch.randn(4, 8).bfloat16()
    with torch.inference_mode():
        torch.testing.assert_close(layer(x)[0], torch.nn.functional.linear(x, bank.student["qkv_proj"]))
        for _ in range(3):
            with manager.suspended():
                assert layer.base_layer.weight is bank.base["qkv_proj"]
                torch.testing.assert_close(layer(x)[0], torch.nn.functional.linear(x, bank.base["qkv_proj"]))
            assert layer.base_layer.weight is bank.student["qkv_proj"]
            assert not any(layer._diffusion_lora_active_slices)
    manager.close()
    manager.close()
    assert bank.closed == 1
    assert pipeline.events == ["bind", "unbind", "bank-close"]
    assert pipeline.transformer.qkv_proj.weight is bank.base["qkv_proj"]


@pytest.mark.parametrize("failure", ["activate", "bind"])
def test_constructor_failure_restores_original_modules_and_parameters(pipeline, tmp_path, failure):
    pipeline.bank = bank = _Bank(pipeline, fail_activate=failure == "activate")
    pipeline.fail_bind = failure == "bind"
    with pytest.raises(RuntimeError, match="failed"):
        _manager(pipeline, tmp_path)
    assert pipeline.manager is None
    assert bank.closed == 1
    for name, original in pipeline.originals.items():
        assert getattr(pipeline.transformer, name) is original
        assert original.weight is bank.base[name]


def test_packed_scaling_mismatch_rejects_and_unwinds(pipeline, tmp_path):
    pipeline._get_diffusion_lora_packed_modules = lambda *_: None
    with pytest.raises(ValueError, match="one fused A/B pair"):
        _manager(pipeline, tmp_path)
    assert pipeline.manager is None
    assert all(getattr(pipeline.transformer, name) is original for name, original in pipeline.originals.items())
