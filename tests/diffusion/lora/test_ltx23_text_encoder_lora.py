# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for native Gemma text projections and merged LTX LoRA."""

from copy import deepcopy

import pytest
import torch
from transformers import Gemma3TextConfig, Gemma3TextModel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.models.ltx2.ltx2_lora import (
    LTX23TextLinear,
    ltx23_text_lora_aliases,
    prepare_ltx23_text_encoder_lora,
)
from vllm_omni.lora.capability import MergedLoRASpec, resolve_merged_lora_spec
from vllm_omni.lora.manager import LoRAManager
from vllm_omni.lora.request import TensorLoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _set_tp(monkeypatch, size=1):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)


class _Pipeline(torch.nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.text_encoder = torch.nn.Module()
        self.text_encoder.model = torch.nn.Module()
        self.text_encoder.model.language_model = language_model
        self.transformer = torch.nn.Module()
        self.transformer.proj = torch.nn.Linear(8, 8, bias=False)

    def merged_lora_spec(self):
        return MergedLoRASpec(
            components=("text_encoder.model.language_model",),
            name_aliases=ltx23_text_lora_aliases,
        )


def _tiny_pipeline():
    torch.manual_seed(41)
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        use_cache=False,
        _attn_implementation="eager",
    )
    return _Pipeline(Gemma3TextModel(config).eval()).eval()


def _request(adapter_id, target, a, b):
    return TensorLoRARequest(
        "text_policy",
        adapter_id,
        "tensor",
        lora_tensors={
            f"base_model.model.{target}.lora_A.weight": a,
            f"base_model.model.{target}.lora_B.weight": b,
        },
        peft_config={"r": a.shape[0], "lora_alpha": a.shape[0], "target_modules": [target]},
    )


def test_native_gemma_projection_identity_forward_and_reentry(monkeypatch):
    _set_tp(monkeypatch, size=2)
    with set_current_vllm_config(VllmConfig()):
        pipeline = _tiny_pipeline()
        model = pipeline.text_encoder.model.language_model
        source = model.layers[0].self_attn.q_proj
        weight = source.weight
        input_ids = torch.tensor([[1, 2, 3]])
        baseline = model(input_ids=input_ids).last_hidden_state.detach().clone()
        dit_weight = pipeline.transformer.proj.weight
        prepare_ltx23_text_encoder_lora(pipeline)
        assert pipeline._ltx23_text_encoder_lora_prepared
        adapted = model.layers[0].self_attn.q_proj
        assert isinstance(adapted, LTX23TextLinear)
        assert adapted.weight is weight
        assert adapted.in_features == source.in_features
        assert adapted.out_features == source.out_features
        assert adapted.return_bias is False
        assert adapted.tp_size == 1
        assert adapted(torch.ones(2, 8)).shape == (2, 8)
        assert adapted(input=torch.ones(2, 8)).shape == (2, 8)
        adapted.quant_method.process_weights_after_loading(adapted)
        assert adapted.weight is weight
        assert adapted._cpu_skip_gemm_dispatch
        torch.testing.assert_close(model(input_ids=input_ids).last_hidden_state, baseline)
        assert prepare_ltx23_text_encoder_lora(pipeline) is None
        assert model.layers[0].self_attn.q_proj is adapted
        assert pipeline.transformer.proj.weight is dit_weight
        assert ltx23_text_lora_aliases("text_encoder.model.language_model.layers.0.self_attn.q_proj") == (
            "model.language_model.layers.0.self_attn.q_proj",
        )


def test_native_gemma_merged_updates_forward_and_restores(monkeypatch):
    _set_tp(monkeypatch)
    with set_current_vllm_config(VllmConfig()):
        pipeline = _tiny_pipeline()
        reference = deepcopy(pipeline)
        model = pipeline.text_encoder.model.language_model
        input_ids = torch.tensor([[1, 2, 3]])
        baseline = model(input_ids=input_ids).last_hidden_state.detach().clone()
        dit_before = pipeline.transformer.proj.weight.detach().clone()
        prepare_ltx23_text_encoder_lora(pipeline)
        assert resolve_merged_lora_spec(pipeline, stage="diffusion").components == (
            "text_encoder.model.language_model",
        )
        manager = LoRAManager(pipeline, torch.device("cpu"), torch.float32, lora_application="merged")
        target = "text_encoder.model.language_model.layers.0.self_attn.q_proj"
        layer = model.layers[0].self_attn.q_proj
        pristine = layer.weight.detach().clone()
        generator = torch.Generator().manual_seed(15)
        a = torch.randn(2, 8, generator=generator) * 0.1
        b = torch.randn(8, 2, generator=generator) * 0.1
        manager.merge_adapter(_request(1, target, a, b))
        expected = pristine + b @ a
        torch.testing.assert_close(layer.weight, expected)
        changed = model(input_ids=input_ids).last_hidden_state
        assert not torch.allclose(changed, baseline)
        # The native decoder consumes the merged projection during its real
        # forward; compare against an independent dense weight reference.
        with torch.no_grad():
            reference.text_encoder.model.language_model.layers[0].self_attn.q_proj.weight.copy_(expected)
        torch.testing.assert_close(
            reference.text_encoder.model.language_model(input_ids=input_ids).last_hidden_state,
            changed,
        )
        other_target = "text_encoder.model.language_model.layers.0.self_attn.k_proj"
        other = model.layers[0].self_attn.k_proj
        other_pristine = other.weight.detach().clone()
        b2 = -b
        manager.merge_adapter(_request(2, other_target, a, b2))
        torch.testing.assert_close(layer.weight, pristine)
        torch.testing.assert_close(other.weight, other_pristine + b2 @ a)
        previous = layer.weight.detach().clone()
        with pytest.raises(ValueError):
            manager.merge_adapter(_request(3, "transformer.proj", a, b))
        torch.testing.assert_close(layer.weight, previous)
        torch.testing.assert_close(other.weight, other_pristine + b2 @ a)
        with pytest.raises(ValueError):
            manager.merge_adapter(_request(4, other_target, a[:, :-1], b))
        torch.testing.assert_close(other.weight, other_pristine + b2 @ a)
        assert manager.list_adapters() == [2]
        torch.testing.assert_close(pipeline.transformer.proj.weight, dit_before)
        manager.close()
        manager.close()
        torch.testing.assert_close(layer.weight, pristine)
        torch.testing.assert_close(other.weight, other_pristine)
        torch.testing.assert_close(model(input_ids=input_ids).last_hidden_state, baseline)


def test_alias_and_late_unsupported_linear_leave_graph_untouched(monkeypatch):
    _set_tp(monkeypatch)
    with set_current_vllm_config(VllmConfig()):
        pipeline = _tiny_pipeline()
        model = pipeline.text_encoder.model.language_model
        first = model.layers[0].self_attn.q_proj
        model.layers[0].self_attn.k_proj = torch.nn.Linear(8, 8, bias=False)
        last = model.layers[0].self_attn.k_proj
        last.register_forward_hook(lambda _module, _args, output: output)
        with pytest.raises(ValueError, match="hooks"):
            prepare_ltx23_text_encoder_lora(pipeline)
        assert model.layers[0].self_attn.q_proj is first
        assert model.layers[0].self_attn.k_proj is last
        last._forward_hooks.clear()
        pipeline.transformer.alias = first
        with pytest.raises(ValueError, match="shared module"):
            prepare_ltx23_text_encoder_lora(pipeline)
        assert model.layers[0].self_attn.q_proj is first
        del pipeline.transformer.alias
        pipeline.transformer.alias_weight = first.weight
        with pytest.raises(ValueError, match="shared Parameter"):
            prepare_ltx23_text_encoder_lora(pipeline)
        assert model.layers[0].self_attn.q_proj is first


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bias_dtype_and_storage_alias_contract(monkeypatch, dtype):
    _set_tp(monkeypatch)
    with set_current_vllm_config(VllmConfig()):
        language_model = torch.nn.Module()
        language_model.proj = torch.nn.Linear(3, 2, bias=True, dtype=dtype)
        pipeline = _Pipeline(language_model)
        original = language_model.proj
        weight, bias = original.weight, original.bias
        input_ = torch.randn(4, 3, dtype=dtype)
        expected = original(input_)
        prepare_ltx23_text_encoder_lora(pipeline)
        adapted = language_model.proj
        assert adapted.weight is weight and adapted.bias is bias
        assert adapted.weight.dtype == dtype
        assert adapted.training == original.training
        assert isinstance(adapted(input_), torch.Tensor)
        torch.testing.assert_close(adapted(input_), expected)

        second = _Pipeline(torch.nn.Module())
        second.text_encoder.model.language_model.proj = torch.nn.Linear(3, 2, bias=False)
        candidate = second.text_encoder.model.language_model.proj
        second.transformer.alias_storage = torch.nn.Parameter(candidate.weight.detach())
        with pytest.raises(ValueError, match="storage alias"):
            prepare_ltx23_text_encoder_lora(second)
        assert second.text_encoder.model.language_model.proj is candidate
