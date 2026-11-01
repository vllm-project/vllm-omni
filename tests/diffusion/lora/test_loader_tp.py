# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.lora.loader import LoraLoaderMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("layer_cls", "layer_args", "output_sizes", "tp_size"),
    [
        pytest.param(linear.ColumnParallelLinear, (8, 16), (16,), 2, id="column"),
        pytest.param(linear.RowParallelLinear, (8, 16), (16,), 2, id="row"),
        pytest.param(linear.MergedColumnParallelLinear, (8, [8, 16]), (8, 16), 2, id="merged-column"),
        pytest.param(linear.QKVParallelLinear, (8, 2, 4, 4), (8, 8, 8), 2, id="qkv"),
        pytest.param(linear.QKVParallelLinear, (8, 2, 4, 2), (8, 4, 4), 4, id="gqa"),
    ],
)
def test_fused_lora_matches_checkpoint_sharding(monkeypatch, layer_cls, layer_args, output_sizes, tp_size):
    """Fuse/unfuse through the real loader and compare with vLLM checkpoint loading.

    Only rank discovery and accelerator selection are stubbed. TP4 GQA covers
    replicated KV heads, and the uneven merged projection catches slicing the
    concatenated tensor as one contiguous shard.
    """
    monkeypatch.setattr("vllm_omni.diffusion.lora.loader._resolve_lora_compute_device", lambda: None)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: tp_size)

    dtype = torch.float32
    output_size = sum(output_sizes)
    # Distinct rows/columns catch wrong-rank slices. Binary fractions keep the
    # load/unload round trip exact.
    full_weight = (torch.arange(output_size * 8).reshape(output_size, 8) % 17).to(dtype) / 128
    full_bias = torch.arange(output_size).to(dtype) / 128
    state_dict = {}
    deltas = []
    bias_deltas = []
    packed = len(output_sizes) > 1
    for index, size in enumerate(output_sizes):
        name = f"transformer.part{index}" if packed else "transformer.proj"
        a = torch.stack((torch.arange(8), torch.arange(7, -1, -1))).to(dtype)
        b = torch.stack((torch.arange(1, size + 1), torch.full((size,), index + 1)), dim=1).to(dtype) / 128
        bias_delta = (torch.arange(size) + index + 1).to(dtype) / 128
        state_dict[f"{name}.lora_A.weight"] = a
        state_dict[f"{name}.lora_B.weight"] = b
        state_dict[f"{name}.bias"] = bias_delta
        deltas.append(b @ a)
        bias_deltas.append(bias_delta)
    full_delta = torch.cat(deltas)
    full_bias_delta = torch.cat(bias_deltas)

    for tp_rank in range(tp_size):
        monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: tp_rank)
        with set_current_vllm_config(VllmConfig()):
            layer = layer_cls(*layer_args, bias=True, params_dtype=dtype)
        module = torch.nn.Module()
        module.proj = layer
        if packed:
            module.stacked_params_mapping = [(".proj", f".part{index}", index) for index in range(len(output_sizes))]

        # Independent oracle: fuse the full checkpoint before vLLM shards it.
        layer.weight.weight_loader(layer.weight, full_weight + full_delta)
        layer.bias.weight_loader(layer.bias, full_bias + full_bias_delta)
        expected_weight = layer.weight.detach().clone()
        expected_bias = layer.bias.detach().clone()
        layer.weight.weight_loader(layer.weight, full_weight)
        layer.bias.weight_loader(layer.bias, full_bias)
        original_weight = layer.weight.detach().clone()
        original_bias = layer.bias.detach().clone()

        loaded = LoraLoaderMixin.load_lora_into_module(state_dict, module)
        assert loaded == set(state_dict)
        torch.testing.assert_close(layer.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, expected_bias, rtol=0, atol=0)

        unloaded = LoraLoaderMixin.unload_module_lora(state_dict, module)
        assert unloaded == loaded
        torch.testing.assert_close(layer.weight, original_weight, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, original_bias, rtol=0, atol=0)


def test_fused_lora_rejects_incompatible_input_size(monkeypatch):
    """Do not silently crop an incompatible adapter into a valid local shape."""
    monkeypatch.setattr("vllm_omni.diffusion.lora.loader._resolve_lora_compute_device", lambda: None)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    with set_current_vllm_config(VllmConfig()):
        layer = linear.RowParallelLinear(8, 16, bias=False, params_dtype=torch.float32)
    with torch.no_grad():
        layer.weight.zero_()
    module = torch.nn.Module()
    module.proj = layer
    state_dict = {
        "transformer.proj.lora_A.weight": torch.ones(2, 9),
        "transformer.proj.lora_B.weight": torch.ones(16, 2),
    }
    with pytest.raises(ValueError, match="LoRA input size"):
        LoraLoaderMixin.load_lora_into_module(state_dict, module)
    assert torch.count_nonzero(layer.weight) == 0
