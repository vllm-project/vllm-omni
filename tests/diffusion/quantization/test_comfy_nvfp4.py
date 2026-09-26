# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod

from vllm_omni.diffusion.models.flux2.nvfp4_checkpoint import (
    map_bfl_name,
    map_bfl_weight,
    quantized_layer_names,
)
from vllm_omni.quantization.comfy_nvfp4_config import ComfyNvfp4Config, ComfyNvfp4LinearMethod
from vllm_omni.quantization.factory import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("double_blocks.0.img_attn.qkv.weight_scale", "transformer_blocks.0.attn.to_qkv.weight_scale"),
        ("double_blocks.7.txt_attn.norm.query_norm.scale", "transformer_blocks.7.attn.norm_added_q.weight"),
        ("single_blocks.47.linear1.input_scale", "single_transformer_blocks.47.attn.to_qkv_mlp_proj.input_scale"),
        ("single_blocks.2.linear2.weight", "single_transformer_blocks.2.attn.to_out.weight"),
        ("img_in.weight", "x_embedder.weight"),
    ],
)
def test_checkpoint_name_mapping(source, target):
    assert map_bfl_name(source) == target


def test_final_modulation_swaps_scale_and_shift():
    source = torch.arange(24).reshape(6, 4)
    name, mapped = map_bfl_weight("final_layer.adaLN_modulation.1.weight", source)
    assert name == "norm_out.linear.weight"
    torch.testing.assert_close(mapped, torch.cat((source[3:], source[:3])))
    name, preserved = map_bfl_weight("double_blocks.0.img_attn.qkv.weight", source)
    assert preserved is source


def test_metadata_keeps_unquantized_text_attention():
    metadata = {"layers": {"double_blocks.0.img_attn.qkv": {"format": "nvfp4"}}}
    config = ComfyNvfp4Config(list(quantized_layer_names(metadata)))
    layer = object.__new__(ReplicatedLinear)
    nn.Module.__init__(layer)
    assert isinstance(config.get_quant_method(layer, "transformer_blocks.0.attn.to_qkv"), ComfyNvfp4LinearMethod)
    assert isinstance(config.get_quant_method(layer, "transformer_blocks.0.attn.add_kv_proj"), UnquantizedLinearMethod)
    assert isinstance(
        config.get_quant_method(layer, "transformer.transformer_blocks.0.attn.to_qkv"), ComfyNvfp4LinearMethod
    )


def test_non_nvfp4_metadata_is_rejected():
    with pytest.raises(ValueError, match="uniform NVFP4"):
        list(quantized_layer_names({"layers": {"img_in": {"format": "float8"}}}))


def test_serialized_layout_loads_packed_weights_and_scales():
    layer = nn.Module()
    method = ComfyNvfp4LinearMethod()
    method.create_weights(layer, 32, [16, 16, 16], 32, 48, torch.bfloat16)
    packed = torch.arange(48 * 16).reshape(48, 16).to(torch.uint8)
    scales = torch.ones(48, 2, dtype=torch.float8_e4m3fn)
    for parameter, value in (
        (layer.weight, packed),
        (layer.weight_scale, scales),
        (layer.weight_scale_2, torch.tensor(0.125)),
        (layer.input_scale, torch.tensor(0.5)),
    ):
        parameter.weight_loader(parameter, value)
    assert torch.equal(layer.weight, packed)
    assert torch.equal(layer.weight_scale.view(torch.uint8), scales.view(torch.uint8))
    assert layer.weight_scale_2.item() == 0.125
    assert layer.input_scale.item() == 0.5


def test_tensor_parallel_is_rejected_before_allocating():
    layer = nn.Module()
    with pytest.raises(ValueError, match="parallel size 1"):
        ComfyNvfp4LinearMethod().create_weights(layer, 16, [16], 32, 32, torch.bfloat16)
    assert not list(layer.parameters())


def test_factory_routes_explicit_checkpoint_layers():
    config = build_quant_config({"method": "comfy_nvfp4", "quantized_layers": ["transformer_blocks.0.attn.to_qkv"]})
    assert isinstance(config, ComfyNvfp4Config)


def test_prepare_tool_round_trips_serialized_storage(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer/config.json").write_text("{}")
    (base / "model_index.json").write_text('{"_class_name":"Flux2Pipeline"}')
    layer = "double_blocks.0.img_attn.qkv"
    tensors = {
        layer + ".weight": torch.arange(256).reshape(16, 16).to(torch.uint8),
        layer + ".weight_scale": torch.ones(16, 2).to(torch.float8_e4m3fn),
        layer + ".weight_scale_2": torch.tensor(0.125),
        layer + ".input_scale": torch.tensor(0.25),
        "final_layer.adaLN_modulation.1.weight": torch.arange(32).reshape(8, 4).to(torch.bfloat16),
    }
    checkpoint = tmp_path / "source.safetensors"
    save_file(
        tensors,
        checkpoint,
        metadata={
            "_quantization_metadata": json.dumps({"layers": {layer: {"format": "nvfp4"}}}),
        },
    )
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        "sys.argv", ["prepare", "--base-model", str(base), "--checkpoint", str(checkpoint), "--output", str(output)]
    )
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "tools/prepare_flux2_nvfp4.py"), run_name="__main__")
    result = load_file(output / "transformer/diffusion_pytorch_model.safetensors")
    for name, tensor in tensors.items():
        target, expected = map_bfl_weight(name, tensor)
        assert result[target].dtype == expected.dtype
        assert torch.equal(result[target].reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    config = json.loads((output / "transformer/quantization_config.json").read_text())
    assert config["quantized_layers"] == ["transformer_blocks.0.attn.to_qkv"]
    assert (output / "model_index.json").resolve() == (base / "model_index.json")


@pytest.mark.parametrize("shape", [(128,), (17, 128), (2, 17, 128), (2, 3, 17, 128)])
def test_apply_preserves_leading_dimensions(shape):
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.registry import registry
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

    generator = torch.Generator().manual_seed(42)
    layer = nn.Module()
    method = ComfyNvfp4LinearMethod()
    method.create_weights(layer, 128, [128], 128, 128, torch.bfloat16)
    with registry.use_backend("eager"):
        weight = QuantizedTensor.from_float(
            torch.randn(128, 128, generator=generator, dtype=torch.bfloat16), "TensorCoreNVFP4Layout"
        )
        packed, scale, block_scale = TensorCoreNVFP4Layout.get_plain_tensors(weight)
        for parameter, value in (
            (layer.weight, packed),
            (layer.weight_scale, block_scale),
            (layer.weight_scale_2, scale),
            (layer.input_scale, torch.tensor(0.01)),
        ):
            parameter.weight_loader(parameter, value)
        method.process_weights_after_loading(layer)
        assert type(layer.weight) is nn.Parameter
        assert layer.weight.dtype == torch.uint8
        x = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
        bias = torch.randn(128, generator=generator, dtype=torch.bfloat16)
        quantized = QuantizedTensor.from_float(x.reshape(-1, 128), "TensorCoreNVFP4Layout", scale=layer.input_scale)
        expected = torch.nn.functional.linear(quantized, weight, bias).reshape(*shape[:-1], 128)
        torch.testing.assert_close(method.apply(layer, x, bias), expected, rtol=0, atol=0)


def test_finalization_preserves_packed_offload_storage():
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    from vllm_omni.diffusion.offloader.tensor_utils import restore_tensor_storage

    layer = nn.Module()
    method = ComfyNvfp4LinearMethod()
    method.create_weights(layer, 128, [128], 128, 128, torch.bfloat16)
    layer.weight.data.fill_(0x12)
    layer.weight_scale.data.fill_(1)
    layer.weight_scale_2.data.fill_(0.125)
    layer.input_scale.data.fill_(0.25)
    method.process_weights_after_loading(layer)
    parameters = dict(layer.named_parameters())
    originals = {name: tensor.detach().clone() for name, tensor in parameters.items()}
    storage, metadata = LayerwiseOffloadHook._to_cpu(parameters, {}, pin_memory=False)
    assert all(parameter.numel() == 0 for parameter in parameters.values())
    for dtype, entries in metadata.items():
        for entry in entries:
            value = torch.as_strided(
                storage[dtype][entry["offset"] : entry["offset"] + entry["numel"]],
                entry["shape"],
                entry["stride"],
            )
            restore_tensor_storage(parameters[entry["name"]], value, device="cpu")
    for name, parameter in parameters.items():
        assert parameter.dtype == originals[name].dtype
        assert torch.equal(parameter.reshape(-1).view(torch.uint8), originals[name].reshape(-1).view(torch.uint8))
