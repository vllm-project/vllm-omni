# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for SDXL quantization wiring and fused checkpoint loading."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.sdxl import sdxl_unet
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def single_rank_linears(monkeypatch):
    """Use real vLLM parameters/loaders without a distributed process group."""
    import vllm.model_executor.parameter as parameter

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(sdxl_unet, "Attention", lambda **kwargs: nn.Identity())


@pytest.mark.parametrize("mode", ["none", "global", "unet", "other_component"])
def test_unet_quant_config_and_checkpoint_prefixes(mode):
    quant_config = Mock(spec=QuantizationConfig)
    quant_config.packed_modules_mapping = {}
    quant_config.get_quant_method.side_effect = lambda layer, prefix: linear.UnquantizedLinearMethod()
    if mode == "none":
        config = None
    elif mode == "global":
        config = quant_config
    else:
        component = "unet" if mode == "unet" else "transformer"
        config = ComponentQuantizationConfig({component: quant_config})
    # Bypass unrelated config initialization; no model storage is allocated.
    od_config = object.__new__(OmniDiffusionConfig)
    od_config.quantization_config = config
    with torch.device("meta"):
        model = sdxl_unet.SDXLUNet2DConditionModel(od_config=od_config)

    expected_config = quant_config if mode in ("global", "unet") else None
    linears = {name: layer for name, layer in model.named_modules() if isinstance(layer, linear.LinearBase)}
    assert len(linears) == 599
    for name, layer in linears.items():
        expected_prefix = name.replace(".ff.geglu.", ".ff.net.0.").replace(".ff.out_proj", ".ff.net.2")
        assert layer.prefix == expected_prefix
        assert layer.quant_config is expected_config
    assert quant_config.get_quant_method.call_count == (599 if expected_config is not None else 0)
    assert isinstance(model.time_embedding.linear_1, nn.Linear)
    assert isinstance(model.add_embedding.linear_1, nn.Linear)


def test_unquantized_replicated_projections_forward():
    resnet = sdxl_unet.SDXLResnetBlock2D(32, 32, 32)
    transformer = sdxl_unet.SDXLTransformer2DModel(1, 32, 32, 0, 32)
    # vLLM allocates checkpoint parameters with empty(), unlike nn.Linear.
    for model in (resnet, transformer):
        for layer in model.modules():
            if isinstance(layer, linear.LinearBase):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    hidden = torch.randn(1, 32, 2, 2)
    temb = torch.randn(1, 32)
    # Compare against the original nn.Linear behavior, including biases.
    reference = resnet.conv1(resnet.nonlinearity(resnet.norm1(hidden)))
    projected_temb = nn.functional.linear(
        resnet.nonlinearity(temb), resnet.time_emb_proj.weight, resnet.time_emb_proj.bias
    )
    reference = reference + projected_temb[:, :, None, None]
    reference = resnet.conv2(resnet.nonlinearity(resnet.norm2(reference))) + hidden
    torch.testing.assert_close(resnet(hidden, temb), reference)

    reference = transformer.norm(hidden).permute(0, 2, 3, 1).reshape(1, 4, 32)
    for projection in (transformer.proj_in, transformer.proj_out):
        reference = nn.functional.linear(reference, projection.weight, projection.bias)
    reference = reference.reshape(1, 2, 2, 32).permute(0, 3, 1, 2) + hidden
    torch.testing.assert_close(transformer(hidden, torch.empty(1, 1, 32)), reference)


@pytest.mark.parametrize("tensor_name", ["weight", "weight_scale"])
def test_fused_qkv_and_geglu_loading(tensor_name):
    from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter

    # Real vLLM fused loaders, tiny CPU tensors, no quantization kernel.
    block = sdxl_unet.SDXLBasicTransformerBlock(32, 1, 32, 32, prefix="block")
    model = object.__new__(sdxl_unet.SDXLUNet2DConditionModel)
    nn.Module.__init__(model)
    model.block = block
    if tensor_name == "weight_scale":
        for layer in (block.attn1.to_qkv, block.ff.geglu.proj, block.ff.out_proj):
            layer.register_parameter(
                "weight_scale",
                GroupQuantScaleParameter(
                    data=torch.zeros(layer.output_size, layer.input_size // 32, dtype=torch.uint8),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=layer.weight_loader_v2,
                ),
            )
    else:
        for layer in (block.attn1.to_qkv, block.ff.geglu.proj):
            layer.register_parameter(
                "weight",
                ModelWeightParameter(
                    data=torch.zeros(layer.output_size, layer.input_size),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=layer.weight_loader_v2,
                ),
            )
    qkv_param = getattr(block.attn1.to_qkv, tensor_name)
    geglu_param = getattr(block.ff.geglu.proj, tensor_name)
    out_param = getattr(block.ff.out_proj, tensor_name)
    qkv = torch.arange(qkv_param.numel()).reshape(qkv_param.shape).to(qkv_param.dtype)
    geglu = torch.arange(geglu_param.numel()).reshape(geglu_param.shape).to(geglu_param.dtype)
    out = torch.ones_like(out_param)
    bias = torch.arange(block.ff.geglu.proj.bias.numel(), dtype=torch.float32)
    weights = [(f"block.attn1.to_{shard}.{tensor_name}", value) for shard, value in zip("qkv", qkv.chunk(3))]
    weights.extend(
        [
            (f"block.ff.net.0.proj.{tensor_name}", geglu),
            ("block.ff.net.0.proj.bias", bias),
            (f"block.ff.net.2.{tensor_name}", out),
        ]
    )
    loaded = model.load_weights(weights)
    assert loaded == {
        f"block.attn1.to_qkv.{tensor_name}",
        f"block.ff.geglu.proj.{tensor_name}",
        "block.ff.geglu.proj.bias",
        f"block.ff.out_proj.{tensor_name}",
    }
    torch.testing.assert_close(qkv_param, qkv)
    torch.testing.assert_close(geglu_param, geglu)
    torch.testing.assert_close(block.ff.geglu.proj.bias, bias)
    torch.testing.assert_close(out_param, out)
