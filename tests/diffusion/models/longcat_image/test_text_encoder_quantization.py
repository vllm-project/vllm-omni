# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.longcat_image import text_encoder_quantization as quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def encoder():
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
        },
    )
    return Qwen2_5_VLForConditionalGeneration(config).eval()


@pytest.mark.parametrize(
    "config",
    [
        None,
        Fp8Config(),
        ComponentQuantizationConfig({"transformer": Fp8Config()}),
        ComponentQuantizationConfig({"text_encoder": None}, default_config=Fp8Config()),
    ],
)
def test_encoder_quantization_requires_explicit_opt_in(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_text_encoder_fp8(encoder, config) == 0
    assert all(dict(encoder.named_parameters())[name] is parameter for name, parameter in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(is_checkpoint_fp8_serialized=True), Fp8Config(activation_scheme="static")]
)
def test_unsupported_modes_fail_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_text_encoder_fp8(encoder, ComponentQuantizationConfig({"text_encoder": config}))
    assert all(dict(encoder.named_parameters())[name] is parameter for name, parameter in original.items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_only_language_linears_are_loaded_and_adapted(encoder, monkeypatch, dtype):
    encoder.to(dtype)
    layers = encoder.get_decoder().layers
    linears = {name: layer for name, layer in layers.named_modules() if isinstance(layer, nn.Linear)}
    original = dict(encoder.named_parameters())
    language_parameter_ids = {id(p) for layer in linears.values() for p in layer.parameters()}
    untouched = {name: p for name, p in original.items() if id(p) not in language_parameter_ids}
    loaded = []
    prefixes = []
    gemm_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
    default_dtype = torch.get_default_dtype()

    class LoaderLinear(nn.Linear):
        def __init__(
            self, in_features, out_features, *, bias, params_dtype, quant_config, prefix, return_bias, disable_tp
        ):
            assert params_dtype == torch.get_default_dtype() == gemm_dtype
            assert disable_tp and not return_bias
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
            self.quant_method = object()
            prefixes.append(prefix)

            def load(parameter, value):
                loaded.append(value)
                parameter.copy_(value)

            self.weight.weight_loader = load
            if bias:
                self.bias.weight_loader = load

    monkeypatch.setattr(quantization, "_EncoderFp8Linear", LoaderLinear)
    count = quantization.prepare_text_encoder_fp8(encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}))
    assert count == len(linears) == 7
    assert len(loaded) == sum(len(list(layer.parameters())) for layer in linears.values())
    assert torch.get_default_dtype() == default_dtype
    assert all(dict(encoder.named_parameters())[name] is parameter for name, parameter in untouched.items())
    for name, original_layer in linears.items():
        replacement = layers.get_submodule(name)
        assert not replacement.training
        torch.testing.assert_close(replacement.weight, original_layer.weight.to(gemm_dtype), atol=0, rtol=0)
        if original_layer.bias is not None:
            torch.testing.assert_close(replacement.bias, original_layer.bias.to(gemm_dtype), atol=0, rtol=0)
    assert prefixes == [
        f"text_encoder.{name}" for name, module in encoder.named_modules() if isinstance(module, LoaderLinear)
    ]


def test_ignored_layers_retain_original_parameters(encoder, monkeypatch):
    original = dict(encoder.named_parameters())

    class ExcludedLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.quant_method = UnquantizedLinearMethod()

    monkeypatch.setattr(quantization, "_EncoderFp8Linear", ExcludedLinear)
    assert (
        quantization.prepare_text_encoder_fp8(encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()})) == 0
    )
    assert all(dict(encoder.named_parameters())[name] is parameter for name, parameter in original.items())
