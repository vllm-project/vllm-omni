# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration and HF module-boundary contracts; no FP8 kernel claims."""

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.omnigen2 import mllm_quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def encoder(nested=False):
    model = nn.Module()
    model.model = nn.Module()
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8, bias=False))])
    if nested:
        model.model.language_model = decoder
    else:
        model.model.layers = decoder.layers
    model.visual = nn.Linear(8, 8)
    model.lm_head = nn.Linear(8, 8)
    return model.eval(), decoder.layers


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_legacy_and_transformer_only_configs_leave_encoder_unchanged(config):
    model, _ = encoder()
    parameters = dict(model.named_parameters())
    assert mllm_quantization.prepare_mllm_fp8(model, config) == 0
    assert all(dict(model.named_parameters())[name] is param for name, param in parameters.items())


@pytest.mark.parametrize(
    "config",
    [Fp8Config(is_checkpoint_fp8_serialized=True), Fp8Config(activation_scheme="static")],
)
def test_unsupported_encoder_fp8_modes_fail_before_mutation(config):
    model, layers = encoder()
    original = layers[0][0]
    with pytest.raises(ValueError, match="dynamic online FP8"):
        mllm_quantization.prepare_mllm_fp8(model, ComponentQuantizationConfig({"mllm": config}))
    assert layers[0][0] is original


def test_missing_language_decoder_is_rejected():
    model = nn.Module()
    model.model = nn.Module()
    with pytest.raises(ValueError, match="language decoder"):
        mllm_quantization.prepare_mllm_fp8(model, ComponentQuantizationConfig({"mllm": Fp8Config()}))


@pytest.mark.parametrize("nested", [False, True], ids=["transformers_v4", "transformers_v5"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_only_decoder_linears_use_parameter_loaders(monkeypatch, nested, dtype):
    model, layers = encoder(nested)
    model.to(dtype=dtype)
    original_weights = {name: value.detach().clone() for name, value in layers.named_parameters()}
    visual, head = model.visual, model.lm_head
    loaded = []
    prefixes = []
    original_default = torch.get_default_dtype()
    gemm_dtype = torch.bfloat16 if dtype == torch.float32 else dtype

    class LoaderLinear(nn.Linear):
        # Model the ReplicatedLinear parameter-loader contract on CPU. Native
        # backend materialization and FP8 arithmetic are covered by GPU E2E.
        def __init__(
            self, in_features, out_features, *, bias, params_dtype, quant_config, prefix, return_bias, disable_tp
        ):
            assert not return_bias and disable_tp
            assert torch.get_default_dtype() == params_dtype == gemm_dtype
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
            self.quant_method = object()
            prefixes.append(prefix)

            def weight_loader(parameter, value):
                loaded.append(prefix)
                parameter.copy_(value)

            self.weight.weight_loader = weight_loader
            if self.bias is not None:
                self.bias.weight_loader = weight_loader

    monkeypatch.setattr(mllm_quantization, "_MllmFp8Linear", LoaderLinear)
    assert mllm_quantization.prepare_mllm_fp8(model, ComponentQuantizationConfig({"mllm": Fp8Config()})) == 2
    assert len(loaded) == 3
    prefix = "mllm.model.language_model.layers" if nested else "mllm.model.layers"
    assert prefixes == [f"{prefix}.0.0", f"{prefix}.0.2"]
    assert model.visual is visual and model.lm_head is head
    assert not layers[0][0].training and not layers[0][2].training
    assert torch.get_default_dtype() == original_default
    for name, parameter in layers.named_parameters():
        torch.testing.assert_close(parameter, original_weights[name].to(gemm_dtype), atol=0, rtol=0)


def test_backend_excluded_linears_keep_original_parameters(monkeypatch):
    model, layers = encoder()
    originals = dict(layers.named_parameters())

    class ExcludedLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.quant_method = UnquantizedLinearMethod()

    monkeypatch.setattr(mllm_quantization, "_MllmFp8Linear", ExcludedLinear)
    assert mllm_quantization.prepare_mllm_fp8(model, ComponentQuantizationConfig({"mllm": Fp8Config()})) == 0
    assert all(dict(layers.named_parameters())[name] is param for name, param in originals.items())
