# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import UMT5Config, UMT5EncoderModel
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.helios import quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def encoder():
    config = UMT5Config(
        vocab_size=32,
        d_model=128,
        d_ff=256,
        num_layers=2,
        num_heads=2,
        d_kv=64,
        feed_forward_proj="gated-gelu",
    )
    return UMT5EncoderModel(config).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_encoder_requires_component_opt_in(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_helios_text_encoder_fp8(encoder, config, torch.device("cpu")) == 0
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(activation_scheme="static"), Fp8Config(is_checkpoint_fp8_serialized=True)]
)
def test_encoder_rejects_unsupported_fp8_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_helios_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": config}), torch.device("cpu")
        )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


def test_text_projection_conversion_preserves_other_weights_and_hidden_states(encoder, monkeypatch):
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = ids.ne(0)
    original = dict(encoder.named_parameters())
    with torch.inference_mode():
        reference = encoder(ids, attention_mask=mask, output_hidden_states=True).hidden_states
    prefixes = []

    class LoadedLinear(nn.Linear):
        def __init__(
            self, in_features, out_features, *, bias, params_dtype, quant_config, prefix, return_bias, disable_tp
        ):
            assert disable_tp and not return_bias
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
            self.quant_method = object()
            prefixes.append(prefix)
            self.weight.weight_loader = lambda parameter, weight: parameter.copy_(weight)

    monkeypatch.setattr(quantization, "ReplicatedLinear", LoadedLinear)
    assert (
        quantization.prepare_helios_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 4
    )
    assert all(prefix.startswith("text_encoder.encoder.block.") for prefix in prefixes)
    for name, parameter in encoder.named_parameters():
        if not name.endswith(("wi_0.weight", "wi_1.weight")):
            assert parameter is original[name]
    with torch.inference_mode():
        actual = encoder(ids, attention_mask=mask, output_hidden_states=True).hidden_states
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_encoder_rejects_float32_weights(encoder):
    encoder.float()
    with pytest.raises(ValueError, match="BF16 or FP16"):
        quantization.prepare_helios_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )


def test_ignored_projections_keep_original_parameters(encoder, monkeypatch):
    original = dict(encoder.named_parameters())

    class IgnoredLinear(nn.Module):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()
            self.quant_method = quantization.UnquantizedLinearMethod()

    monkeypatch.setattr(quantization, "ReplicatedLinear", IgnoredLinear)
    assert (
        quantization.prepare_helios_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 0
    )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())
