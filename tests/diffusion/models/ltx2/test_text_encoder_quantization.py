# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.ltx2 import ltx2_components, quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("scope", ["encoder", "transformer", "both", "global", "none"])
def test_transformer_receives_only_its_component_config(monkeypatch, scope):
    fp8 = Fp8Config()
    configs = {
        "encoder": ComponentQuantizationConfig({"text_encoder": fp8}),
        "transformer": ComponentQuantizationConfig({"transformer": fp8}),
        "both": ComponentQuantizationConfig({"text_encoder": fp8, "transformer": fp8}),
        "global": fp8,
        "none": None,
    }
    received = []

    class Transformer(nn.Module):
        def __init__(self, num_layers=1, quant_config=None):
            super().__init__()
            received.append((num_layers, quant_config))

    monkeypatch.setattr(ltx2_components, "LTX2VideoTransformer3DModel", Transformer)
    ltx2_components.create_transformer_from_config({"num_layers": 2}, configs[scope])
    assert received == [(2, None if scope in ("encoder", "none") else fp8)]


@pytest.fixture
def encoder():
    config = Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=16,
        ),
        vision_config=SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        ),
        image_token_index=31,
    )
    return Gemma3ForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_encoder_requires_component_opt_in(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_gemma3_fp8(encoder, config, torch.device("cpu")) == 0
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(activation_scheme="static"), Fp8Config(is_checkpoint_fp8_serialized=True)]
)
def test_encoder_rejects_unsupported_fp8_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_gemma3_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": config}), torch.device("cpu")
        )
    assert all(dict(encoder.named_parameters())[name] is value for name, value in original.items())


def test_encoder_rejects_other_architectures():
    with pytest.raises(ValueError, match="Gemma3 only"):
        quantization.prepare_gemma3_fp8(
            nn.Linear(2, 2), ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )


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
        quantization.prepare_gemma3_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cpu")
        )
        == 14
    )
    assert all(prefix.startswith("text_encoder.model.language_model.layers.") for prefix in prefixes)
    for name, parameter in encoder.named_parameters():
        if ".language_model.layers." not in name or name.endswith("norm.weight"):
            assert parameter is original[name]
    with torch.inference_mode():
        actual = encoder(ids, attention_mask=mask, output_hidden_states=True).hidden_states
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
