# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, UMT5Config, UMT5EncoderModel
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.t5_encoder import quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[(T5Config, T5EncoderModel), (UMT5Config, UMT5EncoderModel)])
def encoder(request):
    torch.manual_seed(42)
    config_cls, model_cls = request.param
    return model_cls(
        config_cls(
            vocab_size=32,
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=2,
            num_heads=2,
            feed_forward_proj="gated-gelu",
            dropout_rate=0,
        )
    ).eval()


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({"transformer": Fp8Config()})])
def test_requires_explicit_component(encoder, config):
    original = dict(encoder.named_parameters())
    assert quantization.prepare_t5_fp8(encoder, config, "text_encoder") == 0
    assert all(dict(encoder.named_parameters())[name] is p for name, p in original.items())


@pytest.mark.parametrize(
    "config", [Fp8Config(is_checkpoint_fp8_serialized=True), Fp8Config(activation_scheme="static")]
)
def test_rejects_unsupported_modes_before_mutation(encoder, config):
    original = dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_t5_fp8(encoder, ComponentQuantizationConfig({"text_encoder": config}), "text_encoder")
    assert all(dict(encoder.named_parameters())[name] is p for name, p in original.items())


@pytest.mark.parametrize("component", ["text_encoder", "text_encoder_2", "text_encoder_3"])
@pytest.mark.parametrize("ignore_first_q", [False, True])
def test_preserves_wo_and_non_linear_weights_and_forward(encoder, monkeypatch, component, ignore_first_q):
    original = dict(encoder.named_parameters())
    ids = torch.tensor([[1, 2, 3, 0]])
    with torch.no_grad():
        reference = encoder(ids, attention_mask=ids.ne(0)).last_hidden_state
    loaded = []
    prefixes = []

    class LoaderLinear(nn.Linear):
        def __init__(
            self, in_features, out_features, *, bias, params_dtype, quant_config, prefix, return_bias, disable_tp
        ):
            assert disable_tp and not return_bias
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
            if ignore_first_q and prefix.endswith(".0.layer.0.SelfAttention.q"):
                self.quant_method = quantization.UnquantizedLinearMethod()
            else:
                self.quant_method = object()
                prefixes.append(prefix)

            def load(parameter, value):
                parameter.copy_(value)
                loaded.append(value)

            self.weight.weight_loader = load

        def forward(self, x):
            return super().forward(x.to(self.weight.dtype)).to(x.dtype)

    monkeypatch.setattr(quantization, "_T5Fp8Linear", LoaderLinear)
    expected = 11 if ignore_first_q else 12
    assert (
        quantization.prepare_t5_fp8(encoder, ComponentQuantizationConfig({component: Fp8Config()}), component)
        == expected
    )
    assert len(loaded) == expected
    assert all(name.startswith(component + ".encoder.block.") for name in prefixes)
    for name, p in encoder.named_parameters():
        if name.endswith(".wo.weight") or not any(
            name == prefix.removeprefix(component + ".") + ".weight" for prefix in prefixes
        ):
            assert p is original[name]
    with torch.no_grad():
        actual = encoder(ids, attention_mask=ids.ne(0)).last_hidden_state
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.03)
