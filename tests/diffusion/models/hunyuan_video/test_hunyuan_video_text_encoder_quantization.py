# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers import Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.hunyuan_video import quantization
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def encoder() -> Qwen2_5_VLTextModel:
    config = Qwen2_5_VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    return Qwen2_5_VLTextModel(config).to(dtype=torch.bfloat16)


@pytest.mark.parametrize("capability", [(8, 0), (8, 6)])
def test_ampere_fp8_rejected_before_changing_encoder(encoder, monkeypatch, capability):
    device = torch.device("cuda:0")

    def get_capability(requested_device):
        assert requested_device == device
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    projections = dict(encoder.layers.named_modules())
    weights = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    config = ComponentQuantizationConfig({"text_encoder": Fp8Config()})
    with pytest.raises(ValueError, match="SM89 or newer"):
        quantization.prepare_hunyuan15_text_encoder_fp8(encoder, config, device)

    assert all(encoder.layers.get_submodule(name) is module for name, module in projections.items())
    for name, parameter in encoder.named_parameters():
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("config", [None, Fp8Config(), ComponentQuantizationConfig({})])
def test_default_and_dit_only_configs_do_not_query_cuda(encoder, monkeypatch, config):
    def unexpected_capability_query(device):
        pytest.fail("BF16 and DiT-only settings must not inspect text encoder FP8 capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability_query)
    assert quantization.prepare_hunyuan15_text_encoder_fp8(encoder, config, torch.device("cuda:0")) == 0


@pytest.mark.parametrize(
    "config",
    [Fp8Config(is_checkpoint_fp8_serialized=True), Fp8Config(activation_scheme="static")],
)
def test_unsupported_encoder_fp8_configs(encoder, config):
    with pytest.raises(ValueError, match="dynamic online FP8"):
        quantization.prepare_hunyuan15_text_encoder_fp8(
            encoder, ComponentQuantizationConfig({"text_encoder": config}), torch.device("cpu")
        )
