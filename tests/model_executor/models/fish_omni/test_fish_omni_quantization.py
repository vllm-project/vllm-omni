# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 ``bsgemm-moe`` quantization config."""

from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import fp8, get_quantization_config

from vllm_omni.model_executor.models.fish_omni.quantization import QUANTIZATION_METHOD, ExpertFp8Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHECKPOINT_QUANT_CONFIG = {
    "quant_method": "bsgemm-moe",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}


def test_registered_with_vllm():
    assert get_quantization_config(QUANTIZATION_METHOD) is ExpertFp8Config


def test_from_checkpoint_config():
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    assert config.get_name() == QUANTIZATION_METHOD
    assert config.is_checkpoint_fp8_serialized
    assert config.activation_scheme == "dynamic"
    assert config.weight_block_size == [128, 128]


@pytest.mark.parametrize(
    "override",
    [
        {"activation_scheme": "static"},
        {"weight_block_size": None},
        {"weight_block_size": [128]},
        {"weight_block_size": [0, 128]},
    ],
)
def test_rejects_unsupported_checkpoint_config(override):
    with pytest.raises(ValueError, match=QUANTIZATION_METHOD):
        ExpertFp8Config.from_config({**CHECKPOINT_QUANT_CONFIG, **override})


def test_linear_layers_stay_unquantized():
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    method = config.get_quant_method(MagicMock(spec=LinearBase), prefix="model.layers.0.self_attn.qkv_proj")
    assert isinstance(method, UnquantizedLinearMethod)


def test_routed_experts_use_fp8_moe(monkeypatch):
    sentinel = object()
    created = []

    def fake_moe_method(quant_config, layer):
        created.append((quant_config, layer))
        return sentinel

    monkeypatch.setattr(fp8, "Fp8MoEMethod", fake_moe_method)
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    experts = MagicMock(spec=fp8.RoutedExperts)
    assert config.get_quant_method(experts, prefix="model.layers.0.mlp.experts") is sentinel
    assert created == [(config, experts)]
