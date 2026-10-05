# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig

from vllm_omni.quantization import build_quantization_config
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.torchao_config import OmniTorchAOConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "transformer_spec",
    [
        {
            "method": "torchao",
            "quant_type": {
                "default": {
                    "_type": "Float8WeightOnlyConfig",
                    "_version": 2,
                    "_data": {
                        "weight_dtype": {
                            "_type": "torch.dtype",
                            "_data": "float8_e4m3fn",
                        },
                        "set_inductor_config": False,
                    },
                }
            },
        },
        {"method": "torchao_float8_weight_only"},
    ],
    ids=["serialized-json", "checkpoint-shorthand"],
)
def test_build_quant_config_torchao_checkpoint(transformer_spec):
    quantization = pytest.importorskip("torchao.quantization")
    result = build_quantization_config({"transformer": transformer_spec})

    assert isinstance(result, ComponentQuantizationConfig)
    transformer = result.resolve("transformer")
    assert transformer.get_name() == "torchao"
    assert transformer.is_checkpoint_torchao_serialized is True
    assert isinstance(transformer.torchao_config, quantization.Float8WeightOnlyConfig)
    assert transformer.torchao_config.version == 2
    assert transformer.torchao_config.weight_dtype is torch.float8_e4m3fn
    assert transformer.torchao_config.set_inductor_config is False


def test_build_quant_config_torchao_runtime():
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig

    torchao_config = object()
    result = build_quantization_config({"method": "torchao", "torchao_config": torchao_config})

    assert isinstance(result, TorchAOConfig)
    assert result.torchao_config is torchao_config
    assert result.is_checkpoint_torchao_serialized is False


@pytest.mark.parametrize("config_cls", [TorchAOConfig, OmniTorchAOConfig])
def test_torchao_from_config_without_quant_method_is_not_checkpoint_serialized(config_cls):
    """Ensure from_config without quant_method is not marked checkpoint-serialized."""
    pytest.importorskip("torchao.quantization")
    config = {"quant_type": {"default": {"_type": "Float8WeightOnlyConfig", "_version": 2, "_data": {}}}}

    assert config_cls.from_config(config).is_checkpoint_torchao_serialized is False
