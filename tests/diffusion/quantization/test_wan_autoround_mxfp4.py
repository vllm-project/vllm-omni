# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan AutoRound-format MXFP4 configuration and layer-mapping tests."""

from unittest.mock import patch

import pytest
from torch import nn
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.model_loader.utils import configure_quant_config

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    WanTransformer3DModel,
)
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.inc_config import OmniINCConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _autoround_mxfp4_config() -> dict:
    return {
        "bits": 4,
        "act_bits": 4,
        "data_type": "mx_fp",
        "act_data_type": "mx_fp",
        "group_size": 32,
        "act_group_size": 32,
        "sym": True,
        "act_sym": True,
        "act_dynamic": True,
        "enable_quanted_input": False,
        "block_name_to_quantize": "blocks",
        "quant_method": "auto-round",
        "packing_format": "auto_round:llm_compressor",
        "autoround_version": "0.14.2",
    }


def test_build_wan_autoround_mxfp4_config():
    config = build_quant_config(_autoround_mxfp4_config())

    assert isinstance(config, OmniINCConfig)
    assert config.get_name() == "inc"
    assert config.weight_bits == 4
    assert config.data_type == "mx_fp"
    assert config.group_size == 32
    assert config.packing_format == "auto_round:llm_compressor"
    assert config.block_name_to_quantize == ["blocks"]


def test_wan_autoround_config_uses_runtime_layer_names():
    config = build_quant_config(_autoround_mxfp4_config())
    configure_quant_config(config, WanTransformer3DModel)

    assert config.packed_modules_mapping is WanTransformer3DModel.packed_modules_mapping
    assert config.block_name_to_quantize == ["blocks"]

    layer = object.__new__(LinearBase)
    nn.Module.__init__(layer)
    for prefix in (
        "blocks.0.ffn.net_0.proj",
        "blocks.0.ffn.net_2",
        "blocks.0.attn1.to_out",
    ):
        layer_config = config.config_parser.resolve(layer, prefix)
        assert layer_config.quantized
        assert layer_config.is_mxfp4


def test_wan_autoround_scope_dispatches_only_blocks_to_mxfp4():
    config = build_quant_config(_autoround_mxfp4_config())
    configure_quant_config(config, WanTransformer3DModel)

    layer = object.__new__(LinearBase)
    nn.Module.__init__(layer)
    layer_config = config.config_parser.resolve(layer, "blocks.0.ffn.net_0.proj")
    outside_config = config.config_parser.resolve(layer, "condition_embedder.time_proj")

    assert layer_config.quantized
    assert layer_config.is_mxfp4
    assert not outside_config.quantized

    fake_method = object()
    with patch(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_scheme.INCMxfp4Scheme.get_linear_method",
        return_value=fake_method,
    ):
        method = config.get_quant_method(layer, "blocks.0.ffn.net_0.proj")
    assert method is fake_method
