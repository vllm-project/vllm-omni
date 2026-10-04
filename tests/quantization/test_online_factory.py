# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""build_quant_config builds vLLM's online quantization from a method name or a dict with an ignore list."""

from __future__ import annotations

import pytest

from vllm_omni.quantization.factory import build_quant_config


def _online_classes():
    from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

    return OnlineQuantizationConfig


def test_online_shorthand_string_builds_online_config() -> None:
    OnlineQuantizationConfig = _online_classes()
    config = build_quant_config("fp8_per_channel")
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear is not None and config.ignored_layers == []


def test_online_dict_with_ignore_patterns() -> None:
    OnlineQuantizationConfig = _online_classes()
    ignore = ["*patch_proj*", "final_layer.*", "blocks.*.adaln_proj"]
    config = build_quant_config({"method": "fp8_per_channel", "ignore": ignore})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.ignored_layers == ignore
    explicit = build_quant_config({"method": "online", "linear": "fp8_per_block", "ignore": ignore})
    assert isinstance(explicit, OnlineQuantizationConfig) and explicit.ignored_layers == ignore


def test_online_rejects_unknown_keys() -> None:
    with pytest.raises(TypeError):
        build_quant_config({"method": "online", "linear": "fp8_per_channel", "activation_scheme": "dynamic"})


def test_per_component_online_and_unquantized_encoder() -> None:
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig

    config = build_quant_config(
        {"transformer": {"method": "fp8_per_channel", "ignore": ["*final_layer*"]}, "text_encoder": None}
    )
    assert isinstance(config, ComponentQuantizationConfig)
