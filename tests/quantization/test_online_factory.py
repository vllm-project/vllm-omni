# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""build_quant_config builds vLLM's online quantization from a method name or a dict with an ignore list."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_omni.quantization.factory import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _online_classes():
    from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

    return OnlineQuantizationConfig


@pytest.mark.parametrize("method", ["fp8_per_channel", "fp8_per_tensor", "fp8_per_block"])
def test_online_shorthand_string_builds_online_config(method: str) -> None:
    OnlineQuantizationConfig = _online_classes()
    config = build_quant_config(method)
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear is not None and config.ignored_layers == []
    assert config.args.moe == config.args.linear


@pytest.mark.parametrize("method", ["int8_per_channel_weight_only", "nvfp4_per_token"])
def test_online_moe_only_shorthand(method: str) -> None:
    OnlineQuantizationConfig = _online_classes()
    config = build_quant_config({"method": method, "ignore": ["*gate*"]})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear is None
    assert config.args.moe is not None
    assert config.ignored_layers == ["*gate*"]


def test_online_shorthand_with_linear_override_preserves_moe() -> None:
    from vllm.config.quantization import _ONLINE_SHORTHANDS

    OnlineQuantizationConfig = _online_classes()
    config = build_quant_config({"method": "fp8_per_channel", "linear": "fp8_per_block"})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear == _ONLINE_SHORTHANDS["fp8_per_block"].linear
    assert config.args.moe == _ONLINE_SHORTHANDS["fp8_per_channel"].moe


def test_online_dict_with_ignore_patterns() -> None:
    OnlineQuantizationConfig = _online_classes()
    ignore = ["*patch_proj*", "final_layer.*", "blocks.*.adaln_proj"]
    config = build_quant_config({"method": "fp8_per_channel", "ignore": ignore})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.ignored_layers == ignore
    explicit = build_quant_config({"method": "online", "linear": "fp8_per_block", "ignore": ignore})
    assert isinstance(explicit, OnlineQuantizationConfig) and explicit.ignored_layers == ignore


def test_online_ignore_patterns_skip_layer_quantization() -> None:
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    OnlineQuantizationConfig = _online_classes()
    config = build_quant_config({"method": "fp8_per_channel", "ignore": ["*final_layer*"]})
    assert isinstance(config, OnlineQuantizationConfig)
    layer = LinearBase(16, 16, quant_config=config, prefix="transformer.final_layer.proj", disable_tp=True)
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert config.resolve_quant_method_cls(type(layer), "transformer.blocks.0.attn.to_q") is not None


def test_online_targets() -> None:
    OnlineQuantizationConfig = _online_classes()
    targets = {"blocks.*.attn.*": "fp8_per_channel"}
    config = build_quant_config({"method": "online", "targets": targets})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.targets == targets
    assert config.args.linear is None and config.args.moe is None


def test_online_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="activation_scheme"):
        build_quant_config({"method": "online", "linear": "fp8_per_channel", "activation_scheme": "dynamic"})


def test_per_component_online_and_unquantized_encoder() -> None:
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig

    config = build_quant_config(
        {"transformer": {"method": "fp8_per_channel", "ignore": ["*final_layer*"]}, "text_encoder": None}
    )
    assert isinstance(config, ComponentQuantizationConfig)
    transformer = config.resolve("transformer")
    assert isinstance(transformer, _online_classes())
    assert transformer.ignored_layers == ["*final_layer*"]
    assert transformer.args.moe is not None
    assert config.resolve("text_encoder") is None
