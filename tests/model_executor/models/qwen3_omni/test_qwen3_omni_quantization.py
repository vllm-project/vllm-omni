# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for nested Qwen3-Omni quantization config mapping."""

from dataclasses import dataclass

import pytest

from vllm_omni.model_executor.models.qwen3_omni.quantization import (
    _OUTER_MAPPING_APPLIED,
    apply_nested_quant_config_mapping,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _QuantConfig:
    packed_modules_mapping: dict[str, list[str]]


@dataclass
class _NestedModel:
    quant_config: _QuantConfig | None


@dataclass
class _NestedModelWithPackedModules(_NestedModel):
    packed_modules_mapping: dict[str, list[str]]


def _mapped_quant_config(**packed_modules_mapping):
    quant_config = _QuantConfig(packed_modules_mapping=packed_modules_mapping)
    setattr(quant_config, _OUTER_MAPPING_APPLIED, True)
    return quant_config


def test_nested_mapping_ignores_missing_quant_config():
    apply_nested_quant_config_mapping(_NestedModel(quant_config=None))


def test_nested_mapping_allows_model_without_packed_modules_mapping():
    quant_config = _mapped_quant_config(existing=["module"])

    apply_nested_quant_config_mapping(_NestedModel(quant_config=quant_config))

    assert quant_config.packed_modules_mapping == {"existing": ["module"]}


def test_nested_mapping_merges_packed_modules_mapping():
    quant_config = _mapped_quant_config(existing=["module"])
    model = _NestedModelWithPackedModules(
        quant_config=quant_config,
        packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )

    apply_nested_quant_config_mapping(model)

    assert quant_config.packed_modules_mapping == {
        "existing": ["module"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
