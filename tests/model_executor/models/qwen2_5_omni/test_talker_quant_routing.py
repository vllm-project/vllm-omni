# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Quantization routing in the Qwen2.5-Omni talker constructor.

The inner language model is replaced with a fake that records the quant
config it would be built with after ``with_hf_config``.
"""

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm_omni.model_executor.models.qwen2_5_omni import qwen2_5_omni_talker as talker_mod
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _quant(name: str) -> MagicMock:
    config = MagicMock()
    config.get_name.return_value = name
    return config


@dataclass
class _VllmConfig:
    model_config: Any
    quant_config: Any = None

    def __post_init__(self):
        # Mirrors VllmConfig: a missing quant_config is rebuilt from
        # model_config.quantization on construction and on every replace().
        if self.quant_config is None and self.model_config.quantization is not None:
            self.quant_config = _quant(self.model_config.quantization)


def _build(monkeypatch: pytest.MonkeyPatch, quant_config: Any, quantization: str | None):
    captured: dict[str, Any] = {}

    def fake_language_model(*, vllm_config, **kwargs):
        # init_vllm_registered_model -> with_hf_config -> replace(model_config=copy)
        captured["language_model"] = replace(vllm_config, model_config=copy.copy(vllm_config.model_config)).quant_config
        return MagicMock()

    monkeypatch.setattr(talker_mod, "init_vllm_registered_model", fake_language_model)
    hf_config = SimpleNamespace(embedding_size=8, hidden_size=4)
    model_config = SimpleNamespace(hf_config=hf_config, quantization=quantization)
    vllm_config = _VllmConfig(model_config, quant_config)
    model = talker_mod.Qwen2_5OmniTalkerForConditionalGeneration(vllm_config=vllm_config, prefix="talker")
    return model, captured


def test_dynamic_method_does_not_reach_the_talker_language_model(monkeypatch):
    model, captured = _build(monkeypatch, _quant("fp8"), quantization="fp8")

    assert model.quant_config is None
    assert captured["language_model"] is None


def test_component_config_that_excludes_the_talker_keeps_it_unquantized(monkeypatch):
    router = ComponentQuantizationConfig({"thinker.language_model": _quant("fp8")}, default_config=None)
    _, captured = _build(monkeypatch, router, quantization="fp8")

    assert captured["language_model"] is None


def test_component_config_can_target_the_talker(monkeypatch):
    fp8 = _quant("fp8")
    router = ComponentQuantizationConfig({"talker": fp8}, default_config=None)
    model, captured = _build(monkeypatch, router, quantization="fp8")

    assert model.quant_config is fp8
    assert captured["language_model"] is fp8


def test_pre_quantized_method_stays_on_the_talker(monkeypatch):
    modelopt = _quant("modelopt")
    _, captured = _build(monkeypatch, modelopt, quantization="modelopt")

    assert captured["language_model"] is modelopt
