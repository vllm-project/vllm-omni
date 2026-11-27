# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Quantization routing in the Qwen2.5-Omni thinker constructor.

The encoders and language model are replaced with fakes that record the
quant config they receive, so only the routing in ``__init__`` is exercised.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

from vllm_omni.model_executor.models.qwen2_5_omni import qwen2_5_omni_thinker as thinker_mod
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _VllmConfig:
    model_config: Any
    quant_config: Any


def _quant(name: str) -> MagicMock:
    config = MagicMock()
    config.get_name.return_value = name
    return config


def _build(monkeypatch: pytest.MonkeyPatch, quant_config: Any):
    captured: dict[str, Any] = {}

    def fake_visual(**kwargs):
        captured["visual"] = kwargs["quant_config"]
        return MagicMock()

    def fake_language_model(*, vllm_config, **kwargs):
        captured["language_model"] = vllm_config.quant_config
        return MagicMock()

    cls = thinker_mod.Qwen2_5OmniThinkerForConditionalGeneration
    monkeypatch.setattr(thinker_mod, "Qwen2_5OmniAudioEncoder", MagicMock())
    monkeypatch.setattr(thinker_mod, "Qwen2_5_VisionTransformer", fake_visual)
    monkeypatch.setattr(thinker_mod, "init_vllm_registered_model", fake_language_model)
    monkeypatch.setattr(cls, "_mark_tower_model", lambda self, *args, **kwargs: nullcontext())
    monkeypatch.setattr(cls, "_mark_language_model", lambda self, *args, **kwargs: nullcontext())

    hf_config = SimpleNamespace(
        audio_config=SimpleNamespace(),
        vision_config=SimpleNamespace(),
        text_config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda modality: 1),
    )
    model = cls(vllm_config=_VllmConfig(model_config, quant_config), prefix="thinker")
    return model, captured


def test_component_config_is_resolved_once_for_the_vision_encoder(monkeypatch):
    fp8 = _quant("fp8")
    router = ComponentQuantizationConfig({"thinker.visual": fp8, "thinker.language_model": fp8}, default_config=None)
    _, captured = _build(monkeypatch, router)

    assert captured["visual"] is fp8
    assert captured["language_model"] is router


def test_language_model_only_component_config_leaves_vision_unquantized(monkeypatch):
    # vLLM linear layers reject a router that resolves to None ("All linear
    # layers should support quant method"), so the vision encoder must get
    # None here, not the router.
    router = ComponentQuantizationConfig({"thinker.language_model": _quant("fp8")}, default_config=None)
    _, captured = _build(monkeypatch, router)

    assert captured["visual"] is None
    assert captured["language_model"] is router


def test_online_method_is_scoped_to_language_model(monkeypatch):
    # What vLLM builds for --quantization fp8 on a BF16 checkpoint.
    online = OnlineQuantizationConfig(resolve_quantization_config("fp8_per_tensor", None))
    model, captured = _build(monkeypatch, online)

    assert captured["visual"] is None
    routed = captured["language_model"]
    assert isinstance(routed, ComponentQuantizationConfig)
    assert routed.resolve("thinker.language_model.model.layers.0.mlp.gate_up_proj") is online
    assert routed.resolve("thinker.visual.blocks.0.attn.qkv") is None
    assert model.quant_config is routed


def test_pre_quantized_method_stays_on_language_model(monkeypatch):
    modelopt = _quant("modelopt")
    _, captured = _build(monkeypatch, modelopt)

    assert captured["visual"] is None
    assert captured["language_model"] is modelopt


def test_serialized_fp8_checkpoint_config_is_passed_through(monkeypatch):
    # Serialized checkpoints name their own unquantized layers, so the config
    # reaches every component unchanged, as before online routing existed.
    fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
    model, captured = _build(monkeypatch, fp8)

    assert captured["visual"] is fp8
    assert captured["language_model"] is fp8
    assert model.quant_config is fp8
