# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.quantization.bitsandbytes import (
    DiffusionBitsAndBytesConfig,
    apply_bnb_quantization,
    patch_transformers_for_bnb_load,
)

_DUMMY_BNB_WEIGHT = 0.123


def _install_dummy_bnb(monkeypatch: pytest.MonkeyPatch):
    class DummyLinear8bitLt(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, has_fp16_weights=False, device=None, **kwargs):
            super().__init__(in_features, out_features, bias=bias, device=device)
            self.has_fp16_weights = has_fp16_weights
            nn.init.constant_(self.weight, _DUMMY_BNB_WEIGHT)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    class DummyLinear4bit(nn.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            compute_dtype=None,
            compress_statistics=False,
            quant_type="fp4",
            device=None,
            **kwargs,
        ):
            super().__init__(in_features, out_features, bias=bias, device=device)
            self.compute_dtype = compute_dtype
            self.compress_statistics = compress_statistics
            self.quant_type = quant_type

    dummy_bnb = types.SimpleNamespace(
        nn=types.SimpleNamespace(
            Linear8bitLt=DummyLinear8bitLt,
            Linear4bit=DummyLinear4bit,
        )
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", dummy_bnb)
    return dummy_bnb


def test_quant_config_normalization():
    cfg = OmniDiffusionConfig(
        model="dummy-model",
        quantization="BNB_4BIT",
        quantization_config={
            "modules": "transformer, text_encoder_2",
            "bnb_4bit_compute_dtype": "fp16",
        },
    )
    assert isinstance(cfg.quantization_config, DiffusionBitsAndBytesConfig)
    assert cfg.quantization_config.load_in_4bit is True
    assert cfg.quantization_config.load_in_8bit is False
    assert cfg.quantization_config.modules == ["transformer", "text_encoder_2"]
    assert cfg.quantization_config.bnb_4bit_compute_dtype == torch.float16


def test_apply_bnb_quantization_replaces_linear_modules(monkeypatch):
    bnb = _install_dummy_bnb(monkeypatch)

    class DummyPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Sequential(
                nn.Linear(4, 8, bias=True),
                nn.ReLU(),
                nn.ModuleList([nn.Linear(8, 8, bias=False), nn.Sequential(nn.Linear(8, 4))]),
            )

    pipeline = DummyPipeline()
    cfg = OmniDiffusionConfig(
        model="dummy-model",
        quantization="bitsandbytes",
        quantization_config={"load_in_8bit": True, "modules": ["transformer"]},
    )
    assert isinstance(cfg.quantization_config, DiffusionBitsAndBytesConfig)
    apply_bnb_quantization(pipeline, cfg.quantization_config)

    assert isinstance(pipeline.transformer[0], bnb.nn.Linear8bitLt)
    assert isinstance(pipeline.transformer[2][0], bnb.nn.Linear8bitLt)
    assert isinstance(pipeline.transformer[2][1][0], bnb.nn.Linear8bitLt)


def test_apply_bnb_quantization_copy_weights_false_pre_replace(monkeypatch):
    _install_dummy_bnb(monkeypatch)

    class DummyPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Sequential(nn.Linear(4, 4, bias=False))

    pipeline = DummyPipeline()
    pipeline.transformer[0].weight.data.zero_()
    cfg = OmniDiffusionConfig(
        model="dummy-model",
        quantization="bitsandbytes",
        quantization_config={"load_in_8bit": True, "modules": ["transformer"]},
    )
    apply_bnb_quantization(pipeline, cfg.quantization_config, copy_weights=False)

    assert isinstance(pipeline.transformer[0], nn.Linear)
    assert torch.allclose(
        pipeline.transformer[0].weight,
        torch.full_like(pipeline.transformer[0].weight, _DUMMY_BNB_WEIGHT),
    )


def test_bnb_llm_int8_has_fp16_weight_passed(monkeypatch):
    _install_dummy_bnb(monkeypatch)

    class DummyPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Sequential(nn.Linear(4, 4, bias=False))

    pipeline = DummyPipeline()
    cfg = OmniDiffusionConfig(
        model="dummy-model",
        quantization="bitsandbytes",
        quantization_config={
            "load_in_8bit": True,
            "modules": ["transformer"],
            "llm_int8_has_fp16_weight": True,
        },
    )
    apply_bnb_quantization(pipeline, cfg.quantization_config, copy_weights=False)

    assert getattr(pipeline.transformer[0], "has_fp16_weights", False) is True


def test_bnb_pre_replace_no_false_warning(monkeypatch, caplog):
    from vllm.logger import _print_warning_once

    _print_warning_once.cache_clear()
    _install_dummy_bnb(monkeypatch)

    class DummyPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Sequential(nn.Linear(4, 4, bias=False))

    pipeline = DummyPipeline()
    cfg = OmniDiffusionConfig(
        model="dummy",
        quantization="bitsandbytes",
        quantization_config={"load_in_8bit": True, "modules": ["transformer"]},
    )

    with caplog.at_level("WARNING"):
        apply_bnb_quantization(pipeline, cfg.quantization_config, copy_weights=False)
        apply_bnb_quantization(pipeline, cfg.quantization_config, copy_weights=True)

    assert not any("no Linear layers replaced" in r.message for r in caplog.records)


def test_hf_bnb_patch_inject_and_restore(monkeypatch):
    from vllm.logger import _print_warning_once

    _print_warning_once.cache_clear()

    class DummyBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyPreTrainedModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return kwargs

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.BitsAndBytesConfig = DummyBitsAndBytesConfig
    modeling_utils_mod = types.ModuleType("transformers.modeling_utils")
    modeling_utils_mod.PreTrainedModel = DummyPreTrainedModel

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils_mod)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cfg = DiffusionBitsAndBytesConfig(load_in_8bit=True, modules=["transformer"])
    orig_attr = DummyPreTrainedModel.__dict__["from_pretrained"]

    with patch_transformers_for_bnb_load(cfg, device=torch.device("cuda")) as used:
        out = DummyPreTrainedModel.from_pretrained("transformer", subfolder="transformer")
        assert "quantization_config" in out
        assert "device_map" in out
        assert "transformer" in used

    assert DummyPreTrainedModel.__dict__["from_pretrained"] is orig_attr
