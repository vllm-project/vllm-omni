# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import sys
import types

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.quantization import bitsandbytes as bnb_module
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


def test_vllm_linear_bnb4_return_bias_semantics(monkeypatch):
    dummy_bnb = _install_dummy_bnb(monkeypatch)

    def matmul_4bit(x, w_t, quant_state):
        return x @ w_t

    dummy_bnb.matmul_4bit = matmul_4bit
    monkeypatch.setitem(sys.modules, "bitsandbytes", dummy_bnb)

    class DummyVllmLinear(nn.Module):
        def __init__(self, return_bias: bool, skip_bias_add: bool):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))
            self.bias = nn.Parameter(torch.randn(4))
            self.return_bias = return_bias
            self.skip_bias_add = skip_bias_add
            self.quant_method = None

        def forward(self, x: torch.Tensor):
            bias = self.bias if not self.skip_bias_add else None
            out = self.quant_method.apply(self, x, bias)
            if not self.return_bias:
                return out
            output_bias = self.bias if self.skip_bias_add else None
            return out, output_bias

    method = bnb_module._DiffusionBnbLinearMethod(compute_dtype=torch.float32)

    x = torch.randn(2, 4)
    linear = DummyVllmLinear(return_bias=True, skip_bias_add=True)
    linear.weight.quant_state = object()
    linear.quant_method = method
    out, out_bias = linear(x)
    assert torch.allclose(out, x @ linear.weight.t())
    assert out_bias is linear.bias

    linear2 = DummyVllmLinear(return_bias=True, skip_bias_add=False)
    linear2.weight.quant_state = object()
    linear2.quant_method = method
    out2, out_bias2 = linear2(x)
    assert torch.allclose(out2, x @ linear2.weight.t() + linear2.bias)
    assert out_bias2 is None


def test_apply_bnb_quantization_missing_bnb_raises(monkeypatch):
    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("bitsandbytes missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    pipeline = nn.Sequential(nn.Linear(4, 4))
    cfg = OmniDiffusionConfig(
        model="dummy-model",
        quantization="bitsandbytes",
        quantization_config={"load_in_8bit": True},
    )

    with pytest.raises(ImportError, match="bitsandbytes is required"):
        apply_bnb_quantization(pipeline, cfg.quantization_config)


def test_bnb_config_requires_load_in_flag():
    with pytest.raises(ValueError, match="requires load_in_8bit or load_in_4bit"):
        DiffusionBitsAndBytesConfig(load_in_8bit=False, load_in_4bit=False)
