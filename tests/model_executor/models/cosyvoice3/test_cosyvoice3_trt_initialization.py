# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TRT initialization must preserve the requested attention semantics."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3, flow_estimator_trt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def model(tmp_path, monkeypatch):
    # Only initialization is exercised, without loading model weights or vLLM.
    model = SimpleNamespace(model_dir=str(tmp_path))
    decoder = torch.nn.Module()
    decoder.estimator = torch.nn.Linear(1, 1)
    decoder.estimator.static_chunk_size = 50
    model.code2wav = SimpleNamespace(flow_model=SimpleNamespace(decoder=decoder))
    model._flow_estimator_onnx_dir = lambda: cosyvoice3.CosyVoice3Model._flow_estimator_onnx_dir(model)
    model._resolve_flow_estimator_onnx = Mock(return_value="legacy.onnx")
    model._flow_estimator_cache_key = lambda: "checkpoint-key"
    monkeypatch.setattr(cosyvoice3, "cosyvoice3_packed_inference_enabled", lambda: False)
    monkeypatch.setenv("COSYVOICE3_TRT", "1")
    monkeypatch.delenv("COSYVOICE3_TRT_CHUNK_MASK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return model


def initialize(model):
    cosyvoice3.CosyVoice3Model._maybe_enable_code2wav_trt(model)


@pytest.mark.parametrize("error", [ImportError("onnx missing"), RuntimeError("TRT build failed")])
def test_chunk_engine_failure_preserves_torch_and_does_not_try_legacy(model, monkeypatch, caplog, error):
    original = model.code2wav.flow_model.decoder.estimator
    masked = Mock(side_effect=error)
    legacy = Mock()
    monkeypatch.setattr(flow_estimator_trt, "build_chunk_mask_flow_estimator_trt", masked)
    monkeypatch.setattr(flow_estimator_trt, "build_flow_estimator_trt", legacy)
    with caplog.at_level(logging.WARNING):
        initialize(model)
        initialize(model)
    assert model.code2wav.flow_model.decoder.estimator is original
    assert masked.call_count == 1
    legacy.assert_not_called()
    model._resolve_flow_estimator_onnx.assert_not_called()
    assert "keeping torch estimator (chunk_mask=True)" in caplog.text


def test_chunk_engine_success_swaps_only_after_validation(model, monkeypatch):
    wrapper = SimpleNamespace(supports_attn_mask=True)
    masked = Mock(return_value=wrapper)
    monkeypatch.setattr(flow_estimator_trt, "build_chunk_mask_flow_estimator_trt", masked)
    initialize(model)
    assert model.code2wav.flow_model.decoder.estimator is wrapper
    assert wrapper.static_chunk_size == 50
    assert masked.call_args.kwargs["cache_key"] == "checkpoint-key"
    model._resolve_flow_estimator_onnx.assert_not_called()


def test_chunk_engine_without_mask_keeps_torch(model, monkeypatch):
    original = model.code2wav.flow_model.decoder.estimator
    monkeypatch.setattr(
        flow_estimator_trt,
        "build_chunk_mask_flow_estimator_trt",
        lambda *args, **kwargs: SimpleNamespace(supports_attn_mask=False),
    )
    initialize(model)
    assert model.code2wav.flow_model.decoder.estimator is original


def test_explicit_chunk_mask_opt_out_uses_legacy_engine(model, monkeypatch):
    monkeypatch.setenv("COSYVOICE3_TRT_CHUNK_MASK", "0")
    masked = Mock()
    wrapper = SimpleNamespace(supports_attn_mask=False)
    legacy = Mock(return_value=wrapper)
    monkeypatch.setattr(flow_estimator_trt, "build_chunk_mask_flow_estimator_trt", masked)
    monkeypatch.setattr(flow_estimator_trt, "build_flow_estimator_trt", legacy)
    initialize(model)
    assert model.code2wav.flow_model.decoder.estimator is wrapper
    masked.assert_not_called()
    legacy.assert_called_once_with("legacy.onnx", device="cuda")


def test_exports_use_cache_even_for_writable_model_directory(model, tmp_path, monkeypatch):
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    assert model._flow_estimator_onnx_dir() == str(tmp_path / "cache" / "exported")
