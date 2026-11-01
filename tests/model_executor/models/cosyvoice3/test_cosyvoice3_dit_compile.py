# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU configuration/routing checks; real CUDA quality is validated separately."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_source(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[4] / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_source("cosyvoice3_compile_runtime", "vllm_omni/model_executor/models/cosyvoice3/runtime.py")
config_module = load_source("cosyvoice3_compile_config", "vllm_omni/transformers_utils/configs/cosyvoice3.py")


def make_estimator(backend: str) -> torch.nn.Linear:
    estimator = torch.nn.Linear(3, 3)
    block = Mock()
    block.attn.attn.attn_backend.get_name.return_value = backend
    estimator.transformer_blocks = [block]
    return estimator


def test_compile_config_is_opt_in_and_round_trips():
    config = config_module.CosyVoice3Config()
    assert config.cosyvoice3_compile_dit is False
    config = config_module.CosyVoice3Config(cosyvoice3_compile_dit=True)
    restored = config_module.CosyVoice3Config.from_dict(config.to_dict())
    assert restored.cosyvoice3_compile_dit is True


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_compile_config_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        config_module.CosyVoice3Config(cosyvoice3_compile_dit=value)


@pytest.mark.parametrize("backend", ["CUDNN_ATTN", "FLASH_ATTN", "TORCH_SDPA"])
def test_compile_keeps_native_backend_and_has_no_steady_state_wrapper(monkeypatch, backend):
    estimator = make_estimator(backend)
    original = estimator.forward
    compiled = Mock(side_effect=original)
    compiler = Mock(return_value=compiled)
    monkeypatch.setattr(torch, "compile", compiler)
    monkeypatch.setattr(runtime, "cosyvoice3_packed_inference_enabled", lambda: False)

    runtime.compile_cosyvoice3_padded_dit(estimator, torch.device("cuda"))

    compiler.assert_called_once_with(original, options={"emulate_precision_casts": True})
    assert estimator.forward is compiled
    value = torch.randn(2, 3)
    torch.testing.assert_close(estimator(value), original(value))
    estimator(value)
    compiler.assert_called_once()
    assert estimator.transformer_blocks[0].attn.attn.attn_backend.get_name() == backend


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_compile_rejects_non_cuda_before_calling_compiler(monkeypatch, device):
    compiler = Mock()
    monkeypatch.setattr(torch, "compile", compiler)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        runtime.compile_cosyvoice3_padded_dit(make_estimator("TORCH_SDPA"), torch.device(device))
    compiler.assert_not_called()


def test_compile_rejects_packed_profile(monkeypatch):
    compiler = Mock()
    monkeypatch.setattr(torch, "compile", compiler)
    monkeypatch.setattr(runtime, "cosyvoice3_packed_inference_enabled", lambda: True)
    with pytest.raises(ValueError, match="not the packed Flow"):
        runtime.compile_cosyvoice3_padded_dit(make_estimator("CUDNN_ATTN"), torch.device("cuda"))
    compiler.assert_not_called()


@pytest.mark.parametrize("backends", [["UNKNOWN_ATTN"], ["CUDNN_ATTN", "TORCH_SDPA"], []])
def test_compile_rejects_unvalidated_or_mixed_backends(monkeypatch, backends):
    estimator = make_estimator("CUDNN_ATTN")
    estimator.transformer_blocks = [make_estimator(backend).transformer_blocks[0] for backend in backends]
    compiler = Mock()
    monkeypatch.setattr(torch, "compile", compiler)
    monkeypatch.setattr(runtime, "cosyvoice3_packed_inference_enabled", lambda: False)
    with pytest.raises(ValueError, match="selected attention backends"):
        runtime.compile_cosyvoice3_padded_dit(estimator, torch.device("cuda"))
    compiler.assert_not_called()
