# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.z_image import pipeline_z_image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _TextConfig:
    model_type: str


class _TextEncoder(nn.Module):
    def __init__(self, model_type: str = "qwen3"):
        super().__init__()
        self.config = _TextConfig(model_type=model_type)
        self.model = nn.Linear(2, 2)


def test_compile_qwen3_text_base_model_is_static_and_idempotent(monkeypatch):
    encoder = _TextEncoder()
    calls = []

    def fake_compile(function, **kwargs):
        calls.append((function, kwargs))
        return function

    monkeypatch.setattr(pipeline_z_image.torch, "compile", fake_compile)

    assert pipeline_z_image._compile_qwen3_text_base_model(encoder)
    assert pipeline_z_image._compile_qwen3_text_base_model(encoder)
    assert len(calls) == 1
    assert calls[0][1] == {
        "mode": "default",
        "fullgraph": False,
        "dynamic": False,
    }


def test_compile_text_base_model_skips_non_qwen3(monkeypatch):
    encoder = _TextEncoder(model_type="other")

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("non-Qwen3 text encoder should not be compiled")

    monkeypatch.setattr(pipeline_z_image.torch, "compile", unexpected_compile)
    assert not pipeline_z_image._compile_qwen3_text_base_model(encoder)
