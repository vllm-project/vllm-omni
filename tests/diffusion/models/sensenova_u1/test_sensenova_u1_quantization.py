# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 as pipeline_module
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("quantization", [None, "fp8"], ids=["unquantized", "fp8"])
def test_pipeline_routes_online_fp8_config(monkeypatch, quantization):
    model_config = SimpleNamespace(
        llm_config=SimpleNamespace(hidden_size=8),
        vision_config=SimpleNamespace(patch_size=2),
        downsample_ratio=0.5,
        use_pixel_head=True,
        add_noise_scale_embedding=False,
    )
    language_model = nn.Module()
    language_model.model = nn.Module()
    language_model_class = Mock(return_value=language_model)

    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "_resolve_model_path", lambda path: path)
    monkeypatch.setattr(pipeline_module.SenseNovaU1Config, "from_pretrained", Mock(return_value=model_config))
    monkeypatch.setattr(pipeline_module.AutoTokenizer, "from_pretrained", Mock(return_value=Mock()))
    monkeypatch.setattr(pipeline_module, "SenseNovaU1ForCausalLM", language_model_class)
    monkeypatch.setattr(pipeline_module, "NEOVisionModel", lambda config: nn.Identity())
    monkeypatch.setattr(pipeline_module, "ConvDecoder", lambda hidden_size: nn.Identity())

    quant_config = build_quant_config(quantization) if quantization else None
    od_config = SimpleNamespace(
        model="sensenova-test-model",
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        revision=None,
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline_module.SenseNovaU1Pipeline(od_config=od_config)

    language_model_class.assert_called_once_with(
        model_config.llm_config,
        quant_config=quant_config,
        prefix="language_model",
    )
