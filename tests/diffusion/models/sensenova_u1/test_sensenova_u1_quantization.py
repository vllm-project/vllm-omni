# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 as pipeline_module
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def pipeline_setup(monkeypatch):
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
    monkeypatch.setattr(pipeline_module, "NEOVisionModel", Mock(return_value=nn.Identity()))
    monkeypatch.setattr(pipeline_module, "ConvDecoder", lambda hidden_size: nn.Identity())

    return model_config, language_model_class


@pytest.mark.parametrize(
    ("quantization", "lora_backend", "lora_path"),
    [
        pytest.param(None, "distill", "adapter", id="bf16_lora"),
        pytest.param("fp8", "distill", None, id="fp8_without_lora"),
        pytest.param("fp8", "distill", "", id="fp8_empty_lora_path"),
        pytest.param("component_match", "distill", None, id="component_match"),
        pytest.param("component_unmatched", "distill", "adapter", id="component_unmatched"),
        pytest.param("component_default", "distill", None, id="component_default"),
        pytest.param("component_disabled", "distill", "adapter", id="component_disabled"),
        pytest.param("fp8", "peft", "adapter", id="peft_not_rejected"),
        pytest.param("serialized_fp8", "distill", "adapter", id="serialized_not_rejected"),
    ],
)
def test_pipeline_routes_online_fp8_config(pipeline_setup, quantization, lora_backend, lora_path):
    model_config, language_model_class = pipeline_setup
    fp8_config = build_quant_config("fp8")
    serialized_fp8_config = build_quant_config({"method": "fp8", "is_checkpoint_fp8_serialized": True})
    quant_config, expected_config = {
        None: (None, None),
        "fp8": (fp8_config, fp8_config),
        "serialized_fp8": (serialized_fp8_config, serialized_fp8_config),
        "component_match": (ComponentQuantizationConfig({"language_model": fp8_config}), fp8_config),
        "component_unmatched": (ComponentQuantizationConfig({"vision_model": fp8_config}), None),
        "component_default": (
            ComponentQuantizationConfig({"vision_model": None}, default_config=fp8_config),
            fp8_config,
        ),
        "component_disabled": (ComponentQuantizationConfig({"language_model": None}, default_config=fp8_config), None),
    }[quantization]
    od_config = SimpleNamespace(
        model="sensenova-test-model",
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        revision=None,
        lora_backend=lora_backend,
        lora_path=lora_path,
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline_module.SenseNovaU1Pipeline(od_config=od_config)

    language_model_class.assert_called_once_with(
        model_config.llm_config,
        quant_config=expected_config,
        prefix="language_model",
    )


@pytest.mark.parametrize("quantization", ["fp8", "component_match", "component_default"])
def test_pipeline_rejects_online_fp8_with_distilled_lora_before_loading(pipeline_setup, quantization):
    _, language_model_class = pipeline_setup
    fp8_config = build_quant_config("fp8")
    quant_config = {
        "fp8": fp8_config,
        "component_match": ComponentQuantizationConfig({"language_model": fp8_config}),
        "component_default": ComponentQuantizationConfig({"vision_model": None}, default_config=fp8_config),
    }[quantization]
    od_config = SimpleNamespace(
        model="sensenova-test-model",
        quantization_config=quant_config,
        lora_backend="distill",
        lora_path="adapter",
    )

    with pytest.raises(ValueError) as exc_info:
        pipeline_module.SenseNovaU1Pipeline(od_config=od_config)

    assert str(exc_info.value) == (
        "SenseNova does not support online FP8 with distilled LoRA "
        "Use BF16 without quantization for distilled LoRA, or omit the LoRA options for online FP8."
    )
    language_model_class.assert_not_called()
    pipeline_module.NEOVisionModel.assert_not_called()


def test_distilled_lora_check_does_not_reject_other_quantization():
    pipeline = pipeline_module.SenseNovaU1Pipeline.__new__(pipeline_module.SenseNovaU1Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(lora_backend="distill", lora_path="adapter")
    quant_config = Mock()
    quant_config.get_name.return_value = "int8"

    pipeline._unsupported_quant_methods_check(quant_config)
