# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
    _is_offline_quantized_checkpoint,
)
from vllm_omni.quantization.factory import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({"data_type": "mx_fp"}, True),
        ({"is_checkpoint_quantized": True}, True),
        ({"is_checkpoint_fp8_serialized": True}, True),
        ({"is_checkpoint_nvfp4_serialized": True}, True),
        ({"is_checkpoint_mixed_precision_serialized": True}, True),
        ({}, False),
        ({"is_checkpoint_quantized": False}, False),
        ({"is_checkpoint_fp8_serialized": False}, False),
        ({"is_checkpoint_nvfp4_serialized": False}, False),
        ({"is_checkpoint_mixed_precision_serialized": False}, False),
    ],
)
def test_offload_quant_checkpoint_detection(attrs, expected):
    assert _is_offline_quantized_checkpoint(SimpleNamespace(**attrs)) is expected


class _StopBeforeModelInitializationError(Exception):
    pass


@pytest.mark.parametrize(
    "quant_spec,load_device,expected_device",
    [
        pytest.param(
            {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": []},
            "cpu",
            "cpu",
            id="modelopt-fp8-offload",
        ),
        pytest.param(
            {"quant_method": "modelopt", "quantization": {"quant_algo": "FP8"}},
            "cpu",
            "cpu",
            id="legacy-modelopt-fp8-offload",
        ),
        pytest.param("fp8", "cpu", "cuda", id="online-fp8-offload"),
        pytest.param(None, "cpu", "cpu", id="unquantized-offload"),
        pytest.param(
            {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": []},
            "cuda",
            "cuda",
            id="modelopt-fp8-resident",
        ),
    ],
)
def test_offload_model_initialization_device(monkeypatch, quant_spec, load_device, expected_device):
    loader = DiffusersPipelineLoader(
        LoadConfig(),
        od_config=SimpleNamespace(
            dtype=torch.float32,
            quantization_config=build_quant_config(quant_spec),
            parallel_config=SimpleNamespace(use_hsdp=False),
        ),
    )

    def stop_before_model_initialization(load_format, target_device, custom_pipeline_name, *, is_hsdp):
        assert target_device == torch.device(expected_device)
        raise _StopBeforeModelInitializationError

    monkeypatch.setattr(loader, "_init_from_load_format", stop_before_model_initialization)
    with pytest.raises(_StopBeforeModelInitializationError):
        loader.load_model(load_device=load_device, device=torch.device("cuda"))
