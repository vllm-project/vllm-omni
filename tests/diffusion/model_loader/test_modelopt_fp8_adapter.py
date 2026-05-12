# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    ModelOptFp8CheckpointAdapter,
    ModelOptMixedPrecisionCheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
    get_checkpoint_adapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _PackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Linear(2, 2, bias=False)


class _QuantizedPackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Module()
        self.transformer.block.to_qkv.register_parameter(
            "weight",
            nn.Parameter(torch.empty(2, 2, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )


class _QuantizedPackedModelOptNvFp4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Module()
        self.transformer.block.to_qkv.register_parameter(
            "weight",
            nn.Parameter(torch.empty(2, 1, dtype=torch.uint8), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty(2, 1, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "weight_scale_2",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )


class _QuantConfig:
    def __init__(self, name: str, **attrs) -> None:
        self.name = name
        for attr_name, attr_value in attrs.items():
            setattr(self, attr_name, attr_value)

    def get_name(self) -> str:
        return self.name


def _make_source() -> SimpleNamespace:
    return SimpleNamespace(
        subfolder="transformer",
        prefix="transformer.",
    )


def test_modelopt_adapter_dequantizes_fp8_weight_for_full_precision_target():
    model = _PackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                    ("transformer.block.to_q.weight", fp8_weight),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == ["transformer.block.to_q.weight"]
    assert adapted[0][1].dtype == model.transformer.block.to_qkv.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


def test_modelopt_adapter_keeps_scale_tensors_for_quantized_target():
    model = _QuantizedPackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.weight_scale",
        "transformer.block.to_q.input_scale",
    ]


def test_modelopt_nvfp4_adapter_selected_for_checkpoint_config():
    model = _QuantizedPackedModelOptNvFp4Model()
    quant_config = _QuantConfig(
        "modelopt_fp4",
        is_checkpoint_nvfp4_serialized=True,
    )

    adapter = get_checkpoint_adapter(model, _make_source(), quant_config, use_safetensors=True)

    assert isinstance(adapter, ModelOptNvFp4CheckpointAdapter)


def test_modelopt_mixed_adapter_selected_for_checkpoint_config():
    model = _QuantizedPackedModelOptNvFp4Model()
    quant_config = _QuantConfig("modelopt_mixed")

    adapter = get_checkpoint_adapter(model, _make_source(), quant_config, use_safetensors=True)

    assert isinstance(adapter, ModelOptMixedPrecisionCheckpointAdapter)


def test_modelopt_nvfp4_adapter_keeps_quantized_tensors_for_quantized_target():
    model = _QuantizedPackedModelOptNvFp4Model()
    adapter = ModelOptNvFp4CheckpointAdapter(model, _make_source())
    weight = torch.tensor([[0x12], [0x34]], dtype=torch.uint8)
    weight_scale = torch.tensor([[1.0], [0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    weight_scale_2 = torch.tensor([0.25], dtype=torch.float32)
    input_scale = torch.tensor([0.75], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", weight_scale),
                    ("transformer.block.to_q.weight_scale_2", weight_scale_2),
                    ("transformer.block.to_q.input_scale", input_scale),
                    ("transformer.block.to_q.weight", weight),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.weight_scale",
        "transformer.block.to_q.weight_scale_2",
        "transformer.block.to_q.input_scale",
        "transformer.block.to_q.weight",
    ]
    assert adapted[-1][1].dtype == torch.uint8
