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
        self.transformer.block.to_qkv.register_parameter(
            "pre_quant_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )


class _RemappedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.runtime = nn.Module()
        self.runtime.proj = nn.Linear(2, 2, bias=False)

    @staticmethod
    def remap_checkpoint_key(name: str) -> str:
        return {
            "transformer.orig.proj.weight": "runtime.proj.weight",
        }.get(name, name)


class _FakeQuantConfig:
    def __init__(self, name: str, **attrs: object) -> None:
        self._name = name
        for attr_name, value in attrs.items():
            setattr(self, attr_name, value)

    def get_name(self) -> str:
        return self._name


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


def test_modelopt_adapter_keeps_awq_pre_quant_scale_for_quantized_target():
    model = _QuantizedPackedModelOptModel()
    adapter = ModelOptNvFp4CheckpointAdapter(model, _make_source())
    scale = torch.tensor([0.25], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.pre_quant_scale", scale),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.pre_quant_scale",
    ]
    assert torch.equal(adapted[0][1], scale)


def test_modelopt_adapter_uses_checkpoint_key_remap_for_target_dtype():
    model = _RemappedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.orig.proj.weight", fp8_weight),
                    ("transformer.orig.proj.weight_scale", scale),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == ["runtime.proj.weight"]
    assert adapted[0][1].dtype == model.runtime.proj.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


def test_modelopt_nvfp4_adapter_accepts_serialized_checkpoints():
    quant_config = _FakeQuantConfig(
        "modelopt_fp4",
        is_checkpoint_nvfp4_serialized=True,
    )

    assert ModelOptNvFp4CheckpointAdapter.is_compatible(
        _make_source(),
        quant_config,
        use_safetensors=True,
    )


def test_modelopt_mixed_adapter_requires_serialized_checkpoints():
    serialized_config = _FakeQuantConfig(
        "modelopt_mixed",
        is_checkpoint_mixed_precision_serialized=True,
    )
    online_config = _FakeQuantConfig(
        "modelopt_mixed",
        is_checkpoint_mixed_precision_serialized=False,
    )

    assert ModelOptMixedPrecisionCheckpointAdapter.is_compatible(
        _make_source(),
        serialized_config,
        use_safetensors=True,
    )
    assert not ModelOptMixedPrecisionCheckpointAdapter.is_compatible(
        _make_source(),
        online_config,
        use_safetensors=True,
    )
