# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    ModelOptFp8CheckpointAdapter,
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


class _QuantizedRemappedModelOptModel(nn.Module):
    """FP8 target whose params live under a namespace the raw checkpoint keys
    never use, reachable only via ``remap_checkpoint_key``.

    Mirrors the Cosmos3 case that regressed: the generic weights-mapper cannot
    resolve ``transformer.orig.*`` to ``runtime.*``, so without the remap hook
    the adapter treats every tensor as unresolved and silently drops the scales.
    """

    def __init__(self) -> None:
        super().__init__()
        self.runtime = nn.Module()
        self.runtime.proj = nn.Module()
        self.runtime.proj.register_parameter(
            "weight",
            nn.Parameter(torch.empty(2, 2, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        self.runtime.proj.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )
        self.runtime.proj.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )

    @staticmethod
    def remap_checkpoint_key(name: str) -> str:
        prefix = "transformer.orig.proj."
        if name.startswith(prefix):
            return "runtime.proj." + name[len(prefix) :]
        return name


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


def test_modelopt_adapter_remaps_and_keeps_scales_for_quantized_target():
    """Regression: scales were dropped for a quantized (FP8) target that only the
    model's ``remap_checkpoint_key`` hook can resolve. Assert both scales are
    emitted and the FP8 weight passes through unchanged (no dequantization)."""
    model = _QuantizedRemappedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.5], dtype=torch.float32)
    input_scale = torch.tensor([1.0], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.orig.proj.weight_scale", weight_scale),
                    ("transformer.orig.proj.input_scale", input_scale),
                    ("transformer.orig.proj.weight", fp8_weight),
                ]
            )
        )
    )

    emitted = dict(adapted)
    assert len(adapted) == len(emitted) == 3

    # Both scales are retained (the dropped-scale regression). They keep their
    # source-checkpoint names; the model's load_weights remaps them downstream.
    assert torch.equal(emitted["transformer.orig.proj.weight_scale"], weight_scale)
    assert torch.equal(emitted["transformer.orig.proj.input_scale"], input_scale)

    # The FP8 weight is passed through unchanged under the remapped target name
    # (the quantized param is FP8, so it must NOT be dequantized). Compare via
    # float32 since torch.equal has no FP8 CPU kernel.
    assert emitted["runtime.proj.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(emitted["runtime.proj.weight"].to(torch.float32), fp8_weight.to(torch.float32))
