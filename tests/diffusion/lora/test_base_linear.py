# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _DummyLoRAConfig:
    fully_sharded_loras: bool = False


class _DummyQuantMethod:
    def __init__(self, weight: torch.Tensor):
        self._weight = weight

    def apply(self, _base_layer, x: torch.Tensor, bias: torch.Tensor | None):
        y = x @ self._weight.t()
        if bias is not None:
            y = y + bias
        return y


def test_diffusion_base_linear_apply_multi_slice():
    # Build a fake diffusion LoRA layer with 2 slices and rank=2.
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 3
    out_slices = (2, 1)
    rank = 2

    # Base weight: identity-ish mapping to make base output easy to reason about.
    base_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    # Allocate stacked weights: (max_loras=1, 1, rank, in_dim) and (1, 1, out_slice, rank)
    a0 = torch.zeros((1, 1, rank, in_dim))
    b0 = torch.zeros((1, 1, out_slices[0], rank))
    a1 = torch.zeros((1, 1, rank, in_dim))
    b1 = torch.zeros((1, 1, out_slices[1], rank))

    # Slice 0: delta0 = (x @ A0.T) @ B0.T
    A0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    a0[0, 0, :, :] = A0
    b0[0, 0, :, :] = B0

    # Slice 1: delta1 = (x @ A1.T) @ B1.T
    A1 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # (2, 3)
    B1 = torch.tensor([[2.0, 0.0]])  # (1, 2)
    a1[0, 0, :, :] = A1
    b1[0, 0, :, :] = B1

    layer.lora_a_stacked = (a0, a1)
    layer.lora_b_stacked = (b0, b1)
    layer.output_slices = out_slices

    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = layer.apply(x)

    # Base output is identity: [1,2,3]
    base_out = x @ base_weight.t()
    # delta0:
    # (x @ A0.T) = [1,2]
    # [1,2] @ B0.T = [1,2]
    delta0 = torch.tensor([[1.0, 2.0]])
    # delta1:
    # (x @ A1.T) = [3,1]
    # [3,1] @ B1.T = [6]
    delta1 = torch.tensor([[6.0]])
    expected = torch.cat([base_out[:, :2] + delta0, base_out[:, 2:3] + delta1], dim=-1)
    assert torch.allclose(out, expected)


def test_diffusion_base_linear_reset_lora_disables_fast_path(monkeypatch):
    # Verify that after reset_lora(), apply() skips LoRA matmuls even if the
    # LoRA tensors are still allocated and non-empty.
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 2
    out_dim = 2
    rank = 1

    base_weight = torch.eye(in_dim)
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    a = torch.ones((1, 1, rank, in_dim))
    b = torch.tensor([[[[1.0], [2.0]]]])  # (1,1,out_dim,rank)

    layer.lora_a_stacked = (a,)
    layer.lora_b_stacked = (b,)
    layer.output_slices = (out_dim,)
    layer._diffusion_lora_active_slices = (True,)

    x = torch.tensor([[1.0, 2.0]])
    out_active = layer.apply(x)
    assert torch.allclose(out_active, torch.tensor([[4.0, 8.0]]))

    monkeypatch.setattr(BaseLinearLayerWithLoRA, "reset_lora", lambda self, index: None)
    layer.reset_lora(0)

    assert layer._diffusion_lora_active_slices == (False,)
    out_inactive = layer.apply(x)
    assert torch.allclose(out_inactive, x)


def test_diffusion_base_linear_apply_respects_inactive_slices():
    # Build a fake diffusion LoRA layer with 2 slices and rank=2.
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 3
    out_slices = (2, 1)
    rank = 2

    base_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    a0 = torch.zeros((1, 1, rank, in_dim))
    b0 = torch.zeros((1, 1, out_slices[0], rank))
    a1 = torch.zeros((1, 1, rank, in_dim))
    b1 = torch.zeros((1, 1, out_slices[1], rank))

    A0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    a0[0, 0, :, :] = A0
    b0[0, 0, :, :] = B0

    A1 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # (2, 3)
    B1 = torch.tensor([[2.0, 0.0]])  # (1, 2)
    a1[0, 0, :, :] = A1
    b1[0, 0, :, :] = B1

    layer.lora_a_stacked = (a0, a1)
    layer.lora_b_stacked = (b0, b1)
    layer.output_slices = out_slices
    layer._diffusion_lora_active_slices = (True, False)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = layer.apply(x)

    # Only the first slice should be adapted.
    expected = torch.tensor([[2.0, 4.0, 3.0]])
    assert torch.allclose(out, expected)
