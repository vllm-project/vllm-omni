# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.layers.row_parallel_linear import DiffusionRowParallelLinearWithLoRA
from vllm_omni.diffusion.lora.layers.torch_linear import DiffusionTorchLinearWithLoRA

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _DummyLoRAConfig:
    fully_sharded_loras: bool = False
    max_lora_rank: int = 2
    lora_dtype: torch.dtype = torch.float32


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


def test_diffusion_base_linear_apply_adds_composed_additive_bias():
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(torch.eye(3))
    layer.lora_a_stacked = (torch.zeros((1, 1, 1, 3)),)
    layer.lora_b_stacked = (torch.zeros((1, 1, 3, 1)),)
    layer.output_slices = (3,)
    layer._diffusion_lora_active_slices = (True,)
    layer._diffusion_additive_bias = (torch.tensor([0.5, -1.0, 2.0]),)

    output = layer.apply(torch.tensor([[1.0, 2.0, 3.0]]))

    torch.testing.assert_close(output, torch.tensor([[1.5, 1.0, 5.0]]))


def test_torch_linear_wrapper_applies_weight_and_bias_deltas():
    base_layer = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        base_layer.weight.zero_()
        base_layer.bias.copy_(torch.tensor([1.0, -1.0]))
    layer = DiffusionTorchLinearWithLoRA(base_layer)
    layer.create_lora_weights(1, _DummyLoRAConfig())
    layer.set_lora(
        0,
        lora_a=torch.tensor([[1.0, 0.0, 0.0]]),
        lora_b=torch.tensor([[2.0], [3.0]]),
    )
    layer.set_additive_bias(torch.tensor([0.5, 1.5]))

    output = layer(torch.tensor([[4.0, 5.0, 6.0]]))

    torch.testing.assert_close(output, torch.tensor([[9.5, 12.5]]))


def test_torch_linear_wrapper_rejects_additive_bias_shape_mismatch():
    layer = DiffusionTorchLinearWithLoRA(torch.nn.Linear(3, 2))
    layer.create_lora_weights(1, _DummyLoRAConfig())

    with pytest.raises(ValueError, match=r"got \(1,\), expected \(2,\)"):
        layer.set_additive_bias(torch.ones(1))


def test_torch_linear_wrapper_moves_cpu_sidecars_to_input_device():
    # Offload only tracks parameters and buffers, while the dynamic LoRA
    # sidecars are plain tensors. Simulate an offloaded layer whose base weight
    # and input are on CUDA but whose LoRA tensors remain on CPU.
    with FakeTensorMode(allow_non_fake_inputs=True):
        base_layer = torch.nn.Linear(3, 2, device="cuda")
        layer = DiffusionTorchLinearWithLoRA(base_layer)
        layer.create_lora_weights(1, _DummyLoRAConfig())
        layer.lora_a_stacked = tuple(tensor.cpu() for tensor in layer.lora_a_stacked)
        layer.lora_b_stacked = tuple(tensor.cpu() for tensor in layer.lora_b_stacked)
        layer._diffusion_lora_active_slices = (True,)
        layer._diffusion_additive_bias = (torch.zeros(2),)

        output = layer(torch.randn(1, 3, device="cuda"))

    assert output.device.type == "cuda"


def test_lora_runtime_can_move_without_moving_dense_base():
    base_layer = torch.nn.Linear(3, 2)
    layer = DiffusionTorchLinearWithLoRA(base_layer)
    layer.create_lora_weights(1, _DummyLoRAConfig())
    layer.set_additive_bias(torch.ones(2))

    layer.move_lora_runtime_to(torch.device("meta"))

    assert base_layer.weight.device.type == "cpu"
    assert all(tensor.device.type == "meta" for tensor in layer.lora_a_stacked)
    assert all(tensor.device.type == "meta" for tensor in layer.lora_b_stacked)
    assert all(bias is None or bias.device.type == "meta" for bias in layer._diffusion_additive_bias)


@pytest.mark.parametrize(("tp_rank", "expects_bias"), [(0, True), (1, False)])
def test_row_parallel_additive_bias_is_contributed_once(monkeypatch, tp_rank: int, expects_bias: bool):
    received: list[torch.Tensor | list[torch.Tensor | None] | None] = []
    monkeypatch.setattr(
        DiffusionBaseLinearLayerWithLoRA,
        "set_additive_bias",
        lambda _self, bias: received.append(bias),
    )
    layer = DiffusionRowParallelLinearWithLoRA.__new__(DiffusionRowParallelLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.tp_size = 2
    layer.tp_rank = tp_rank
    bias = torch.tensor([1.0, 2.0])

    layer.set_additive_bias(bias)

    assert (received[0] is bias) is expects_bias
