# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.qwen_image_21 import npu_fused_ops as ops

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_cpu_is_ineligible():
    with torch.inference_mode():
        assert not ops.can_fuse_swiglu(torch.empty(1, 4096, 4096, device="meta"))
        assert not ops.can_fuse_swiglu(torch.empty(1, 1, 4096))


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("rows", [1, 3, 4097])
def test_swiglu_rounding_and_ownership(rows):
    pytest.importorskip("torch_npu")
    if not ops._available(0):
        pytest.skip("Requires Triton Ascend on Ascend910B1")
    with torch.inference_mode():
        torch.manual_seed(42)
        gate = torch.randn(rows, 12288, device="npu", dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        original_up = up.clone()
        expected = torch.nn.functional.silu(gate) * up
        pointer = gate.data_ptr()
        actual = ops.swiglu_inplace(gate, up)
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(up, original_up, rtol=0, atol=0)


@hardware_test(res={"npu": "A2"}, num_cards=1)
def test_all_bf16_gate_values():
    pytest.importorskip("torch_npu")
    if not ops._available(0):
        pytest.skip("Requires Triton Ascend on Ascend910B1")
    with torch.inference_mode():
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        gate = bits.repeat(3).reshape(-1, 12288).to("npu")
        for seed in range(3):
            torch.manual_seed(seed)
            up = torch.randn_like(gate)
            expected = torch.nn.functional.silu(gate) * up
            actual = ops.swiglu_inplace(gate.clone(), up)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("kind", ["width", "strided", "empty", "fp32"])
def test_unsupported_layout_falls_back(monkeypatch, kind):
    pytest.importorskip("torch_npu")
    with torch.inference_mode():
        shape = (2, 31) if kind == "width" else (2, 12288)
        gate = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        if kind == "strided":
            gate = gate.t().contiguous().t()
        elif kind == "empty":
            gate = gate[:0]
        elif kind == "fp32":
            gate = gate.float()
        up = torch.randn_like(gate)
        expected = torch.nn.functional.silu(gate) * up
        monkeypatch.setattr(ops, "_swiglu_kernel", None)
        actual = ops.swiglu_inplace(gate, up)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("eligible", [False, True])
def test_mlp_automatic_fusion_and_fallback(monkeypatch, eligible):
    from torch import nn

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21SwiGLUFeedForward

    pytest.importorskip("torch_npu")
    model = QwenImage21SwiGLUFeedForward.__new__(QwenImage21SwiGLUFeedForward)
    nn.Module.__init__(model)
    model._npu_inplace_mlp = True

    class Projection(nn.Module):
        def forward(self, x):
            return x.clone()

    model.gate_layer = Projection()
    model.proj = Projection()
    model.out = nn.Identity()
    calls = []

    def fused(gate, up):
        calls.append(True)
        return torch.nn.functional.silu(gate, inplace=True).mul_(up)

    monkeypatch.setattr(ops, "can_fuse_swiglu", lambda x: eligible)
    monkeypatch.setattr(ops, "swiglu_inplace", fused)
    with torch.inference_mode():
        x = torch.randn(1, 2, 4, device="npu", dtype=torch.bfloat16)
        original = x.clone()
        expected = torch.nn.functional.silu(x) * x
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert bool(calls) == eligible
