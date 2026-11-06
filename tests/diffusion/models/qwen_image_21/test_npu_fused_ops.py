# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.qwen_image_21 import npu_fused_ops as ops

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def swiglu_kernel(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an available NPU")
    if not ops.HAS_TRITON or "ascend" not in getattr(ops.triton.backends, "backends", {}):
        pytest.skip("Requires Triton Ascend")
    if torch.npu.get_device_name() not in {"Ascend910B1", "Ascend910B3"}:
        pytest.skip("Requires Ascend910B1 or Ascend910B3")

    # Kernel correctness is independent of the production performance whitelist.
    kernel = ops._swiglu_kernel

    class KernelProbe:
        calls = 0

        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(*args, **kwargs):
                result = launch(*args, **kwargs)
                self.calls += 1
                return result

            return run

    probe = KernelProbe()
    monkeypatch.setattr(ops, "_swiglu_kernel", probe)
    return probe


@pytest.mark.cpu
def test_cpu_is_ineligible():
    with torch.inference_mode():
        assert not ops.can_fuse_swiglu(torch.empty(1, 4096, 4096, device="meta"))
        assert not ops.can_fuse_swiglu(torch.empty(1, 1, 4096))


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 12288),
        (3, 12288),
        (4097, 12288),
        # Batch-one decode and prefill shapes from 1024x1024 generation/editing.
        (1, 4096, 12288),
        (1, 4131, 12288),
        (1, 8234, 12288),
    ],
)
def test_swiglu_rounding_and_ownership(shape, swiglu_kernel):
    with torch.inference_mode():
        torch.manual_seed(42)
        gate = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        original_up = up.clone()
        expected = torch.nn.functional.silu(gate) * up
        pointer = gate.data_ptr()
        actual = ops.swiglu_inplace(gate, up)
        assert swiglu_kernel.calls == 1
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(up, original_up, rtol=0, atol=0)


@hardware_test(res={"npu": "A2"}, num_cards=1)
def test_all_bf16_gate_values(swiglu_kernel):
    with torch.inference_mode():
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        gate = bits.repeat(3).reshape(-1, 12288).to("npu")
        for seed in range(3):
            torch.manual_seed(seed)
            up = torch.randn_like(gate)
            original_up = up.clone()
            expected = torch.nn.functional.silu(gate) * up
            work = gate.clone()
            actual = ops.swiglu_inplace(work, up)
            assert swiglu_kernel.calls == seed + 1
            assert actual.data_ptr() == work.data_ptr()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(up, original_up, rtol=0, atol=0)


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("kind", ["width", "tp_shard_width", "strided", "empty", "fp32", "up_fp32"])
def test_unsupported_layout_falls_back(monkeypatch, kind):
    pytest.importorskip("torch_npu")
    with torch.inference_mode():
        width = {"width": 31, "tp_shard_width": 6144}.get(kind, 12288)
        shape = (2, width)
        gate = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        if kind == "strided":
            gate = gate.t().contiguous().t()
        elif kind == "empty":
            gate = gate[:0]
        elif kind == "fp32":
            gate = gate.float()
        up = torch.randn_like(gate)
        if kind == "up_fp32":
            up = up.float()
        original_up = up.clone()
        # In-place multiplication rounds back to gate.dtype for mixed dtypes.
        expected = torch.nn.functional.silu(gate.clone(), inplace=True).mul_(up)
        pointer = gate.data_ptr()
        monkeypatch.setattr(ops, "_swiglu_kernel", None)
        actual = ops.swiglu_inplace(gate, up)
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(up, original_up, rtol=0, atol=0)


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
