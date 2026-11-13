# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers.activation import SiluAndMul

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_silu_and_mul_native_matches_packed_reference() -> None:
    packed = torch.randn(257, 256, dtype=torch.float32)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(SiluAndMul().forward_native(packed), F.silu(gate) * up)


@pytest.mark.cpu
def test_silu_and_mul_npu_dispatch_falls_back_for_cpu_tensor() -> None:
    packed = torch.randn(257, 256, dtype=torch.float32)

    torch.testing.assert_close(
        SiluAndMul().forward_npu(packed),
        SiluAndMul().forward_native(packed),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=hardware_marks(res={"npu": "A2"}, num_cards=1), id="A2-bf16"),
        pytest.param(torch.bfloat16, marks=hardware_marks(res={"npu": "A3"}, num_cards=1), id="A3-bf16"),
        pytest.param(torch.bfloat16, marks=hardware_marks(res={"npu": "A5"}, num_cards=1), id="A5-bf16"),
        pytest.param(torch.float16, marks=hardware_marks(res={"npu": "310P"}, num_cards=1), id="310P-fp16"),
    ],
)
def test_silu_and_mul_npu_matches_packed_reference(dtype: torch.dtype) -> None:
    packed = torch.randn(257, 256, device="npu", dtype=dtype)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(
        SiluAndMul()(packed),
        F.silu(gate) * up,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_silu_and_mul_hip_avoids_precompiled_cuda_kernel(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    def unsupported_cuda(self: SiluAndMul, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("unsupported fused kernel")

    monkeypatch.setenv("VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS", "1")
    monkeypatch.setattr(SiluAndMul, "forward_cuda", unsupported_cuda)
    packed = torch.randn(17, 256, generator=torch.Generator().manual_seed(42), dtype=dtype)
    gate, up = packed.chunk(2, dim=-1)

    output = SiluAndMul().forward_hip(packed)
    torch.testing.assert_close(output, F.silu(gate) * up)
    assert output.dtype == dtype


@pytest.mark.cpu
def test_silu_and_mul_hip_uses_cuda_path_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS", raising=False)
    sentinel = torch.zeros(1)
    monkeypatch.setattr(SiluAndMul, "forward_cuda", lambda self, x: sentinel)

    assert SiluAndMul().forward_hip(torch.randn(2, 8)) is sentinel
