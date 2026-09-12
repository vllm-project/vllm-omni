# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers import qwen_select01_modulation
from vllm_omni.diffusion.layers.qwen_select01_modulation import (
    can_use_qwen_select01_triton,
    fused_layernorm_select01,
    fused_residual_layernorm_select01,
    select01_modulation_native,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("platform_is_cuda", "tensor_is_cuda", "has_triton", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_qwen_select01_triton_requires_cuda_platform(
    monkeypatch,
    platform_is_cuda: bool,
    tensor_is_cuda: bool,
    has_triton: bool,
    expected: bool,
):
    class FakePlatform:
        @staticmethod
        def is_cuda() -> bool:
            return platform_is_cuda

    class FakeTensor:
        is_cuda = tensor_is_cuda

    monkeypatch.setattr(qwen_select01_modulation, "current_platform", FakePlatform())
    monkeypatch.setattr(qwen_select01_modulation, "HAS_TRITON", has_triton)

    assert can_use_qwen_select01_triton(FakeTensor()) is expected


def _make_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device)
    generator.manual_seed(202817)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, generator=generator)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, generator=generator)
    residual_gate = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device, generator=generator)
    mod_params = torch.randn(batch_size * 2, hidden_size * 3, dtype=dtype, device=device, generator=generator)
    index = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64, device=device, generator=generator)
    return x, residual, residual_gate, mod_params, index


def _reference_layernorm_select01(
    x: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, shift, gate = select01_modulation_native(mod_params, index)
    out = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    return out * (1 + scale) + shift, gate


def _reference_residual_layernorm_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale, shift, gate = select01_modulation_native(mod_params, index)
    residual_out = residual + residual_gate * x
    out = F.layer_norm(residual_out.float(), (residual_out.shape[-1],), eps=eps).to(residual_out.dtype)
    return out * (1 + scale) + shift, residual_out, gate


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_select01_native_fallback_matches_reference(dtype: torch.dtype):
    device = torch.device("cpu")
    x, residual, residual_gate, mod_params, index = _make_inputs(2, 17, 64, dtype, device)
    eps = 1e-6

    actual_norm, actual_gate = fused_layernorm_select01(x, mod_params, index, eps)
    ref_norm, ref_gate = _reference_layernorm_select01(x, mod_params, index, eps)
    torch.testing.assert_close(actual_norm, ref_norm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)

    actual_norm, actual_residual, actual_gate = fused_residual_layernorm_select01(
        x,
        residual,
        residual_gate,
        mod_params,
        index,
        eps,
    )
    ref_norm, ref_residual, ref_gate = _reference_residual_layernorm_select01(
        x,
        residual,
        residual_gate,
        mod_params,
        index,
        eps,
    )
    torch.testing.assert_close(actual_norm, ref_norm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_residual, ref_residual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
@pytest.mark.parametrize("seq_len", [1, 7, 128, 1024])
def test_qwen_select01_triton_matches_reference(dtype: torch.dtype, atol: float, rtol: float, seq_len: int):
    device = torch.device("cuda:0")
    x, residual, residual_gate, mod_params, index = _make_inputs(2, seq_len, 256, dtype, device)
    eps = 1e-6

    actual_norm, actual_gate = fused_layernorm_select01(x, mod_params, index, eps)
    ref_norm, ref_gate = _reference_layernorm_select01(x, mod_params, index, eps)
    torch.testing.assert_close(actual_norm, ref_norm, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)

    actual_norm, actual_residual, actual_gate = fused_residual_layernorm_select01(
        x,
        residual,
        residual_gate.expand_as(residual),
        mod_params,
        index,
        eps,
    )
    ref_norm, ref_residual, ref_gate = _reference_residual_layernorm_select01(
        x,
        residual,
        residual_gate.expand_as(residual),
        mod_params,
        index,
        eps,
    )
    torch.testing.assert_close(actual_norm, ref_norm, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_residual, ref_residual, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_qwen_select01_triton_matches_reference_for_structured_multi_batch_index(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    device = torch.device("cuda:0")
    x, residual, residual_gate, mod_params, _ = _make_inputs(3, 16, 256, dtype, device)
    index = torch.tensor(
        [
            [0] * 16,
            [1] * 16,
            [0, 1] * 8,
        ],
        dtype=torch.int64,
        device=device,
    )
    eps = 1e-6

    actual_norm, actual_gate = fused_layernorm_select01(x, mod_params, index, eps)
    ref_norm, ref_gate = _reference_layernorm_select01(x, mod_params, index, eps)
    torch.testing.assert_close(actual_norm, ref_norm, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)

    actual_norm, actual_residual, actual_gate = fused_residual_layernorm_select01(
        x,
        residual,
        residual_gate.expand_as(residual),
        mod_params,
        index,
        eps,
    )
    ref_norm, ref_residual, ref_gate = _reference_residual_layernorm_select01(
        x,
        residual,
        residual_gate.expand_as(residual),
        mod_params,
        index,
        eps,
    )
    torch.testing.assert_close(actual_norm, ref_norm, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_residual, ref_residual, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_select01_cuda_falls_back_without_triton(monkeypatch):
    device = torch.device("cuda:0")
    x, residual, residual_gate, mod_params, index = _make_inputs(2, 7, 256, torch.bfloat16, device)
    residual_gate = residual_gate.expand_as(residual)
    eps = 1e-6

    def fail_if_launched(*args, **kwargs):
        pytest.fail("Triton launcher must not run when HAS_TRITON is false")

    monkeypatch.setattr(qwen_select01_modulation, "HAS_TRITON", False)
    monkeypatch.setattr(qwen_select01_modulation, "_launch_layernorm_select01", fail_if_launched)
    monkeypatch.setattr(qwen_select01_modulation, "_launch_residual_layernorm_select01", fail_if_launched)

    actual_norm, actual_gate = fused_layernorm_select01(x, mod_params, index, eps)
    ref_norm, ref_gate = _reference_layernorm_select01(x, mod_params, index, eps)
    torch.testing.assert_close(actual_norm, ref_norm, atol=0, rtol=0)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)

    actual_norm, actual_residual, actual_gate = fused_residual_layernorm_select01(
        x,
        residual,
        residual_gate,
        mod_params,
        index,
        eps,
    )
    ref_norm, ref_residual, ref_gate = _reference_residual_layernorm_select01(
        x,
        residual,
        residual_gate,
        mod_params,
        index,
        eps,
    )
    torch.testing.assert_close(actual_norm, ref_norm, atol=0, rtol=0)
    torch.testing.assert_close(actual_residual, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)
