# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

import vllm_omni.diffusion.layers.sana_rms_norm as sana_rms

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required"),
]


@pytest.fixture(autouse=True)
def _reset_fusion_state() -> Iterator[None]:
    sana_rms._VERIFIED_SIGNATURES.clear()
    sana_rms._DISABLED_SIGNATURES.clear()
    yield
    sana_rms._VERIFIED_SIGNATURES.clear()
    sana_rms._DISABLED_SIGNATURES.clear()


def _reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normalized = normalized.to(weight.dtype)
    return normalized * weight


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype is torch.bfloat16:
        return torch.equal(left.view(torch.int16), right.view(torch.int16))
    if left.dtype is torch.float32:
        return torch.equal(left.view(torch.int32), right.view(torch.int32))
    raise TypeError(left.dtype)


def _all_bf16_values() -> torch.Tensor:
    return torch.arange(2**16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_square_matches_aten_for_every_bf16_bit_pattern() -> None:
    values = _all_bf16_values().contiguous()
    expected = values.float().pow(2)
    result = sana_rms._square_bf16_to_fp32(values)

    assert _raw_equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_affine_tail_preserves_bf16_rounding_boundaries() -> None:
    values = _all_bf16_values().reshape(1, 256, 256).contiguous()
    inverse_rms = torch.linspace(0.25, 1.75, 256, device="cuda").reshape(1, 256, 1)
    weight = _all_bf16_values().roll(137)[:256].contiguous()

    expected = (values * inverse_rms).to(torch.bfloat16) * weight
    result = sana_rms._rms_norm_affine_tail(values, inverse_rms, weight)

    assert _raw_equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("shape", [(1, 1024, 2240), (1, 1025, 2240), (2, 2640, 2240)])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_exact_sana_rms_norm_matches_production_shapes(shape: tuple[int, ...], eps: float) -> None:
    torch.manual_seed(17)
    hidden_states = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, weight, eps)

    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm(hidden_states, weight, eps)

    assert _raw_equal(result, expected)
    assert sana_rms._signature(hidden_states) in sana_rms._VERIFIED_SIGNATURES
    assert sana_rms._signature(hidden_states) not in sana_rms._DISABLED_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_small_text_shape_stays_on_eager_path(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((2, 300, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, weight, 1e-5)

    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm",
        lambda *_args: pytest.fail("small text RMSNorm must not launch the fast path"),
    )
    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    assert _raw_equal(result, expected)
    assert not sana_rms._VERIFIED_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_mismatch_disables_signature_and_returns_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, weight, 1e-5)
    calls = 0

    def incorrect(*_args) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm", incorrect)
    with torch.no_grad():
        first = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)
        second = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    assert _raw_equal(first, expected)
    assert _raw_equal(second, expected)
    assert calls == 1
    assert sana_rms._signature(hidden_states) in sana_rms._DISABLED_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_launch_failure_disables_signature_and_returns_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, weight, 1e-5)
    calls = 0

    def fail(*_args) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm", fail)
    with torch.no_grad():
        first = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)
        second = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    assert _raw_equal(first, expected)
    assert _raw_equal(second, expected)
    assert calls == 1
    assert sana_rms._signature(hidden_states) in sana_rms._DISABLED_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_verified_signature_skips_reference_and_runs_during_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        expected = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)
    assert sana_rms._signature(hidden_states) in sana_rms._VERIFIED_SIGNATURES

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        sana_rms,
        "_eager_sana_rms_norm",
        lambda *_args: pytest.fail("a verified signature must not repeat eager validation"),
    )
    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    assert _raw_equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_verified_signature_supports_cuda_graph_capture_and_replay() -> None:
    torch.manual_seed(31)
    static_input = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        sana_rms.exact_sana_rms_norm(static_input, weight, 1e-5)
    assert sana_rms._signature(static_input) in sana_rms._VERIFIED_SIGNATURES
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        result = sana_rms.exact_sana_rms_norm(static_input, weight, 1e-5)

    replacement = torch.randn_like(static_input)
    static_input.copy_(replacement)
    graph.replay()
    torch.accelerator.synchronize()

    expected = _reference(replacement, weight, 1e-5)
    assert _raw_equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_unverified_capture_uses_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, weight, 1e-5)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm",
        lambda *_args: pytest.fail("an unverified signature must not launch during capture"),
    )
    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    assert _raw_equal(result, expected)
    assert not sana_rms._VERIFIED_SIGNATURES
    assert not sana_rms._DISABLED_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("case", ["fp16", "mixed_weight", "noncontiguous", "grad_enabled"])
def test_unsupported_inputs_use_eager(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if case == "fp16":
        hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.float16)
        weight = torch.randn(2240, device="cuda", dtype=torch.float16)
    elif case == "mixed_weight":
        hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2240, device="cuda", dtype=torch.float32)
    elif case == "noncontiguous":
        hidden_states = torch.randn((1, 2240, 1024), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    else:
        hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)

    expected = _reference(hidden_states, weight, 1e-5)
    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm",
        lambda *_args: pytest.fail("unsupported inputs must not launch the fast path"),
    )

    if case == "grad_enabled":
        result = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)
    else:
        with torch.no_grad():
            result = sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    assert result.dtype == expected.dtype
    if result.dtype is torch.bfloat16:
        assert _raw_equal(result, expected)
    else:
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sana_module_compile_fallback_matches_reference() -> None:
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaRMSNorm

    torch.manual_seed(23)
    module = SanaRMSNorm(2240, eps=1e-5).to(device="cuda", dtype=torch.bfloat16).eval()
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, module.weight, module.eps)
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    with torch.no_grad():
        result = compiled(hidden_states)

    assert _raw_equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_sana_module_routes_no_bias_affine_norm_to_fast_path() -> None:
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaRMSNorm

    torch.manual_seed(29)
    module = SanaRMSNorm(2240, eps=1e-5).to(device="cuda", dtype=torch.bfloat16).eval()
    hidden_states = torch.randn((1, 1025, 2240), device="cuda", dtype=torch.bfloat16)
    expected = _reference(hidden_states, module.weight, module.eps)

    with torch.no_grad():
        result = module(hidden_states)

    assert _raw_equal(result, expected)
    assert sana_rms._signature(hidden_states) in sana_rms._VERIFIED_SIGNATURES
