# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical and layout contracts for the MAGI-2 native/NPU layers."""

from __future__ import annotations

import importlib

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.common_layers import MultiModalityRMSNorm
from vllm_omni.diffusion.models.magi2.layers.native import ModalityDispatcher
from vllm_omni.diffusion.models.magi2.layers.native.rope import apply_rotary_emb
from vllm_omni.diffusion.models.magi2.layers.npu import rope as npu_rope
from vllm_omni.diffusion.models.magi2.layers.npu.normalization import MultiModalityRMSNorm as NPUMultiModalityRMSNorm

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize("device", ["cpu", "cuda", "musa", "npu"])
def test_hardware_dispatch_retains_native_defaults(monkeypatch: pytest.MonkeyPatch, mocker, device: str) -> None:
    """Select Python interfaces only; no hardware operators or model weights run."""
    from vllm_omni import platforms
    from vllm_omni.diffusion.models.magi2 import common_layers, layers
    from vllm_omni.diffusion.models.magi2.layers import cuda, musa, native, npu
    from vllm_omni.platforms.interface import OmniPlatform

    # Load all leaves before changing the selector so their platform references
    # and CustomOp instances cannot leak a test-only platform into later tests.
    backends = {"cpu": native, "cuda": cuda, "musa": musa, "npu": npu}
    platform = mocker.Mock(spec=OmniPlatform)
    platform.is_cuda.return_value = device == "cuda"
    platform.is_musa.return_value = device == "musa"
    platform.is_npu.return_value = device == "npu"
    try:
        with monkeypatch.context() as context:
            context.setattr(platforms, "current_omni_platform", platform)
            importlib.reload(layers)
            assert layers.ModalityDispatcher is common_layers.ModalityDispatcher
            assert layers.Magi2GroupedLinear is common_layers.Magi2GroupedLinear
            assert layers.ElementWiseFourierEmbed is common_layers.ElementWiseFourierEmbed
            assert layers.swiglu7 is common_layers.swiglu7
            assert layers.Magi2MultiHeadMoE is backends[device].Magi2MultiHeadMoE
            assert layers.varlen_attention_with_sink is backends[device].varlen_attention_with_sink
            norm = npu.MultiModalityRMSNorm if device == "npu" else native.MultiModalityRMSNorm
            rope = npu.apply_rotary_emb if device == "npu" else native.apply_rotary_emb
            mhc = musa.MHCHandler if device == "musa" else native.MHCHandler
            assert layers.MultiModalityRMSNorm is norm
            assert layers.apply_rotary_emb is rope
            assert layers.MHCHandler is mhc
            handler = layers.MHCHandler(4, 64, sinkhorn_iterations=3, sinkhorn_epsilon=1e-8, dtype=torch.bfloat16)
            assert handler.fp32_stream_contractions == (device == "musa")
            assert handler.sinkhorn_iterations == 3 and handler.sinkhorn_epsilon == 1e-8
            assert handler.dtype == torch.bfloat16
    finally:
        importlib.reload(layers)


def _rms_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    result = x.float()
    return result * torch.rsqrt(result.square().mean(dim=-1, keepdim=True) + eps) * weight.float()


def _rope_reference(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, interleaved: bool = False
) -> torch.Tensor:
    rotary_dim = cos.shape[-1] * 2
    prefix = x[..., :rotary_dim]
    if interleaved:
        first, second = prefix[..., ::2], prefix[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        cos, sin = (value.repeat_interleave(2, dim=-1).unsqueeze(-2) for value in (cos, sin))
    else:
        first, second = prefix.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        cos, sin = (torch.cat((value, value), dim=-1).unsqueeze(-2) for value in (cos, sin))
    return torch.cat((prefix * cos + rotated * sin, x[..., rotary_dim:]), dim=-1)


def _require_npu() -> torch.device:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    return torch.device("npu")


@pytest.mark.cpu
def test_native_norm_scales_in_fp32_before_output_cast() -> None:
    norm = MultiModalityRMSNorm(4, out_dtype=torch.float32)
    x = torch.tensor([[0.125, -0.75, 1.5, 3.0], [-2.0, 0.375, 0.625, 4.0]], dtype=torch.bfloat16)
    gamma = torch.tensor([1.03125, 0.8125, 1.1875, 0.96875])
    with torch.no_grad():
        norm.weight.copy_(gamma - 1.0)
    actual = norm(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, _rms_reference(x, gamma, norm.eps), rtol=0, atol=0)
    norm.out_dtype = None
    torch.testing.assert_close(norm(x), actual.to(x.dtype), rtol=0, atol=0)


@pytest.mark.cpu
def test_multimodality_norm_state_dict_keeps_flat_fp32_zero_delta() -> None:
    norm = MultiModalityRMSNorm(8, num_modality=3, num_patterns=2)
    state = norm.state_dict()
    assert set(state) == {"weight"}
    assert state["weight"].shape == (48,)
    assert state["weight"].dtype == torch.float32
    assert torch.count_nonzero(state["weight"]) == 0


@pytest.mark.cpu
def test_multimodality_norm_keeps_token_order_and_empty_groups() -> None:
    generator = torch.Generator().manual_seed(17)
    norm = MultiModalityRMSNorm(4, num_modality=3, out_dtype=torch.float32)
    delta = torch.randn(3, 4, generator=generator) * 0.1
    with torch.no_grad():
        norm.weight.copy_(delta.flatten())
    x = torch.randn(5, 4, generator=generator).to(torch.bfloat16)
    mapping = torch.tensor([2, 0, 2, 0, 2], dtype=torch.int32)
    dispatcher = ModalityDispatcher(mapping, 3)
    actual = dispatcher.inverse_permute(norm(dispatcher.permute(x), dispatcher))
    expected = _rms_reference(x, delta[mapping.long()] + 1.0, norm.eps)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_multimodality_norm_normalizes_each_pattern() -> None:
    generator = torch.Generator().manual_seed(18)
    norm = MultiModalityRMSNorm(4, num_patterns=2, out_dtype=torch.float32)
    delta = torch.randn(2, 4, generator=generator) * 0.1
    with torch.no_grad():
        norm.weight.copy_(delta.flatten())
    x = torch.randn(3, 2, 4, generator=generator).to(torch.bfloat16)
    actual = norm(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, _rms_reference(x, delta + 1.0, norm.eps), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("batch_dependent", [False, True])
@pytest.mark.parametrize("rotary_dim", [96, 128])
def test_native_rope_keeps_batch_broadcast_and_partial_tail(
    interleaved: bool, batch_dependent: bool, rotary_dim: int
) -> None:
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(2, 5, 3, 128, generator=generator)
    phase_shape = (2, 5, rotary_dim // 2) if batch_dependent else (5, rotary_dim // 2)
    phase = torch.randn(*phase_shape, generator=generator)
    cos, sin = phase.cos(), phase.sin()
    actual = apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    expected = _rope_reference(x, cos, sin, interleaved=interleaved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:])


@pytest.mark.cpu
def test_native_rope_keeps_dtype_promotion() -> None:
    generator = torch.Generator().manual_seed(20)
    x = torch.randn(2, 4, 3, 128, generator=generator).to(torch.bfloat16)
    phase = torch.randn(2, 4, 48, generator=generator)
    actual = apply_rotary_emb(x, phase.cos(), phase.sin())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, _rope_reference(x, phase.cos(), phase.sin()), rtol=0, atol=0)
    assert torch.equal(actual[..., 96:], x[..., 96:].float())


@pytest.mark.cpu
def test_native_rope_keeps_oversized_rotary_dimension_error() -> None:
    with pytest.raises(ValueError, match=r"RoPE dimension 130 exceeds head dimension 128"):
        apply_rotary_emb(torch.empty(1, 2, 3, 128), torch.empty(2, 65), torch.empty(2, 65))


@pytest.mark.cpu
def test_npu_rope_falls_back_for_cpu_tensor() -> None:
    generator = torch.Generator().manual_seed(21)
    x = torch.randn(2, 4, 3, 128, generator=generator)
    phase = torch.randn(2, 4, 48, generator=generator)
    actual = npu_rope.apply_rotary_emb(x, phase.cos(), phase.sin())
    torch.testing.assert_close(actual, apply_rotary_emb(x, phase.cos(), phase.sin()), rtol=0, atol=0)


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_rms_norm_npu_matches_fp32_reference() -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(20)
    x_cpu = torch.randn(17, 64, generator=generator)
    gamma = torch.randn(64, generator=generator) * 0.1 + 1.0
    norm = NPUMultiModalityRMSNorm(64, out_dtype=torch.float32).to(device)
    with torch.no_grad():
        norm.weight.copy_((gamma - 1.0).to(device))
    actual = norm(x_cpu.to(device))
    assert actual.device.type == "npu"
    assert actual.dtype == torch.float32
    assert actual.shape == x_cpu.shape
    torch.testing.assert_close(actual.cpu(), _rms_reference(x_cpu, gamma, norm.eps), rtol=2e-4, atol=2e-4)


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_npu_rope_keeps_batch_cos_dtype_and_partial_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(21)
    x_cpu = torch.randn(2, 5, 4, 128, generator=generator).to(torch.bfloat16)
    phase = torch.randn(2, 5, 48, generator=generator)
    cos_cpu, sin_cpu = phase.cos(), phase.sin()
    x = x_cpu.to(device)
    torch_npu = importlib.import_module("torch_npu")
    real_op = torch_npu.npu_rotary_mul
    calls = []

    def checked_rotary_mul(prefix, op_cos, op_sin, *, rotary_mode):
        assert prefix.shape == (2, 5, 4, 96)
        assert op_cos.shape == op_sin.shape == (2, 5, 1, 96)
        assert prefix.is_contiguous() and op_cos.is_contiguous() and op_sin.is_contiguous()
        assert prefix.dtype == op_cos.dtype == op_sin.dtype == x.dtype
        assert rotary_mode == "half"
        calls.append(True)
        return real_op(prefix, op_cos, op_sin, rotary_mode=rotary_mode)

    monkeypatch.setattr(torch_npu, "npu_rotary_mul", checked_rotary_mul)
    actual = npu_rope.apply_rotary_emb(x, cos_cpu.to(device), sin_cpu.to(device))
    assert calls == [True]
    assert actual.device.type == "npu"
    assert actual.dtype == x.dtype
    assert actual.shape == x.shape
    assert torch.equal(actual[..., 96:], x[..., 96:])
    # BF16 fused evaluation may differ from the CPU formula's evaluation order.
    expected = _rope_reference(x_cpu, cos_cpu.to(x_cpu.dtype), sin_cpu.to(x_cpu.dtype))
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
    with pytest.raises(ValueError, match=r"RoPE dimension 130"):
        npu_rope.apply_rotary_emb(x, torch.empty(2, 5, 65, device=device), torch.empty(2, 5, 65, device=device))


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.parametrize(
    ("shape", "interleaved"),
    [((5, 4, 128), False), ((1, 5, 4, 128), True)],
    ids=["rank-three", "interleaved"],
)
def test_npu_rope_unsupported_layouts_keep_native_semantics(shape: tuple[int, ...], interleaved: bool) -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(22)
    x = torch.randn(*shape, generator=generator).to(torch.bfloat16).to(device)
    phase = torch.randn(shape[-3], 48, generator=generator).to(device)
    cos, sin = phase.cos(), phase.sin()
    actual = npu_rope.apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    expected = apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-4, atol=2e-4)


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_multimodality_npu_norm_handles_an_empty_modality_group() -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(23)
    norm = NPUMultiModalityRMSNorm(8, num_modality=3, out_dtype=torch.float32).to(device)
    delta = torch.randn(3, 8, generator=generator) * 0.1
    with torch.no_grad():
        norm.weight.copy_(delta.flatten().to(device))
    x_cpu = torch.randn(5, 8, generator=generator).to(torch.bfloat16)
    mapping_cpu = torch.tensor([2, 0, 2, 0, 2], dtype=torch.int32)
    x = x_cpu.to(device)
    dispatcher = ModalityDispatcher(mapping_cpu.to(device), 3)
    actual = dispatcher.inverse_permute(norm(dispatcher.permute(x), dispatcher))
    expected = _rms_reference(x_cpu, delta[mapping_cpu.long()] + 1.0, norm.eps)
    assert actual.dtype == torch.float32
    assert actual.shape == x.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_multimodality_npu_norm_pattern_two_uses_native_contract() -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(24)
    norm = NPUMultiModalityRMSNorm(8, num_patterns=2, out_dtype=torch.float32).to(device)
    delta = torch.randn(2, 8, generator=generator) * 0.1
    with torch.no_grad():
        norm.weight.copy_(delta.flatten().to(device))
    x_cpu = torch.randn(7, 2, 8, generator=generator).to(torch.bfloat16)
    actual = norm(x_cpu.to(device))
    assert actual.dtype == torch.float32
    assert actual.shape == x_cpu.shape
    torch.testing.assert_close(actual.cpu(), _rms_reference(x_cpu, delta + 1.0, norm.eps), rtol=2e-4, atol=2e-4)
