# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the CustomOp platform-dispatch base class."""

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.layers import custom_op
from vllm_omni.diffusion.layers.custom_op import CustomOp

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _NativeOnlyOp(CustomOp):
    """An op that implements only the PyTorch-native path."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


@pytest.fixture
def mock_platform(monkeypatch):
    def select(backend):
        platform = Mock(spec=["is_rocm", "is_cuda", "is_npu", "is_xpu", "is_musa"])
        for name in ("rocm", "cuda", "npu", "xpu", "musa"):
            getattr(platform, f"is_{name}").return_value = name == backend
        monkeypatch.setattr(custom_op, "current_omni_platform", platform)
        return platform

    return select


@pytest.mark.skipif(
    any(
        getattr(custom_op.current_omni_platform, f"is_{backend}")()
        for backend in ("rocm", "cuda", "npu", "xpu", "musa")
    ),
    reason="Requires the native platform dispatch path",
)
def test_native_platform_dispatch_without_mocks():
    op = _NativeOnlyOp()
    x = torch.randn(4, 8)

    assert torch.equal(op(x), x * 2)


@pytest.mark.parametrize(
    ("backend", "method"),
    [
        ("native", "forward_native"),
        ("cuda", "forward_cuda"),
        ("rocm", "forward_hip"),
        ("npu", "forward_npu"),
        ("xpu", "forward_xpu"),
        ("musa", "forward_musa"),
    ],
)
def test_constructor_dispatch_and_argument_forwarding(mock_platform, monkeypatch, backend, method):
    platform = mock_platform(backend)
    methods = {}
    for name in ("native", "cuda", "hip", "npu", "xpu", "musa"):
        spy = Mock(return_value=object())
        monkeypatch.setattr(_NativeOnlyOp, f"forward_{name}", spy)
        methods[f"forward_{name}"] = spy

    op = _NativeOnlyOp()
    for name in ("rocm", "cuda", "npu", "xpu", "musa"):
        getattr(platform, f"is_{name}").side_effect = AssertionError("Dispatch must happen at construction")

    x, residual = torch.randn(4, 8), torch.randn(4, 8)
    metadata = {"scale": 2}
    for scale in (0.5, 2.0):
        result = op(x, residual, scale=scale, metadata=metadata)
        assert result is methods[method].return_value
        args, kwargs = methods[method].call_args
        assert args[0] is x and args[1] is residual
        assert kwargs == {"scale": scale, "metadata": metadata}
        assert kwargs["metadata"] is metadata
    assert methods[method].call_count == 2
    for name, spy in methods.items():
        if name != method:
            spy.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "fallback"),
    [("xpu", "forward_native"), ("rocm", "forward_cuda"), ("musa", "forward_cuda")],
)
def test_inherited_backend_fallback(mock_platform, monkeypatch, backend, fallback):
    mock_platform(backend)
    spy = Mock(return_value=object())
    monkeypatch.setattr(_NativeOnlyOp, fallback, spy)
    op = _NativeOnlyOp()
    x = torch.randn(4, 8)
    assert op(x, scale=0.5) is spy.return_value
    args, kwargs = spy.call_args
    assert spy.call_count == 1
    assert args[0] is x
    assert kwargs == {"scale": 0.5}


@pytest.mark.parametrize(
    ("backend", "method", "fallback"),
    [
        ("xpu", "forward_xpu", "forward_native"),
        ("rocm", "forward_hip", "forward_cuda"),
        ("musa", "forward_musa", "forward_cuda"),
    ],
)
def test_backend_override_precedes_fallback(mock_platform, monkeypatch, backend, method, fallback):
    mock_platform(backend)
    override = Mock(return_value=object())
    native = Mock(side_effect=AssertionError("Backend override must bypass fallback"))
    monkeypatch.setattr(_NativeOnlyOp, method, override)
    monkeypatch.setattr(_NativeOnlyOp, fallback, native)
    assert _NativeOnlyOp()(torch.randn(4, 8)) is override.return_value
    override.assert_called_once()
    native.assert_not_called()


@pytest.mark.parametrize("backend", ["cuda", "rocm", "npu", "musa"])
def test_missing_backend_implementation_raises(mock_platform, backend):
    mock_platform(backend)
    op = _NativeOnlyOp()
    with pytest.raises(NotImplementedError):
        op(torch.randn(4, 8))


def test_forward_xpu_falls_back_to_native():
    """XPU has no bespoke kernels, so the base class must reuse forward_native.

    Without this fallback every CustomOp subclass has to define an identical
    forward_xpu, and any that forgets crashes with NotImplementedError on XPU.
    """
    op = _NativeOnlyOp()
    x = torch.randn(4, 8)

    assert torch.equal(op.forward_xpu(x), op.forward_native(x))


def test_forward_cuda_still_requires_an_implementation():
    """The native fallback must not mask a missing CUDA kernel."""
    op = _NativeOnlyOp()

    with pytest.raises(NotImplementedError):
        op.forward_cuda(torch.randn(4, 8))


def test_mot_rmsnorm_dispatches_on_xpu_without_an_override(mock_platform):
    """MoTRMSNorm relies on the inherited fallback (regression test)."""
    from vllm_omni.diffusion.models.bagel.mot.mot_layernorm import MoTRMSNorm

    assert "forward_xpu" not in MoTRMSNorm.__dict__

    mock_platform("xpu")
    norm = MoTRMSNorm(16)
    with torch.no_grad():
        norm.gen_weight.fill_(2.0)
    x = torch.randn(8, 16)
    text_indices = torch.arange(0, 4)
    vae_indices = torch.arange(4, 8)

    assert torch.equal(norm(x), norm.forward_native(x))
    assert torch.equal(
        norm(x, text_indices=text_indices, vae_indices=vae_indices),
        norm.forward_native(x, text_indices, vae_indices),
    )
