# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the CustomOp platform-dispatch base class."""

import pytest
import torch

from vllm_omni.diffusion.layers.custom_op import CustomOp

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _NativeOnlyOp(CustomOp):
    """An op that implements only the PyTorch-native path."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


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


def test_mot_rmsnorm_dispatches_on_xpu_without_an_override():
    """MoTRMSNorm relies on the inherited fallback (regression test)."""
    from vllm_omni.diffusion.models.bagel.mot.mot_layernorm import MoTRMSNorm

    assert "forward_xpu" not in MoTRMSNorm.__dict__

    norm = MoTRMSNorm(16)
    x = torch.randn(8, 16)
    text_indices = torch.arange(0, 4)
    vae_indices = torch.arange(4, 8)

    assert torch.equal(norm.forward_xpu(x), norm.forward_native(x))
    assert torch.equal(
        norm.forward_xpu(x, text_indices, vae_indices),
        norm.forward_native(x, text_indices, vae_indices),
    )


def test_external_registration_preserves_constructor_and_builtin_dispatch(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.layers import custom_op

    platform = SimpleNamespace(
        **{name: lambda: False for name in ("is_rocm", "is_cuda", "is_npu", "is_xpu", "is_musa")},
        is_out_of_tree=lambda: True,
    )
    monkeypatch.setattr(custom_op, "current_omni_platform", platform)
    monkeypatch.setattr(CustomOp, "_oot_registry", {})

    class WeightedOp(_NativeOnlyOp):
        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(width))

        def forward_native(self, x):
            return x * self.weight

    @WeightedOp.register_oot
    class ExternalOp(WeightedOp):
        def forward_oot(self, x):
            return super().forward_native(x) + 1

    external = WeightedOp(3)
    assert isinstance(external, ExternalOp)
    weights = torch.arange(3, dtype=torch.float32)
    external.load_state_dict({"weight": weights}, strict=True)
    torch.testing.assert_close(external(torch.ones(3)), weights + 1)
    torch.testing.assert_close(external.forward_native(torch.ones(3)), weights)
    torch.testing.assert_close(_NativeOnlyOp()(torch.ones(3)), torch.full((3,), 2.0))
    unrelated = type("WeightedOp", (_NativeOnlyOp,), {})
    assert type(unrelated()) is unrelated
    monkeypatch.setattr(platform, "is_out_of_tree", lambda: False)
    builtin = WeightedOp(3)
    assert type(builtin) is WeightedOp
    torch.testing.assert_close(builtin(torch.ones(3)), torch.ones(3))
    with pytest.raises(ValueError, match="already registered"):
        WeightedOp.register_oot(ExternalOp)
    with pytest.raises(TypeError, match="must inherit"):
        _NativeOnlyOp.register_oot(torch.nn.Linear)
