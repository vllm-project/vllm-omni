# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    from vllm_omni.diffusion.layers.mot.mot_layernorm import MoTRMSNorm

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
