# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for SenseNova-U1 RMSNorm and the U1.5 config surface.

``Qwen3RMSNorm`` was a hand-rolled fp32 up/down-cast chain; it now calls the fused
``F.rms_norm``. These pin the mathematics against an independent reference so a
future kernel swap cannot silently change what the layer computes, and pin the two
config fields that distinguish U1.5 from U1.
"""

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import Qwen3RMSNorm
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm written out in float64 so it is independent of the implementation."""
    x64 = x.to(torch.float64)
    var = x64.pow(2).mean(-1, keepdim=True)
    return (weight.to(torch.float64) * (x64 * torch.rsqrt(var + eps))).to(x.dtype)


@pytest.mark.parametrize("hidden", [8, 128, 4096])
@pytest.mark.parametrize(
    "dtype,atol",
    [(torch.float32, 1e-5), (torch.bfloat16, 3e-2), (torch.float16, 4e-3)],
)
def test_rmsnorm_matches_float64_reference(hidden: int, dtype: torch.dtype, atol: float):
    torch.manual_seed(0)
    norm = Qwen3RMSNorm(hidden, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(hidden, dtype=dtype) + 0.5)
    x = torch.randn(7, hidden, dtype=dtype)
    got = norm(x)
    want = _reference(x, norm.weight, norm.variance_epsilon)
    assert got.dtype == dtype
    assert got.shape == x.shape
    torch.testing.assert_close(got.to(torch.float32), want.to(torch.float32), atol=atol, rtol=atol)


def _cast_chain(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The implementation this PR replaced, kept so its accuracy can be compared."""
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return weight * (x.to(torch.float32) * torch.rsqrt(var + eps)).to(weight.dtype)


@pytest.mark.parametrize("hidden", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_is_closer_to_float64_than_the_cast_chain(hidden: int, dtype: torch.dtype):
    """The cast chain rounds to ``dtype`` before the weight multiply and so rounds
    twice, while ``F.rms_norm`` rounds once. Restoring the cast chain fails here."""
    torch.manual_seed(0)
    norm = Qwen3RMSNorm(hidden, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(hidden, dtype=dtype) + 0.5)
    x = torch.randn(64, hidden, dtype=dtype)

    reference = _reference(x, norm.weight, norm.variance_epsilon).to(torch.float64)
    scale = reference.abs().mean()

    def relative_error(out: torch.Tensor) -> float:
        return ((out.to(torch.float64) - reference).abs().mean() / scale).item()

    fused = relative_error(norm(x))
    legacy = relative_error(_cast_chain(x, norm.weight, norm.variance_epsilon))
    assert fused < legacy, f"{dtype} hidden={hidden}: fused {fused:.3e} not better than {legacy:.3e}"


def test_rmsnorm_is_scale_equivariant():
    """RMSNorm divides by the RMS, so scaling the input must not change the output."""
    torch.manual_seed(0)
    norm = Qwen3RMSNorm(64, eps=1e-12)
    x = torch.randn(4, 64)
    torch.testing.assert_close(norm(x), norm(x * 7.5), atol=1e-4, rtol=1e-4)


def test_rmsnorm_preserves_leading_dims():
    norm = Qwen3RMSNorm(16, eps=1e-6)
    for shape in [(16,), (3, 16), (2, 3, 16)]:
        assert norm(torch.randn(*shape)).shape == shape


def test_rmsnorm_weight_is_applied():
    norm = Qwen3RMSNorm(4, eps=1e-6)
    x = torch.randn(2, 4)
    base = norm(x)
    with torch.no_grad():
        norm.weight.mul_(3.0)
    torch.testing.assert_close(norm(x), base * 3.0, atol=1e-5, rtol=1e-5)


def test_u15_config_fields_default_to_u1_values():
    """U1.5 flips exactly these two; the defaults must stay U1's so old checkpoints
    keep their behaviour when the keys are absent."""
    cfg = SenseNovaU1Config()
    assert cfg.use_pixel_head is False
    assert cfg.noise_scale_max_value == 10.0


def test_u15_config_fields_are_read_from_kwargs():
    cfg = SenseNovaU1Config(use_pixel_head=True, noise_scale_max_value=16.0)
    assert cfg.use_pixel_head is True
    assert cfg.noise_scale_max_value == 16.0
