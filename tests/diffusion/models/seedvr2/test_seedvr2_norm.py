# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.diffusion.models.seedvr2.nadit import RMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("affine", [False, True])
def test_rms_norm_large_finite_inputs(dtype: torch.dtype, affine: bool) -> None:
    # All inputs are finite in FP16, but squaring an outlier in FP16 yields
    # infinity and silently zeros the entire normalized token.
    x = torch.ones(2, 3, 2560, dtype=dtype)
    x[0, 0, 0] = 256
    x[0, 1, 0] = 300
    x[0, 2, 0] = 60000
    x[1, 0].fill_(300)
    x[1, 1].zero_()
    x[1, 2].fill_(0.0001)
    norm = RMSNorm(2560, eps=1e-5, elementwise_affine=affine).to(dtype)
    if affine:
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 2560, dtype=dtype))
    high_precision = x.double()
    expected = (high_precision / (high_precision.square().mean(-1, keepdim=True) + norm.eps).sqrt()).to(dtype)
    if affine:
        expected = expected * norm.weight
    actual = norm(x)
    assert actual.dtype == dtype
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected)
