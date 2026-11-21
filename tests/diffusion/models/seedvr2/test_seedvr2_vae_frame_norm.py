# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.seedvr2.vae import _frame_norm, _frame_norm_silu

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the fused path"),
]


# Encoder and decoder ResNet widths, and odd spatial sizes from a 848x480 input.
@pytest.mark.parametrize("shape", [(1, 128, 5, 30, 53), (1, 256, 3, 60, 106), (1, 512, 2, 15, 27)])
def test_cuda_frame_norm_silu_matches_unfused(shape):
    """The fused CUDA path matches the transpose + nn.GroupNorm + SiLU path in FP16."""
    torch.manual_seed(0)
    norm = nn.GroupNorm(32, shape[1], eps=1e-6).cuda().half()
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)

    actual = _frame_norm_silu(norm, x)
    expected = F.silu(_frame_norm(norm, x))

    assert actual.shape == x.shape
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
