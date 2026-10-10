# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.models.normalization import RMSNorm

from vllm_omni.diffusion.models.qwen_image_21.qk_norm import qk_rotary

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_cpu_declines_fused_reduction():
    q = torch.randn(1, 3, 2, 128)
    assert qk_rotary(q, q, torch.ones(128), torch.ones(128), torch.ones(3, 64, dtype=torch.complex64), 1e-6) is None


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("sequence", [880, 911, 913, 914])
@torch.no_grad()
def test_strided_qkv_matches_original_rms_and_complex_rope(sequence):
    torch.manual_seed(sequence)
    projection = torch.randn(2, sequence, 3, 32, 128, device="cuda", dtype=torch.bfloat16) * 3
    q, k = projection[:, :, 0], projection[:, :, 1]
    nq, nk = (RMSNorm(128, eps=1e-6).to("cuda", torch.bfloat16) for _ in range(2))
    nq.weight.copy_(torch.randn_like(nq.weight) * 0.1 + 1)
    nk.weight.copy_(torch.randn_like(nk.weight) * 0.1 + 1)
    freqs = torch.polar(torch.ones(sequence, 64, device="cuda"), torch.randn(sequence, 64, device="cuda"))

    def reference(x, norm):
        normalized = norm(x)
        return (
            torch.view_as_real(
                torch.view_as_complex(normalized.float().reshape(2, sequence, 32, 64, 2)) * freqs.unsqueeze(1)
            )
            .flatten(3)
            .to(x.dtype)
        )

    result = qk_rotary(q, k, nq.weight, nk.weight, freqs, 1e-6)
    assert result is not None
    torch.testing.assert_close(result[0], reference(q, nq), rtol=0, atol=0)
    torch.testing.assert_close(result[1], reference(k, nk), rtol=0, atol=0)
    assert qk_rotary(q, k, nq.weight.float(), nk.weight, freqs, 1e-6) is None
    strided_weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)[::2]
    assert qk_rotary(q, k, strided_weight, nk.weight, freqs, 1e-6) is None
