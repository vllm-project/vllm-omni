# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNAttentionImpl, VDNLayout
from vllm_omni.diffusion.models.minimax_h3.vdnh3 import _delta_factors

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("XPU is unavailable", allow_module_level=True)


@hardware_test(res={"xpu": "B70"}, num_cards=1)
def test_vdn_forward_xpu_window_is_finite():
    window = VDNLayout(
        used=44, text_len=4, video_start=4, num_frames=5, frame_height=2, frame_width=4, chunk=0, radius=1
    )
    impl = VDNAttentionImpl(num_heads=2, head_size=128, softmax_scale=128**-0.5)
    q = torch.randn(1, 44, 2, 128, dtype=torch.bfloat16, device="xpu")
    out = impl.forward_xpu(q, q, q, AttentionMetadata(extra={"vdn_window": window}))
    torch.xpu.synchronize()
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()


@hardware_test(res={"xpu": "B70"}, num_cards=1)
def test_vdn_delta_factors_xpu_matches_cpu():
    torch.manual_seed(7)
    a = torch.randn(8, 4, 128, 128, dtype=torch.float32)
    a = 0.5 * (a + a.transpose(-1, -2))
    a = a @ a.transpose(-1, -2)
    b = torch.randn(8, 4, 128, 128, dtype=torch.float32)
    alpha = torch.randn(8, 4, 128, dtype=torch.float32)
    cpu_out = _delta_factors(a, b, alpha)
    xpu_out = _delta_factors(a.to("xpu"), b.to("xpu"), alpha.to("xpu"))
    torch.testing.assert_close(xpu_out[0].cpu(), cpu_out[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(xpu_out[1].cpu(), cpu_out[1], rtol=1e-4, atol=1e-4)
