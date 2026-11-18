# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""fused_gelu_quant: GELU + MXFP8 quantization in one kernel for an MXFP8 linear consumer."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers import fused_gelu_quant as fgq
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"xpu": "B60"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_xpu(), reason="XPU fused GELU+MXFP8 kernel"),
]


class _Consumer(torch.nn.Module):
    def __init__(self, key, block_major):
        super().__init__()
        self._input_quant_key = key
        self.mxfp8_block_major_a_scale = block_major


def _dequant(out, scales, block_major):
    s = scales.to(torch.float32)
    if block_major:
        s = s.contiguous()
    return (out.to(torch.float32).view(out.shape[0], -1, 32) * s[:, :, None]).view(out.shape)


@pytest.mark.parametrize("block_major", [False, True])
@pytest.mark.parametrize("approximate", ["tanh", "none"])
def test_fused_matches_unfused(block_major, approximate):
    if not fgq._FUSED_GELU_MXFP8:
        pytest.skip("fused_gelu_mxfp8_quant op not present")
    torch.manual_seed(0)
    x = torch.randn(3, 257, 13824, dtype=torch.bfloat16, device="xpu") * 3
    qa = fgq.fused_gelu_quant(_Consumer(fgq.kMxfp8Dynamic, block_major), x, approximate)
    assert not isinstance(qa, torch.Tensor)
    assert qa.data.shape == (3 * 257, 13824) and qa.data.dtype == torch.float8_e4m3fn
    assert qa.orig_shape == x.shape and qa.quant_key == fgq.kMxfp8Dynamic
    assert qa.scale.shape == (3 * 257, 13824 // 32)
    assert qa.scale.t().is_contiguous() if block_major else qa.scale.is_contiguous()
    ref = torch.nn.functional.gelu(x.float(), approximate=approximate).reshape(-1, 13824)
    got = _dequant(qa.data, qa.scale, block_major)
    # MXFP8 (e4m3, 3 mantissa bits) with a per-32 e8m0 scale: relative error per element <= 2^-4 of the block max.
    blockmax = ref.abs().view(ref.shape[0], -1, 32).amax(-1, keepdim=True).expand(-1, -1, 32).reshape(ref.shape)
    assert ((got - ref).abs() <= blockmax / 16 + 1e-6).all()


def test_falls_back_without_mx_consumer():
    x = torch.randn(2, 64, 5120, dtype=torch.bfloat16, device="xpu")
    y = fgq.fused_gelu_quant(_Consumer(None, False), x, "tanh")
    assert isinstance(y, torch.Tensor)
    torch.testing.assert_close(y, torch.nn.functional.gelu(x, approximate="tanh"))
