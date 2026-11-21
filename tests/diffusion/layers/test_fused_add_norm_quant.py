# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""fused_add_layernorm_quant: gated residual add + LayerNorm/modulation + MXFP8 quant in one kernel."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers import fused_add_norm_quant as fan
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"xpu": "B60"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_xpu(), reason="XPU deepklox fused kernel"),
]


class _Consumer(torch.nn.Module):
    def __init__(self, key, block_major=False):
        super().__init__()
        self._input_quant_key = key
        self.mxfp8_block_major_a_scale = block_major


def _dequant(q, s):
    m, k = q.shape
    return (q.float().view(m, -1, 32) * s.to(torch.float32).view(m, -1, 1)).view(m, k)


def _mx_error_ok(got, ref):
    # MXFP8 (e4m3 payload, per-32 e8m0 scale): error per element within ~2^-3 of the block max.
    blockmax = ref.abs().view(ref.shape[0], -1, 32).amax(-1, keepdim=True).expand(-1, -1, 32).reshape(ref.shape)
    return ((got - ref).abs() <= blockmax / 8 + 1e-6).all()


@pytest.mark.parametrize("block_major", [False, True])
@pytest.mark.parametrize("gated", [True, False])
def test_fused_matches_unfused(block_major, gated):
    if not fan._FUSED_ADD_LN_MXFP8:
        pytest.skip("deepklox fused_add_layer_norm_quant not present")
    torch.manual_seed(0)
    dev, bf = "xpu", torch.bfloat16
    S, K = 513, 5120
    x = torch.randn(1, S, K, device=dev, dtype=bf)
    y = torch.randn(1, S, K, device=dev, dtype=bf)
    gate = torch.randn(1, 1, K, device=dev, dtype=bf) if gated else None
    scale = 0.1 * torch.randn(1, 1, K, device=dev, dtype=bf)
    shift = 0.1 * torch.randn(1, 1, K, device=dev, dtype=bf)
    consumer = _Consumer(fan.kMxfp8Dynamic, block_major)
    assert fan.can_fuse_add_layernorm_quant(consumer, x, gate, scale, shift)

    ref_res = (x + y * gate).type_as(x) if gated else x + y
    ref_out = (
        torch.nn.functional.layer_norm(ref_res.float(), (K,), None, None, 1e-6) * (1 + scale.float()) + shift.float()
    )

    x_in = x.clone()
    res, qa = fan.fused_add_layernorm_quant(consumer, x_in, y, gate, 1 + scale, shift, 1e-6)
    assert res is x_in and res.shape == x.shape
    torch.testing.assert_close(res, ref_res, atol=1 / 16, rtol=1 / 64)  # bf16 rounding order
    assert qa.data.shape == (S, K) and qa.data.dtype == torch.float8_e4m3fn
    assert qa.scale.shape == (S, K // 32) and qa.scale.dtype == torch.float8_e8m0fnu
    assert qa.orig_shape == x.shape and qa.quant_key == fan.kMxfp8Dynamic
    assert qa.scale.t().is_contiguous() if block_major else qa.scale.is_contiguous()
    assert _mx_error_ok(_dequant(qa.data, qa.scale), ref_out.view(S, K))


def test_not_fused_without_mx_consumer_or_per_batch_modulation():
    x = torch.randn(1, 64, 5120, device="xpu", dtype=torch.bfloat16)
    scale = torch.zeros(1, 1, 5120, device="xpu", dtype=torch.bfloat16)
    assert not fan.can_fuse_add_layernorm_quant(_Consumer(None), x, scale)
    assert not fan.can_fuse_layernorm_quant(_Consumer(None), x, scale)
    if fan._FUSED_ADD_LN_MXFP8:
        x2 = torch.randn(2, 64, 5120, device="xpu", dtype=torch.bfloat16)
        per_batch = torch.zeros(2, 1, 5120, device="xpu", dtype=torch.bfloat16)
        assert not fan.can_fuse_add_layernorm_quant(_Consumer(fan.kMxfp8Dynamic), x2, per_batch)
        assert fan.can_fuse_add_layernorm_quant(_Consumer(fan.kMxfp8Dynamic), x2, scale)


@pytest.mark.parametrize("block_major", [False, True])
def test_layernorm_quant_matches_unfused(block_major):
    if not fan._LN_MXFP8:
        pytest.skip("vllm-xpu-kernels layer_norm / per_token_group_fp8_quant not present")
    torch.manual_seed(0)
    dev, bf = "xpu", torch.bfloat16
    S, K = 513, 5120
    x = torch.randn(1, S, K, device=dev, dtype=bf)
    scale = 0.1 * torch.randn(1, 1, K, device=dev, dtype=bf)
    shift = 0.1 * torch.randn(1, 1, K, device=dev, dtype=bf)
    consumer = _Consumer(fan.kMxfp8Dynamic, block_major)
    assert fan.can_fuse_layernorm_quant(consumer, x, scale, shift)
    x_before = x.clone()
    qa = fan.fused_layernorm_quant(consumer, x, 1 + scale, shift, 1e-6)
    assert torch.equal(x, x_before)
    assert qa.data.shape == (S, K) and qa.scale.shape == (S, K // 32)
    assert qa.scale.t().is_contiguous() if block_major else qa.scale.is_contiguous()
    ref = torch.nn.functional.layer_norm(x.float(), (K,), None, None, 1e-6) * (1 + scale.float()) + shift.float()
    assert _mx_error_ok(_dequant(qa.data, qa.scale), ref.view(S, K))
