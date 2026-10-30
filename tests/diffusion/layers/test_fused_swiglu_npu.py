# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.fused_swiglu import FusedSwiGLU, fused_swiglu_available

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.parametrize("tokens", [17, 4096])
def test_eagle_ffn_real_shape(tokens):
    pytest.importorskip("torch_npu")
    if not fused_swiglu_available():
        pytest.skip("Requires A5 and the public MindIE-SD Eagle FFN API and compiled operator")
    torch.manual_seed(42)
    # Gate-first packed weights match the public Eagle API, unlike some
    # standalone SwiGLU APIs that accept [up, gate].
    gate = torch.randn(12288, 4096, device="npu", dtype=torch.bfloat16) / 4096**0.5
    up = torch.randn_like(gate) / 4096**0.5
    down = torch.randn(4096, 12288, device="npu", dtype=torch.bfloat16) / 12288**0.5
    x = torch.randn(1, tokens, 4096, device="npu", dtype=torch.bfloat16)
    packed = torch.cat((gate, up), dim=0)
    with torch.inference_mode():
        expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        actual = FusedSwiGLU()(x, packed, down)
    torch.npu.synchronize()
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2.item() < 0.01
