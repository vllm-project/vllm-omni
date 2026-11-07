# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import swiglu

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A2"}, num_cards=1)
def test_native_bf16_swiglu_matches_fp32_activation() -> None:
    # Cover every finite BF16 gate value, including subnormals and signed zero.
    gates = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gates = gates[torch.isfinite(gates)].to("npu")
    up = torch.full_like(gates, -0.625)
    expected = F.silu(gates.float()).to(gates.dtype) * up
    actual = swiglu(gates, up)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_swiglu_cpu_keeps_fp32_activation(dtype: torch.dtype) -> None:
    gates = torch.tensor([[-10, -1, 0, 0.25, 10]], dtype=dtype)
    up = torch.tensor([[2, -3, 4, 0.125, 0.5]], dtype=dtype)
    expected = F.silu(gates.float()).to(gates.dtype) * up

    torch.testing.assert_close(swiglu(gates, up), expected, rtol=0, atol=0)
