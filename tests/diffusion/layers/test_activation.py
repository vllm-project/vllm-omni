# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.activation import SiluAndMul

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_silu_and_mul_native_matches_packed_reference() -> None:
    packed = torch.randn(257, 256, dtype=torch.float32)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(SiluAndMul().forward_native(packed), F.silu(gate) * up)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_silu_and_mul_npu_matches_packed_reference() -> None:
    packed = torch.randn(257, 256, device="npu", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(
        SiluAndMul()(packed),
        F.silu(gate) * up,
        atol=2e-2,
        rtol=2e-2,
    )
