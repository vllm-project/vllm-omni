# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2 import mh_moe

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
@pytest.mark.parametrize("tokens", [1, 65])
@pytest.mark.parametrize("deterministic", [False, True])
def test_expert_kernel_steps_down_from_oversized_blackwell_tile(tokens: int, deterministic: bool) -> None:
    """The smaller tile must fit and match Torch at the released MAGI-2 dimensions."""

    torch.manual_seed(0)
    d_head, d_expert = 256, 1280
    device = "cuda"
    x = torch.randn(tokens, 1, d_head, device=device, dtype=torch.bfloat16)
    gather_ids = torch.arange(tokens, device=device, dtype=torch.int32)
    probs = torch.ones(tokens, device=device, dtype=torch.float32)
    expert_offsets = torch.tensor([0, tokens], device=device, dtype=torch.int64)
    # Keep activations and outputs at unit scale for a meaningful BF16 tolerance.
    w_gate = torch.randn(1, d_head, d_expert, device=device, dtype=torch.bfloat16) / d_head**0.5
    w_up = torch.randn(1, d_head, d_expert, device=device, dtype=torch.bfloat16) / d_head**0.5
    w_down = torch.randn(1, d_expert, d_head, device=device, dtype=torch.bfloat16) / d_expert**0.5

    mh_moe._RESOLVED_BLOCK_CONFIG.clear()
    try:
        with torch.inference_mode():
            output = mh_moe.triton_mh_moe_forward(
                x,
                gather_ids,
                probs,
                expert_offsets,
                w_gate,
                w_up,
                w_down,
                deterministic=deterministic,
            )
            expected = mh_moe.torch_mh_moe_forward(
                x,
                gather_ids,
                probs,
                expert_offsets,
                w_gate,
                w_up,
                w_down,
            )

        assert mh_moe._RESOLVED_BLOCK_CONFIG[(d_head, d_expert)] == (64, 64, 32, 2, 8)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        # Torch rounds the gate/up projections to BF16; Triton retains FP32
        # accumulators until the fused activation, so bitwise equality is not expected.
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    finally:
        mh_moe._RESOLVED_BLOCK_CONFIG.clear()
