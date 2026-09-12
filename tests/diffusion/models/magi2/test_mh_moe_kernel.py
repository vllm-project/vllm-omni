# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2 import mh_moe

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
def test_expert_kernel_steps_down_from_oversized_blackwell_tile() -> None:
    """The released MAGI-2 dimensions must fit after the preferred tile fails."""

    tokens, d_head, d_expert = 1, 256, 1280
    device = "cuda"
    x = torch.randn(tokens, 1, d_head, device=device, dtype=torch.bfloat16)
    gather_ids = torch.arange(tokens, device=device, dtype=torch.int32)
    probs = torch.ones(tokens, device=device, dtype=torch.float32)
    expert_offsets = torch.tensor([0, tokens], device=device, dtype=torch.int64)
    w_gate = torch.randn(1, d_head, d_expert, device=device, dtype=torch.bfloat16)
    w_up = torch.randn(1, d_head, d_expert, device=device, dtype=torch.bfloat16)
    w_down = torch.randn(1, d_expert, d_head, device=device, dtype=torch.bfloat16)

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
            )

        assert mh_moe._RESOLVED_BLOCK_CONFIG[(d_head, d_expert)] == (64, 64, 32, 2, 8)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    finally:
        mh_moe._RESOLVED_BLOCK_CONFIG.clear()
