# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_grouped_mm_matches_per_expert_reference() -> None:
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm is unavailable")

    torch.manual_seed(42)
    block = module.LingBotVideoSparseMoeBlock(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=8,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=None,
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        block.experts.w1.normal_(mean=0.0, std=0.02)
        block.experts.w2.normal_(mean=0.0, std=0.02)
        block.experts.w3.normal_(mean=0.0, std=0.02)
    tokens = torch.randn(23, 16, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([5, 0, 9, 9], device="cuda", dtype=torch.int64)

    expected = block._run_experts_for_loop(tokens, counts)
    actual = block._run_grouped_experts(tokens, counts)

    torch.testing.assert_close(actual, expected)
