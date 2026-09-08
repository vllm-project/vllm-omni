# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _upstream_sparse_moe_reference(block, hidden_states, padding_mask=None):
    """Independent eager reference for the upstream LingBot sparse MoE path."""
    batch_size, _, hidden_size = hidden_states.shape
    tokens = hidden_states.reshape(-1, hidden_size)

    logits = F.linear(tokens.float(), block.router.weight.float())
    if block.router.score_func == "softmax":
        scores = F.softmax(logits, dim=-1)
    else:
        scores = logits.sigmoid()
    scores_for_choice = scores + block.router.e_score_correction_bias.unsqueeze(0)

    if block.router.n_group is not None and block.router.n_group > 1:
        experts_per_group = block.router.num_experts // block.router.n_group
        grouped = scores_for_choice.view(-1, block.router.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(
            group_scores,
            k=block.router.topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores_for_choice)
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    top_indices = torch.topk(
        scores_for_choice,
        k=block.router.top_k,
        dim=-1,
        sorted=False,
    )[1]
    top_scores = scores.gather(1, top_indices)
    if block.router.top_k > 1 and block.router.norm_topk_prob:
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
    top_scores = top_scores.to(tokens.dtype) * block.router.route_scale

    if padding_mask is not None:
        mask = padding_mask.unsqueeze(-1).to(top_scores.dtype)
        top_scores = top_scores * mask
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)
        top_scores = top_scores * block.router.route_scale

    routed = torch.zeros(
        tokens.shape,
        dtype=torch.float32,
        device=tokens.device,
    )
    for expert_idx in range(block.router.num_experts):
        token_idx, route_idx = torch.where(top_indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_tokens = tokens[token_idx]
        h = F.silu(expert_tokens @ block.experts.w1[expert_idx].transpose(-2, -1))
        h = h * (expert_tokens @ block.experts.w3[expert_idx].transpose(-2, -1))
        expert_output = h @ block.experts.w2[expert_idx].transpose(-2, -1)
        weighted = expert_output.float() * top_scores[token_idx, route_idx, None].float()
        routed.index_add_(0, token_idx, weighted)

    output = routed.to(tokens.dtype).reshape(batch_size, -1, hidden_size)
    if block.shared_experts is not None:
        shared = F.linear(hidden_states, block.shared_experts.gate_proj.weight)
        shared = F.silu(shared) * F.linear(hidden_states, block.shared_experts.up_proj.weight)
        shared = F.linear(shared, block.shared_experts.down_proj.weight)
        output = output + shared
    return output, top_indices, top_scores


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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sparse_moe_block_matches_upstream_reference() -> None:
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm is unavailable")

    torch.manual_seed(42)
    block = module.LingBotVideoSparseMoeBlock(
        hidden_size=16,
        num_experts=8,
        top_k=2,
        moe_intermediate_size=8,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=4,
        topk_group=1,
        routed_scaling_factor=1.5,
        n_shared_experts=1,
    )
    with torch.no_grad():
        block.router.weight.zero_()
        block.router.e_score_correction_bias.copy_(torch.tensor([0.9, 0.8, 1.0, 0.0, 0.7, 0.6, 0.5, 0.4]))
        for parameter in (
            block.experts.w1,
            block.experts.w2,
            block.experts.w3,
            block.shared_experts.gate_proj.weight,
            block.shared_experts.up_proj.weight,
            block.shared_experts.down_proj.weight,
        ):
            parameter.normal_(mean=0.0, std=0.02)
    block = block.to(device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn(2, 5, 16, device="cuda", dtype=torch.bfloat16)
    padding_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        device="cuda",
        dtype=torch.float32,
    )

    with torch.no_grad():
        expected, expected_indices, expected_scores = _upstream_sparse_moe_reference(
            block,
            hidden_states,
            padding_mask,
        )
        actual_indices, actual_scores = block.router(hidden_states.reshape(-1, 16))
        actual_scores = actual_scores * padding_mask.unsqueeze(-1).to(actual_scores.dtype)
        actual_scores = actual_scores / (actual_scores.sum(dim=-1, keepdim=True) + 1e-9)
        actual_scores = actual_scores * block.router.route_scale
        actual = block(hidden_states, padding_mask=padding_mask)

    corrected_scores = (
        torch.full(
            (8,),
            0.5,
            device="cuda",
        )
        + block.router.e_score_correction_bias
    )
    unrestricted_indices = torch.topk(corrected_scores, k=2, sorted=False)[1]

    assert set(actual_indices[0].tolist()) == {0, 1}
    assert set(unrestricted_indices.tolist()) == {0, 2}
    assert torch.equal(actual_indices, expected_indices)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
