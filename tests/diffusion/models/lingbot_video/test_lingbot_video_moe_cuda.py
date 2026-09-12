# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def _single_rank_model_parallel(unused_tcp_port):
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_distributed_init_method

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=get_distributed_init_method(
            "127.0.0.1",
            unused_tcp_port,
        ),
        backend="nccl",
    )
    initialize_model_parallel()
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _eager_sparse_moe_reference(block, hidden_states, padding_mask):
    """Independent eager oracle for LingBot routing and packed expert weights."""
    batch_size, _, hidden_size = hidden_states.shape
    tokens = hidden_states.reshape(-1, hidden_size)
    valid_indices = torch.where(padding_mask.bool())[0]
    valid_tokens = tokens.index_select(0, valid_indices)
    runner = block.experts
    routed_experts = runner.routed_experts
    router = runner.router

    logits = F.linear(valid_tokens.float(), runner.gate.weight.float())
    scores = logits.sigmoid()
    corrected_scores = scores + routed_experts.e_score_correction_bias.unsqueeze(0)
    grouped = corrected_scores.view(
        -1,
        router.num_expert_group,
        corrected_scores.shape[-1] // router.num_expert_group,
    )
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_indices = group_scores.topk(
        router.topk_group,
        dim=-1,
        sorted=False,
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_indices, True)
    expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(corrected_scores)
    top_indices = (
        corrected_scores.masked_fill(
            ~expert_mask,
            float("-inf"),
        )
        .topk(
            router.top_k,
            dim=-1,
            sorted=False,
        )
        .indices
    )

    top_scores = scores.gather(1, top_indices)
    if router.renormalize:
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
    top_scores = top_scores * router.routed_scaling_factor

    w13 = routed_experts.w13_weight
    intermediate_size = w13.shape[1] // 2
    w1 = w13[:, :intermediate_size]
    w3 = w13[:, intermediate_size:]
    w2 = routed_experts.w2_weight
    valid_output = torch.zeros_like(valid_tokens, dtype=torch.float32)
    for expert_idx in range(block.num_experts):
        token_idx, route_idx = torch.where(top_indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_tokens = valid_tokens[token_idx]
        hidden = F.silu(F.linear(expert_tokens, w1[expert_idx]))
        hidden = hidden * F.linear(expert_tokens, w3[expert_idx])
        expert_output = F.linear(hidden, w2[expert_idx])
        valid_output.index_add_(
            0,
            token_idx,
            expert_output.float() * top_scores[token_idx, route_idx, None],
        )

    output = tokens.new_zeros(tokens.shape)
    output.index_copy_(0, valid_indices, valid_output.to(tokens.dtype))
    output = output.reshape(batch_size, -1, hidden_size)
    shared = F.silu(F.linear(hidden_states, block.shared_experts.gate_proj.weight))
    shared = shared * F.linear(hidden_states, block.shared_experts.up_proj.weight)
    shared = F.linear(shared, block.shared_experts.down_proj.weight)
    return output + shared, top_indices, top_scores


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_common_fused_moe_matches_eager_lingbot_reference(_single_rank_model_parallel) -> None:
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotVideoTransformer3DModel,
    )

    torch.manual_seed(42)
    init_workspace_manager(torch.device("cuda"))
    with set_default_torch_dtype(torch.bfloat16):
        model = LingBotVideoTransformer3DModel(
            patch_size=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            hidden_size=16,
            num_attention_heads=1,
            depth=1,
            intermediate_size=32,
            text_dim=8,
            freq_dim=8,
            axes_dims=(4, 4, 8),
            axes_lens=(32, 32, 32),
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            n_shared_experts=1,
            n_group=4,
            topk_group=1,
            routed_scaling_factor=1.5,
            prefix="test_lingbot_common_fused_moe",
        )
    model.to(device="cuda", dtype=torch.bfloat16)
    block = model.blocks[0].ffn
    with torch.no_grad():
        block.experts.gate.weight.zero_()
        block.experts.routed_experts.e_score_correction_bias.copy_(
            torch.tensor(
                [0.9, 0.8, 1.0, 0.0, 0.7, 0.6, 0.5, 0.4],
                device="cuda",
            )
        )
        for parameter in (
            block.experts.routed_experts.w13_weight,
            block.experts.routed_experts.w2_weight,
            block.shared_experts.gate_proj.weight,
            block.shared_experts.up_proj.weight,
            block.shared_experts.down_proj.weight,
        ):
            parameter.normal_(mean=0.0, std=0.02)
    block.experts.routed_experts.quant_method.process_weights_after_loading(block.experts.routed_experts)

    hidden_states = torch.randn(2, 5, 16, device="cuda", dtype=torch.bfloat16)
    padding_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        device="cuda",
        dtype=torch.float32,
    )

    with torch.no_grad():
        expected, expected_indices, expected_scores = _eager_sparse_moe_reference(
            block,
            hidden_states,
            padding_mask,
        )
        valid_tokens = hidden_states.reshape(-1, 16)[:8]
        router_logits, _ = block.experts.gate(valid_tokens)
        actual_scores, actual_indices = block.experts.router.select_experts(
            hidden_states=valid_tokens,
            router_logits=router_logits,
            topk_indices_dtype=None,
        )
        actual = block(hidden_states, padding_mask=padding_mask)

    unrestricted_indices = torch.topk(
        torch.full((8,), 0.5, device="cuda") + block.experts.routed_experts.e_score_correction_bias,
        k=2,
        sorted=False,
    ).indices

    assert set(actual_indices[0].tolist()) == {0, 1}
    assert set(unrestricted_indices.tolist()) == {0, 2}
    assert torch.equal(actual_indices, expected_indices.to(actual_indices.dtype))
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    assert torch.isfinite(actual).all()
