# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_rope_moves_precomputed_tables_to_runtime_device() -> None:
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    rope = module.LingBotVideoRotaryEmbedding(
        axes_dims=(4, 4, 8),
        axes_lens=(32, 32, 32),
        theta=256.0,
    )
    position_ids = module.make_joint_position_ids(
        text_len=3,
        grid_t=1,
        grid_h=2,
        grid_w=2,
        device=torch.device("cuda"),
    )

    output = rope(position_ids)
    first_data_ptrs = [getattr(rope, f"freqs_cis_{axis}").data_ptr() for axis in range(3)]
    repeated = rope(position_ids)

    assert output.device.type == "cuda"
    assert all(getattr(rope, f"freqs_cis_{axis}").device.type == "cuda" for axis in range(3))
    assert [getattr(rope, f"freqs_cis_{axis}").data_ptr() for axis in range(3)] == first_data_ptrs
    torch.testing.assert_close(repeated, output)


def _eager_sparse_moe_reference(block, hidden_states):
    """Independent eager reference for LingBot grouped top-k routing."""
    batch_size, _, hidden_size = hidden_states.shape
    tokens = hidden_states.reshape(-1, hidden_size)
    runner = block.experts
    routed_experts = runner.routed_experts

    logits = F.linear(tokens.float(), runner.gate.weight.float())
    scores = logits.sigmoid()
    corrected_scores = scores + routed_experts.e_score_correction_bias.unsqueeze(0)

    num_experts = corrected_scores.shape[-1]
    num_groups = 4
    experts_per_group = num_experts // num_groups
    grouped = corrected_scores.view(-1, num_groups, experts_per_group)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_indices = group_scores.topk(1, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_indices, True)
    expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(corrected_scores)
    top_indices = (
        corrected_scores.masked_fill(~expert_mask, float("-inf"))
        .topk(
            2,
            dim=-1,
            sorted=False,
        )
        .indices
    )

    top_scores = scores.gather(1, top_indices)
    top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
    top_scores = top_scores * 1.5

    w13 = routed_experts.w13_weight
    intermediate_size = w13.shape[1] // 2
    w1 = w13[:, :intermediate_size]
    w3 = w13[:, intermediate_size:]
    w2 = routed_experts.w2_weight
    routed = torch.zeros_like(tokens, dtype=torch.float32)
    for expert_idx in range(num_experts):
        token_idx, route_idx = torch.where(top_indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_tokens = tokens[token_idx]
        hidden = F.silu(F.linear(expert_tokens, w1[expert_idx]))
        hidden = hidden * F.linear(expert_tokens, w3[expert_idx])
        expert_output = F.linear(hidden, w2[expert_idx])
        routed.index_add_(
            0,
            token_idx,
            expert_output.float() * top_scores[token_idx, route_idx, None],
        )

    output = routed.to(tokens.dtype).reshape(batch_size, -1, hidden_size)
    shared = F.silu(F.linear(hidden_states, block.shared_experts.gate_proj.weight))
    shared = shared * F.linear(hidden_states, block.shared_experts.up_proj.weight)
    shared = F.linear(shared, block.shared_experts.down_proj.weight)
    return output + shared


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_common_fused_moe_matches_eager_lingbot_reference(init_fake_tp_group) -> None:
    from vllm.forward_context import is_forward_context_available
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    torch.manual_seed(42)
    init_workspace_manager(torch.device("cuda"))
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
        block.experts.gate.weight.normal_(mean=0.0, std=0.02)
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
    with torch.no_grad():
        expected = _eager_sparse_moe_reference(block, hidden_states)
        compiled_forward = torch.compile(block.forward, dynamic=True)
        assert not is_forward_context_available()
        actual = block(hidden_states)
        compiled_actual = compiled_forward(hidden_states)
        assert not is_forward_context_available()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(compiled_actual, actual)
    assert torch.isfinite(actual).all()
