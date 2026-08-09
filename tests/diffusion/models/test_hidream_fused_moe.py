# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_omni.diffusion.models.hidream_image.hidream_image_transformer as hidream

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _reference_fused_experts(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
):
    output = torch.zeros_like(hidden_states)
    intermediate_size = w1.shape[1] // 2
    for token_idx in range(hidden_states.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_idx = int(topk_ids[token_idx, route_idx])
            gate_up = F.linear(hidden_states[token_idx], w1[expert_idx])
            expert_output = F.linear(
                F.silu(gate_up[:intermediate_size]) * gate_up[intermediate_size:],
                w2[expert_idx],
            )
            output[token_idx] += expert_output * topk_weights[token_idx, route_idx]
    return output


class _ReferenceRoutedExperts(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2_weight = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))


class _ReferenceFusedMoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool,
        custom_routing_function,
        pcp_size: int,
        prefix: str,
    ):
        super().__init__()
        del renormalize, prefix
        self.pcp_size = pcp_size
        self.custom_routing_function = custom_routing_function
        self.top_k = top_k
        self.routed_experts = _ReferenceRoutedExperts(num_experts, intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=False,
        )
        return _reference_fused_experts(
            hidden_states,
            self.routed_experts.w13_weight,
            self.routed_experts.w2_weight,
            topk_weights,
            topk_ids,
        )


class _TestPlatform:
    def __init__(self, *, cuda: bool):
        self.cuda = cuda

    def is_cuda(self) -> bool:
        return self.cuda


def test_hidream_packed_fused_moe_matches_native_without_duplicate_weights(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(hidream, "current_omni_platform", _TestPlatform(cuda=False))
    native = hidream.MOEFeedForwardSwiGLU(
        dim=8,
        hidden_dim=12,
        num_routed_experts=4,
        num_activated_experts=2,
        _force_inference_output=True,
    ).eval()
    assert isinstance(native.experts, nn.ModuleList)

    monkeypatch.setattr(hidream, "current_omni_platform", _TestPlatform(cuda=True))
    monkeypatch.setattr(hidream, "FusedMoE", _ReferenceFusedMoE)
    packed = hidream.MOEFeedForwardSwiGLU(
        dim=8,
        hidden_dim=12,
        num_routed_experts=4,
        num_activated_experts=2,
        _force_inference_output=True,
    ).eval()
    x = torch.randn(2, 5, 8)

    packed.gate.load_state_dict(native.gate.state_dict())
    packed.shared_experts.load_state_dict(native.shared_experts.state_dict())
    routed_experts = packed.experts.routed_experts
    with torch.no_grad():
        routed_experts.w13_weight.copy_(
            torch.stack([torch.cat((expert.w1.weight, expert.w3.weight)) for expert in native.experts])
        )
        routed_experts.w2_weight.copy_(torch.stack([expert.w2.weight for expert in native.experts]))

    expected = native(x)
    actual = packed(x)

    native_routed_numel = sum(parameter.numel() for expert in native.experts for parameter in expert.parameters())
    packed_routed_numel = sum(parameter.numel() for parameter in routed_experts.parameters())
    assert packed._use_fused_moe
    assert packed_routed_numel == native_routed_numel
    assert packed.experts.pcp_size == 1
    assert not hasattr(packed, "_fused_w13")
    assert not hasattr(packed, "_fused_w2")
    torch.testing.assert_close(actual, expected)


class _WeightLoaderRoutedExperts(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.w13_weight = nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size, hidden_size))
        self.w2_weight = nn.Parameter(torch.zeros(num_experts, hidden_size, intermediate_size))
        self.w13_weight.weight_loader = self._load_weight
        self.w2_weight.weight_loader = self._load_weight

    def _load_weight(
        self,
        param,
        loaded_weight,
        weight_name,
        *,
        shard_id,
        expert_id,
        return_success,
    ):
        del weight_name
        if shard_id == "w1":
            param.data[expert_id, : self.intermediate_size].copy_(loaded_weight)
        elif shard_id == "w3":
            param.data[expert_id, self.intermediate_size :].copy_(loaded_weight)
        else:
            param.data[expert_id].copy_(loaded_weight)
        return return_success


def test_hidream_packed_expert_checkpoint_mapping(monkeypatch):
    model = hidream.HiDreamImageTransformer2DModel.__new__(hidream.HiDreamImageTransformer2DModel)
    nn.Module.__init__(model)
    model.num_routed_experts = 4

    block_wrapper = nn.Module()
    block_wrapper.block = nn.Module()
    block_wrapper.block.ff_i = nn.Module()
    block_wrapper.block.ff_i.experts = nn.Module()
    routed_experts = _WeightLoaderRoutedExperts(num_experts=4, intermediate_size=6, hidden_size=8)
    block_wrapper.block.ff_i.experts.routed_experts = routed_experts
    model.double_stream_blocks = nn.ModuleList([block_wrapper])

    monkeypatch.setattr(hidream, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(hidream, "get_tensor_model_parallel_world_size", lambda: 1)
    weights = []
    expected = {}
    for expert_id in range(4):
        for shard_name, shape in (("w1", (6, 8)), ("w2", (8, 6)), ("w3", (6, 8))):
            tensor = torch.full(
                shape,
                expert_id * 10 + {"w1": 1, "w2": 2, "w3": 3}[shard_name],
                dtype=torch.float32,
            )
            name = f"double_stream_blocks.0.block.ff_i.experts.{expert_id}.{shard_name}.weight"
            weights.append((name, tensor))
            expected[(expert_id, shard_name)] = tensor

    loaded = model.load_weights(weights)

    assert {name for name, _ in weights}.issubset(loaded)
    for expert_id in range(4):
        torch.testing.assert_close(
            routed_experts.w13_weight[expert_id, :6],
            expected[(expert_id, "w1")],
        )
        torch.testing.assert_close(
            routed_experts.w13_weight[expert_id, 6:],
            expected[(expert_id, "w3")],
        )
        torch.testing.assert_close(routed_experts.w2_weight[expert_id], expected[(expert_id, "w2")])
