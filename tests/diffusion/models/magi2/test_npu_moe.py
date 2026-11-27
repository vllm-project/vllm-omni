# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend multi-head MoE local kernels against the native expert oracle."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.layers.native.mh_moe import (
    Magi2MultiHeadMoEConfig,
    global_sort_routes,
    torch_mh_moe_forward,
)
from vllm_omni.diffusion.models.magi2.layers.npu import mh_moe as npu_moe
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture
def npu_device():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an NPU device")
    return torch.device("npu")


def _fixed_route_inputs(tokens: int, heads: int, top_k: int):
    generator = torch.Generator().manual_seed(419 + tokens)
    experts, head_dim, intermediate = 4, 64, 96
    x = torch.randn(tokens, heads, head_dim, generator=generator).to(torch.bfloat16)
    shape = (heads * experts, head_dim, intermediate)
    gate = (torch.randn(shape, generator=generator) * 0.3).to(torch.bfloat16)
    up = (torch.randn(shape, generator=generator) * 0.7 - 0.5).to(torch.bfloat16)
    down = (torch.randn(heads * experts, intermediate, head_dim, generator=generator) * 0.2).to(torch.bfloat16)
    # Repeated head-local ids must select each head's independently drawn bank.
    # With top_k <= 2, experts 1 and 3 receive no routes.
    choices = torch.tensor([0, 2])[:top_k] if top_k <= 2 else torch.arange(top_k)
    indices = choices.expand(heads, tokens, top_k).contiguous()
    # Unequal, non-unit probabilities exercise weighted unpermutation. These
    # fractions are exact in BF16, including the NPU kernel's probability cast.
    probabilities = torch.arange(1, top_k + 1, dtype=torch.float32) * 0.125
    probabilities = probabilities.expand(heads, tokens, top_k).contiguous()
    return x, probabilities, indices, gate, up, down


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
@pytest.mark.parametrize(
    ("tokens", "heads", "top_k"),
    [(1, 1, 1), (17, 3, 2), (129, 2, 4)],
    ids=["one-route", "head-banks-and-empty-experts", "all-experts"],
)
def test_npu_mh_moe_forward_matches_native_reference(npu_device, tokens, heads, top_k):
    x, probabilities, indices, gate, up, down = _fixed_route_inputs(tokens, heads, top_k)
    with torch.no_grad():
        # Native BF16 GEMMs round gate/up before SwiGLU7, matching this backend's
        # staged grouped MatMuls. Unpermutation weights and sums routes in FP32;
        # isolate native expert outputs to avoid its BF16 weighted index_add_.
        expected = torch.zeros_like(x, dtype=torch.float32)
        for choice in range(top_k):
            route_probs = torch.ones_like(probabilities[..., choice : choice + 1])
            gather_ids, sorted_probs, offsets = global_sort_routes(
                route_probs, indices[..., choice : choice + 1], num_experts=4
            )
            route = torch_mh_moe_forward(x, gather_ids, sorted_probs, offsets, gate, up, down)
            probability = probabilities[..., choice].transpose(0, 1).unsqueeze(-1).to(x.dtype)
            expected.add_(route.float() * probability.float())
        expected = expected.to(x.dtype)
        actual = npu_moe.npu_mh_moe_forward(
            x.to(npu_device),
            probabilities.to(npu_device),
            indices.to(npu_device),
            4,
            gate.to(npu_device),
            up.to(npu_device),
            down.to(npu_device),
        )

    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "npu"
    # GEMM and top-k accumulation order may differ across CPU and Ascend BF16.
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
def test_npu_local_forward_uses_real_routing_and_grouped_matmuls(npu_device, mocker):
    import torch_npu

    x, _, _, gate, up, down = _fixed_route_inputs(17, 3, 2)
    layer = npu_moe.Magi2MultiHeadMoE(
        Magi2MultiHeadMoEConfig(192, 3, 4, 2, 96, torch.bfloat16),
        ep_group=Magi2ParallelGroup(None, 1, 0),
    ).to(npu_device)
    generator = torch.Generator().manual_seed(113)
    with torch.no_grad():
        layer.gate.copy_(torch.randn(layer.gate.shape, generator=generator).to(npu_device) * 0.1)
        layer.W_gate.copy_(gate.to(npu_device))
        layer.W_up.copy_(up.to(npu_device))
        layer.W_down.copy_(down.to(npu_device))
    route = mocker.spy(layer, "_route")
    forward = mocker.spy(npu_moe, "npu_mh_moe_forward")
    init_routing = mocker.spy(torch_npu, "npu_moe_init_routing_v2")
    grouped_matmul = mocker.spy(torch_npu, "npu_grouped_matmul")
    unpermute = mocker.spy(torch_npu, "npu_moe_token_unpermute")

    with torch.no_grad():
        actual = layer._local_forward(x.to(npu_device))

    route.assert_called_once()
    forward.assert_called_once()
    init_routing.assert_called_once()
    assert grouped_matmul.call_count == 3
    unpermute.assert_called_once()
    assert actual is forward.spy_return
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("invalid_input", "message"),
    [
        ("input_rank", "multi-head MoE input must"),
        ("routing_shape", "top-k probabilities and indices must"),
        ("routing_layout", "top-k routing shape does not match"),
        ("expert_groups", "at most 1024 local expert groups"),
        ("weight_experts", "expected 8 flattened experts"),
    ],
)
def test_npu_mh_moe_rejects_invalid_local_inputs(invalid_input, message):
    x = torch.zeros(2, 2, 64, dtype=torch.bfloat16)
    probabilities = torch.ones(2, 2, 2)
    indices = torch.zeros(2, 2, 2, dtype=torch.long)
    gate = torch.zeros(8, 64, 96, dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    down = torch.zeros(8, 96, 64, dtype=torch.bfloat16)
    experts = 4
    if invalid_input == "input_rank":
        x = x.flatten(1)
    elif invalid_input == "routing_shape":
        probabilities = probabilities[..., :1]
    elif invalid_input == "routing_layout":
        probabilities = torch.ones(2, 3, 2)
        indices = torch.zeros(2, 3, 2, dtype=torch.long)
    elif invalid_input == "expert_groups":
        x = x[:, :1]
        probabilities = probabilities[:1]
        indices = indices[:1]
        experts = 1025
    elif invalid_input == "weight_experts":
        gate = gate[:-1]

    with pytest.raises(ValueError, match=message):
        npu_moe.npu_mh_moe_forward(x, probabilities, indices, experts, gate, up, down)
