# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact parity of the MAGI-2 routing and BF16 route layout.

The references below keep the head-major route ids, the ``scatter_add_``
expert histogram and the strided router operand.  The production code must
reproduce their outputs bit for bit.
"""

import pytest
import torch

import vllm_omni.diffusion.models.magi2.mh_moe as moe
from tests.diffusion.models.magi2.test_bf16_moe_wiring import _gpu_device
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

_DEVICES = [
    pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu]),
    pytest.param("musa", marks=[pytest.mark.musa, pytest.mark.gpu]),
]


def _reference_align_bf16_routes(route_ids, num_experts, block_size, buffers=None):
    flat_ids = route_ids.reshape(-1).to(torch.int32)
    route_count = flat_ids.numel()
    order = torch.argsort(flat_ids)
    sorted_experts = flat_ids[order]
    counts = torch.zeros(num_experts, device=flat_ids.device, dtype=torch.int32)
    counts.scatter_add_(0, flat_ids.long(), torch.ones_like(flat_ids))
    counts = counts.long()
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.cumsum(padded_counts, 0) - padded_counts
    ends = torch.cumsum(counts, 0)
    begins = ends - counts
    positions = torch.arange(route_count, device=flat_ids.device, dtype=torch.int64)
    destinations = starts[sorted_experts.long()] + positions - begins[sorted_experts.long()]
    if buffers is None:
        buffers = moe._allocate_bf16_route_buffers(route_count, num_experts, block_size, flat_ids.device)
    sorted_ids, expert_ids, num_padded = buffers
    sorted_ids.fill_(route_count)
    sorted_ids[destinations] = order.to(torch.int32)
    block_starts = torch.arange(expert_ids.numel(), device=flat_ids.device) * block_size
    expert_ids.copy_(torch.searchsorted(padded_counts.cumsum(0), block_starts, right=True))
    num_padded.copy_(padded_counts.sum().reshape(1))
    return sorted_ids, expert_ids, num_padded


def _reference_bf16_fused_moe_forward(x_heads, probabilities, indices, packed_w13, w_down, route_buffers=None):
    num_tokens, num_heads, hidden_size = x_heads.shape
    top_k = probabilities.shape[-1]
    num_experts = packed_w13.shape[0]
    experts_per_head = num_experts // num_heads
    if num_tokens == 0:
        return torch.zeros_like(x_heads)
    head_offsets = (
        torch.arange(num_heads, device=x_heads.device, dtype=torch.int32).view(num_heads, 1, 1) * experts_per_head
    )
    route_ids = (indices.to(torch.int32) + head_offsets).reshape(num_heads * num_tokens, top_k)
    route_weights = probabilities.reshape(num_heads * num_tokens, top_k).contiguous()
    sorted_ids, expert_ids, num_padded = _reference_align_bf16_routes(route_ids, num_experts, 128, route_buffers)
    hidden = x_heads.permute(1, 0, 2).contiguous().reshape(num_heads * num_tokens, hidden_size)
    intermediate_size = packed_w13.shape[1] // 2
    intermediate = torch.empty(
        (num_heads * num_tokens * top_k, intermediate_size), device=x_heads.device, dtype=x_heads.dtype
    )
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32 if not moe.current_omni_platform.is_musa() else 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 4 if not moe.current_omni_platform.is_musa() else 16,
        "num_stages": 3 if not moe.current_omni_platform.is_musa() else 1,
    }
    moe.invoke_fused_moe_bf16(
        hidden,
        packed_w13,
        intermediate,
        route_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=top_k,
        config=config,
        fuse_swiglu=True,
    )
    route_output = torch.empty((num_heads * num_tokens, top_k, hidden_size), device=x_heads.device, dtype=x_heads.dtype)
    moe.invoke_fused_moe_bf16(
        intermediate,
        w_down.transpose(1, 2),
        route_output,
        route_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=1,
        config=config,
        fuse_swiglu=False,
    )
    return route_output.sum(dim=1).reshape(num_heads, num_tokens, hidden_size).permute(1, 0, 2)


def _reference_route(layer, x_heads):
    gate = layer.gate.view(layer.local_num_heads, layer.num_experts, layer.d_head).float()
    logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
    bias = layer.router.expert_bias_ema.view(layer.local_num_heads, layer.num_experts)
    probs, indices = moe.compute_topk_probs_and_indices(
        logits,
        layer.top_k,
        score_func=layer.config.score_func,
        expert_bias=bias,
        route_norm=layer.config.route_norm,
    )
    return probs * layer.config.route_scale, indices


def _skewed_route_ids(num_routes, num_experts, seed):
    # Production routing is skewed: a few experts receive most routes.
    generator = torch.Generator().manual_seed(seed)
    weights = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float64) ** 1.2
    weights = weights[torch.randperm(num_experts, generator=generator)]
    return torch.multinomial(weights, num_routes, replacement=True, generator=generator)


def _route_buffers(route_count, num_experts, block_size, device, fill):
    buffers = moe._allocate_bf16_route_buffers(route_count, num_experts, block_size, device)
    for buffer in buffers:
        buffer.fill_(fill)
    return buffers


def _assert_same_alignment(route_ids, num_experts, block_size):
    device = route_ids.device
    route_count = route_ids.numel()
    expected = _reference_align_bf16_routes(
        route_ids, num_experts, block_size, _route_buffers(route_count, num_experts, block_size, device, -7)
    )
    buffers = _route_buffers(route_count, num_experts, block_size, device, 11)
    actual = moe._align_bf16_routes(route_ids, num_experts, block_size, buffers)
    assert all(result is buffer for result, buffer in zip(actual, buffers))
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype == torch.int32
        assert torch.equal(result, reference)
    # Without caller buffers both sides allocate the same capacity.
    for result, reference in zip(
        moe._align_bf16_routes(route_ids, num_experts, block_size),
        _reference_align_bf16_routes(route_ids, num_experts, block_size),
    ):
        assert torch.equal(result, reference)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    ("route_ids", "num_experts", "block_size"),
    [
        ([], 5, 4),
        ([3], 5, 4),
        ([2, 0, 2, 0, 2], 5, 4),
        ([3] * 9 + [0, 3, 0], 5, 4),
        ([1] * 8, 5, 4),
        ([0] * 5, 5, 4),
        ([4] * 3, 5, 4),
        ([[0, 4], [4, 0], [2, 2]], 5, 4),
    ],
)
def test_align_bf16_routes_matches_histogram_reference(route_ids, num_experts, block_size, dtype):
    _assert_same_alignment(torch.tensor(route_ids, dtype=dtype), num_experts, block_size)


@pytest.mark.cpu
@pytest.mark.parametrize(("num_routes", "seed"), [(1, 3), (127, 4), (5_000, 5), (14_601, 6)])
def test_align_bf16_routes_matches_histogram_reference_on_skewed_routes(num_routes, seed):
    num_experts = 3 * 256
    route_ids = _skewed_route_ids(num_routes, num_experts, seed).to(torch.int32).view(-1, 1)
    _assert_same_alignment(route_ids, num_experts, 128)


@pytest.mark.parametrize("device_type", _DEVICES)
def test_align_bf16_routes_matches_histogram_reference_on_device(device_type):
    device = _gpu_device(device_type)
    # One EP4 rank: 3 local heads x 256 experts, top-6 routes for 14,808 tokens.
    num_experts = 3 * 256
    route_ids = _skewed_route_ids(14_808 * 3 * 6, num_experts, 7).to(torch.int32).view(-1, 6)
    _assert_same_alignment(route_ids.to(device), num_experts, 128)


def _routed_inputs(num_tokens, num_heads, top_k, experts_per_head, *, hidden_size, intermediate_size, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, num_heads, hidden_size, generator=generator).to(torch.bfloat16)
    num_experts = num_heads * experts_per_head
    gate = (torch.randn(num_experts, hidden_size, intermediate_size, generator=generator) * 0.3).to(torch.bfloat16)
    up = (torch.randn(num_experts, hidden_size, intermediate_size, generator=generator) * 0.7 - 0.5).to(torch.bfloat16)
    down = (torch.randn(num_experts, intermediate_size, hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    scores = torch.rand(num_heads, num_tokens, experts_per_head, generator=generator)
    # Bias the scores so several experts get no routes and others get more than one block.
    scores = scores + torch.linspace(1.0, 0.0, experts_per_head) ** 4
    indices = scores.topk(top_k, dim=-1).indices
    probabilities = torch.rand(num_heads, num_tokens, top_k, generator=generator)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return x, probabilities, indices, moe._pack_bf16_w13(gate, up), down


def _rowwise_invoke(a, b, c, weights, sorted_ids, expert_ids, num_padded, *, top_k, config, fuse_swiglu):
    """Evaluate each route on its own, so no value depends on its slot in a block."""
    route_count = weights.numel()
    output = c.view(route_count, c.shape[-1])
    block_size = config["BLOCK_SIZE_M"]
    for block, expert in enumerate(expert_ids[: int(num_padded.item()) // block_size].tolist()):
        for route in sorted_ids[block * block_size : (block + 1) * block_size].tolist():
            if route >= route_count:
                continue
            gemm = a[route // top_k].float() @ b[expert].float().T
            if fuse_swiglu:
                gate = gemm[0::2].clamp(max=7)
                up = gemm[1::2].clamp(-7, 7)
                result = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            else:
                result = gemm * weights.reshape(-1)[route]
            output[route] = result.to(c.dtype)


def _routes_by_expert(sorted_ids, expert_ids, num_padded, num_tokens, num_heads, top_k, *, token_major):
    """Map each expert to the sorted ``(head, token, choice)`` routes in its blocks.

    Blocks are compared per expert: a sort that does not keep tie order may
    split one expert's routes across its blocks differently.
    """
    route_count = num_tokens * num_heads * top_k
    live_blocks = int(num_padded.item()) // 128
    experts: dict[int, list[tuple[int, int, int]]] = {}
    for expert, block in zip(expert_ids[:live_blocks].tolist(), sorted_ids[: live_blocks * 128].view(-1, 128).tolist()):
        for route in block:
            if route >= route_count:
                continue
            row, choice = divmod(route, top_k)
            first, second = divmod(row, num_heads if token_major else num_tokens)
            experts.setdefault(expert, []).append((second, first, choice) if token_major else (first, second, choice))
    return {expert: sorted(routes) for expert, routes in experts.items()}


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "top_k", "experts_per_head"),
    [(0, 3, 2, 4), (1, 1, 2, 4), (1, 3, 6, 8), (7, 3, 2, 4), (129, 2, 2, 4), (131, 3, 6, 8)],
)
def test_token_major_forward_matches_head_major_reference(monkeypatch, num_tokens, num_heads, top_k, experts_per_head):
    x, probabilities, indices, packed_w13, down = _routed_inputs(
        num_tokens, num_heads, top_k, experts_per_head, hidden_size=16, intermediate_size=10, seed=num_tokens
    )
    launches = []

    def invoke(*args, **kwargs):
        _rowwise_invoke(*args, **kwargs)
        launches.append((args[2].clone(), args[4].clone(), args[5].clone(), args[6].clone()))

    monkeypatch.setattr(moe, "invoke_fused_moe_bf16", invoke)
    expected = _reference_bf16_fused_moe_forward(x, probabilities, indices, packed_w13, down)
    actual = moe._bf16_fused_moe_forward(x, probabilities, indices, packed_w13, down)

    assert actual.shape == expected.shape == x.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    if num_tokens == 0:
        assert launches == []
        return
    assert len(launches) == 4
    (_, old_ids, old_experts, old_padded), (old_routes, *_) = launches[0], launches[1]
    (_, new_ids, new_experts, new_padded), (new_routes, *_) = launches[2], launches[3]
    # Same expert per block and the same routes per expert; only the id encoding differs.
    assert torch.equal(new_padded, old_padded)
    live_blocks = int(old_padded.item()) // 128
    assert torch.equal(new_experts[:live_blocks], old_experts[:live_blocks])
    shape = (num_tokens, num_heads, top_k)
    assert _routes_by_expert(new_ids, new_experts, new_padded, *shape, token_major=True) == _routes_by_expert(
        old_ids, old_experts, old_padded, *shape, token_major=False
    )
    # The down GEMM writes route (token, head, choice) to the same values.
    hidden_size = x.shape[-1]
    assert torch.equal(
        new_routes.view(num_tokens, num_heads, top_k, hidden_size),
        old_routes.view(num_heads, num_tokens, top_k, hidden_size).transpose(0, 1),
    )


@pytest.mark.parametrize("device_type", _DEVICES)
@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "top_k", "experts_per_head", "hidden_size", "intermediate_size"),
    [
        (1, 1, 2, 4, 64, 96),
        (17, 3, 2, 4, 64, 96),
        (129, 2, 6, 8, 256, 1280),
        # One EP4 rank of MAGI-2 Preview: 3 local heads over 14,808 tokens.
        (14_808, 3, 6, 256, 256, 1280),
    ],
)
def test_token_major_forward_matches_head_major_reference_on_device(
    device_type, num_tokens, num_heads, top_k, experts_per_head, hidden_size, intermediate_size
):
    device = _gpu_device(device_type)
    inputs = _routed_inputs(
        num_tokens,
        num_heads,
        top_k,
        experts_per_head,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=num_tokens,
    )
    x, probabilities, indices, packed_w13, down = (tensor.to(device) for tensor in inputs)
    route_count = num_tokens * num_heads * top_k
    expected = _reference_bf16_fused_moe_forward(
        x,
        probabilities,
        indices,
        packed_w13,
        down,
        moe._allocate_bf16_route_buffers(route_count, packed_w13.shape[0], 128, device),
    )
    actual = moe._bf16_fused_moe_forward(
        x,
        probabilities,
        indices,
        packed_w13,
        down,
        moe._allocate_bf16_route_buffers(route_count, packed_w13.shape[0], 128, device),
    )
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device_type", _DEVICES)
def test_token_major_routes_fill_identical_blocks_on_device(monkeypatch, device_type):
    """Each expert serves one head, and for a fixed head both id layouts order
    routes by ``(token, choice)``.  The device radix sort keeps that order, so
    every GEMM block holds the same routes in the same rows."""
    device = _gpu_device(device_type)
    num_tokens, num_heads, top_k = 14_808, 3, 6
    inputs = _routed_inputs(num_tokens, num_heads, top_k, 256, hidden_size=16, intermediate_size=8, seed=11)
    x, probabilities, indices, packed_w13, down = (tensor.to(device) for tensor in inputs)
    launches = []

    def record(*args, **kwargs):
        launches.append(tuple(tensor.clone() for tensor in args[4:7]))

    monkeypatch.setattr(moe, "invoke_fused_moe_bf16", record)
    _reference_bf16_fused_moe_forward(x, probabilities, indices, packed_w13, down)
    moe._bf16_fused_moe_forward(x, probabilities, indices, packed_w13, down)
    assert len(launches) == 4
    (old_ids, old_experts, old_padded), (new_ids, new_experts, new_padded) = launches[0], launches[2]
    assert torch.equal(new_padded, old_padded)
    assert torch.equal(new_experts, old_experts)
    route_count = num_tokens * num_heads * top_k
    valid = new_ids < route_count
    new_ids = new_ids.long()
    rows, choices = new_ids // top_k, new_ids % top_k
    head_major = ((rows % num_heads) * num_tokens + rows // num_heads) * top_k + choices
    assert torch.equal(torch.where(valid, head_major, new_ids), old_ids.long())


def _router_layer(num_heads, num_experts, top_k, d_head, seed):
    config = moe.Magi2MultiHeadMoEConfig(
        hidden_size=num_heads * d_head,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        expert_intermediate_size=8,
        params_dtype=torch.bfloat16,
        route_scale=4.9,
    )
    layer = moe.Magi2MultiHeadMoE(config, ep_group=Magi2ParallelGroup(None, 1, 0))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        layer.gate.copy_(torch.randn(layer.gate.shape, generator=generator) / d_head**0.5)
        layer.router.expert_bias_ema.copy_(torch.randn(layer.router.expert_bias_ema.shape, generator=generator) * 0.1)
    return layer


def _assert_same_route(monkeypatch, layer, x_heads):
    logits = []
    topk = moe.compute_topk_probs_and_indices

    def capture(router_logits, *args, **kwargs):
        logits.append(router_logits)
        return topk(router_logits, *args, **kwargs)

    monkeypatch.setattr(moe, "compute_topk_probs_and_indices", capture)
    monkeypatch.delenv("MAGI2_ROUTER_BIAS_SOURCE", raising=False)
    expected = _reference_route(layer, x_heads)
    actual = layer._route(x_heads)
    assert len(logits) == 2
    assert logits[0].dtype == logits[1].dtype == torch.float32
    assert torch.equal(logits[1], logits[0])
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        assert torch.equal(result, reference)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "num_experts", "top_k", "strided"),
    [(0, 3, 8, 2, False), (1, 1, 8, 2, False), (7, 3, 16, 6, False), (33, 2, 256, 6, False), (9, 3, 16, 6, True)],
)
def test_route_matches_strided_operand_reference(monkeypatch, num_tokens, num_heads, num_experts, top_k, strided):
    d_head = 32
    layer = _router_layer(num_heads, num_experts, top_k, d_head, seed=num_tokens)
    generator = torch.Generator().manual_seed(num_tokens + 1)
    x = torch.randn(num_tokens, num_heads + int(strided), d_head, generator=generator).to(torch.bfloat16)
    # The strided case reads a head slice of a wider buffer.
    _assert_same_route(monkeypatch, layer, x[:, int(strided) :])


@pytest.mark.parametrize("device_type", _DEVICES)
@pytest.mark.parametrize("num_tokens", [1, 129, 14_808])
def test_route_matches_strided_operand_reference_on_device(monkeypatch, device_type, num_tokens):
    """The router bmm consumes the head-major copy that MUSA bmm makes of a strided operand."""
    device = _gpu_device(device_type)
    layer = _router_layer(3, 256, 6, 256, seed=num_tokens).to(device)
    generator = torch.Generator().manual_seed(num_tokens + 1)
    x = torch.randn(num_tokens, 3, 256, generator=generator).to(torch.bfloat16)
    _assert_same_route(monkeypatch, layer, x.to(device))
