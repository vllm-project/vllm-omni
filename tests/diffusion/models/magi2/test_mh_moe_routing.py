# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Parity tests for the fused MAGI-2 multi-head MoE route construction.

The fused CUDA path must reproduce the unfused fallback: the same expert bank
per head, a bias that steers selection only, the same route weights, and a
bit-identical CSR layout.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.mh_moe import (
    _reference_global_sort_routes,
    _reference_topk_probs_and_indices,
    compute_topk_probs_and_indices,
    global_sort_routes,
    torch_mh_moe_forward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# ``heads, tokens, experts, top_k``.  The released MAGI-2 Preview bank is
# ``(12, S, 256, 6)``; the rest cover padding, non-power-of-two banks and the
# degenerate top_k values.
ROUTING_SHAPES = [
    (12, 1024, 256, 6),
    (12, 1, 256, 6),
    (1, 37, 256, 6),
    (4, 512, 8, 2),
    (3, 100, 12, 5),
    (2, 77, 64, 1),
    (2, 64, 32, 32),
    (5, 129, 1024, 8),
]


def _router_inputs(
    heads: int,
    tokens: int,
    experts: int,
    *,
    device: str,
    with_bias: bool = True,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(heads, tokens, experts, device=device, generator=generator)
    if not with_bias:
        return logits, None
    return logits, torch.randn(heads, experts, device=device, generator=generator) * 0.01


# The fused kernel evaluates sigmoid with libdevice's exp, which sits within one
# ULP of torch.sigmoid rather than on top of it, and folds the L1 normalization
# into the same pass.  Both leave a few-ULP margin on the route weights.
ROUTE_WEIGHT_RTOL = 1e-6
ROUTE_WEIGHT_ATOL = 1e-6


def _resolvable_rows(
    logits: torch.Tensor,
    expert_bias: torch.Tensor | None,
    top_k: int,
    *,
    margin: float = 1e-5,
) -> torch.Tensor:
    """Mask of ``[head, token]`` rows whose top-k selection is unambiguous.

    Only the ``top_k + 1`` largest selection scores decide which experts are
    picked and in what order.  Rows where two of them are closer than the
    combined sigmoid and reduction error cannot be compared index-for-index:
    ``torch.topk`` does not define the order of an exact tie either.
    """

    selection_scores = torch.sigmoid(logits)
    if expert_bias is not None:
        selection_scores = selection_scores + expert_bias.view(logits.shape[0], 1, -1)
    boundary = min(top_k + 1, selection_scores.shape[-1])
    ordered = selection_scores.topk(boundary, dim=-1).values
    if boundary == 1:
        return torch.ones(ordered.shape[:-1], dtype=torch.bool, device=ordered.device)
    return ((ordered[..., :-1] - ordered[..., 1:]).abs() > margin).all(dim=-1)


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
@pytest.mark.parametrize(("heads", "tokens", "experts", "top_k"), ROUTING_SHAPES)
@pytest.mark.parametrize("route_norm", [True, False])
def test_fused_routing_matches_reference(heads: int, tokens: int, experts: int, top_k: int, route_norm: bool) -> None:
    logits, expert_bias = _router_inputs(heads, tokens, experts, device="cuda")
    reference_probs, reference_indices = _reference_topk_probs_and_indices(
        logits, top_k, expert_bias=expert_bias, route_norm=route_norm
    )
    fused_probs, fused_indices = compute_topk_probs_and_indices(
        logits, top_k, expert_bias=expert_bias, route_norm=route_norm
    )

    assert fused_probs.shape == reference_probs.shape
    assert fused_probs.dtype == reference_probs.dtype
    assert fused_indices.dtype == reference_indices.dtype

    resolvable = _resolvable_rows(logits, expert_bias, top_k)
    # Guards against the comparison below going vacuous, nothing more.  The
    # unresolvable fraction grows with the bank width -- the top few order
    # statistics of 1024 samples sit closer together than those of 256 -- and
    # CUDA's RNG does not produce identical draws across GPU architectures, so
    # this floor is deliberately loose.
    assert resolvable.float().mean() > 0.95, "random router rows are unexpectedly ambiguous; the check below is vacuous"
    assert torch.equal(fused_indices[resolvable], reference_indices[resolvable])
    torch.testing.assert_close(
        fused_probs[resolvable], reference_probs[resolvable], rtol=ROUTE_WEIGHT_RTOL, atol=ROUTE_WEIGHT_ATOL
    )


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
def test_fused_routing_without_bias_matches_reference() -> None:
    logits, _ = _router_inputs(12, 512, 256, device="cuda", with_bias=False)
    reference = _reference_topk_probs_and_indices(logits, 6)
    fused = compute_topk_probs_and_indices(logits, 6)
    resolvable = _resolvable_rows(logits, None, 6)
    assert torch.equal(fused[1][resolvable], reference[1][resolvable])
    torch.testing.assert_close(fused[0], reference[0], rtol=ROUTE_WEIGHT_RTOL, atol=ROUTE_WEIGHT_ATOL)


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
def test_selection_bias_does_not_reach_the_route_weight() -> None:
    """A bias large enough to reorder selection must not change the weights."""

    logits, _ = _router_inputs(4, 256, 64, device="cuda", with_bias=False)
    expert_bias = torch.zeros(4, 64, device="cuda")
    expert_bias[:, ::2] = 5.0
    probs, indices = compute_topk_probs_and_indices(logits, 4, expert_bias=expert_bias, route_norm=False)

    # Every selected expert comes from the boosted half, and the returned weight
    # is the unbiased sigmoid score.
    assert (indices % 2 == 0).all()
    expected = torch.sigmoid(logits).gather(-1, indices)
    torch.testing.assert_close(probs, expected, rtol=ROUTE_WEIGHT_RTOL, atol=ROUTE_WEIGHT_ATOL)


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
def test_fused_routing_ties_break_towards_the_lowest_expert() -> None:
    """Exact ties are undefined for torch.topk; the fused kernel is explicit."""

    logits = torch.full((1, 1, 8), -10.0, device="cuda")
    logits[0, 0, [1, 3, 5]] = 2.0
    probs, indices = compute_topk_probs_and_indices(logits, 3, route_norm=False)
    assert indices[0, 0].tolist() == [1, 3, 5]

    logits[0, 0, 6] = 2.0
    probs, indices = compute_topk_probs_and_indices(logits, 2, route_norm=False)
    assert indices[0, 0].tolist() == [1, 3]
    torch.testing.assert_close(
        probs[0, 0], torch.sigmoid(logits[0, 0, [1, 3]]), rtol=ROUTE_WEIGHT_RTOL, atol=ROUTE_WEIGHT_ATOL
    )


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
@pytest.mark.parametrize(("heads", "tokens", "experts", "top_k"), ROUTING_SHAPES)
def test_global_sort_routes_is_bit_identical_to_reference(heads: int, tokens: int, experts: int, top_k: int) -> None:
    logits, expert_bias = _router_inputs(heads, tokens, experts, device="cuda")
    probs, indices = compute_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias)
    expected = _reference_global_sort_routes(probs, indices, experts)
    actual = global_sort_routes(probs, indices, experts)
    for reference_tensor, candidate, name in zip(expected, actual, ("gather_ids", "probs", "offsets"), strict=True):
        assert candidate.dtype == reference_tensor.dtype, name
        assert torch.equal(candidate, reference_tensor), name


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
def test_global_sort_routes_layout_invariants() -> None:
    heads, tokens, experts, top_k = 6, 333, 64, 3
    logits, expert_bias = _router_inputs(heads, tokens, experts, device="cuda")
    probs, indices = compute_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias)
    gather_ids, sorted_probs, offsets = global_sort_routes(probs, indices, experts)

    assert offsets.numel() == heads * experts + 1
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == heads * tokens * top_k
    assert (offsets.diff() >= 0).all()
    assert gather_ids.numel() == sorted_probs.numel() == heads * tokens * top_k
    assert int(gather_ids.min()) >= 0 and int(gather_ids.max()) < tokens

    # Each flat expert bucket holds exactly the tokens that routed to it, in
    # ascending token order -- the stability the expert kernel's tiling relies on.
    for flat_expert in range(0, heads * experts, 17):
        head, expert = divmod(flat_expert, experts)
        begin, end = int(offsets[flat_expert]), int(offsets[flat_expert + 1])
        bucket = gather_ids[begin:end].long()
        assert torch.equal(bucket, bucket.sort().values)
        expected_tokens = (indices[head] == expert).any(dim=-1).nonzero().flatten()
        assert torch.equal(bucket, expected_tokens.to(bucket.dtype))


@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
def test_routes_drive_the_expert_kernel_identically() -> None:
    """The two route builders must produce interchangeable expert inputs."""

    heads, tokens, experts, top_k, head_dim, expert_dim = 3, 64, 16, 4, 32, 48
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(7)
    logits, expert_bias = _router_inputs(heads, tokens, experts, device=device, seed=7)
    hidden = torch.randn(tokens, heads, head_dim, device=device, generator=generator)
    flat_experts = heads * experts
    w_gate = torch.randn(flat_experts, head_dim, expert_dim, device=device, generator=generator) * 0.05
    w_up = torch.randn(flat_experts, head_dim, expert_dim, device=device, generator=generator) * 0.05
    w_down = torch.randn(flat_experts, expert_dim, head_dim, device=device, generator=generator) * 0.05

    outputs = []
    for builder, layout in (
        (_reference_topk_probs_and_indices, _reference_global_sort_routes),
        (compute_topk_probs_and_indices, global_sort_routes),
    ):
        probs, indices = builder(logits, top_k, expert_bias=expert_bias)
        gather_ids, sorted_probs, offsets = layout(probs, indices, experts)
        outputs.append(torch_mh_moe_forward(hidden, gather_ids, sorted_probs, offsets, w_gate, w_up, w_down))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
@pytest.mark.parametrize(("heads", "tokens", "experts", "top_k"), [(3, 40, 32, 4), (2, 16, 12, 3)])
def test_cpu_routing_falls_back_to_the_reference(heads: int, tokens: int, experts: int, top_k: int) -> None:
    logits, expert_bias = _router_inputs(heads, tokens, experts, device="cpu")
    reference = _reference_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias)
    fallback = compute_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias)
    assert torch.equal(fallback[0], reference[0])
    assert torch.equal(fallback[1], reference[1])

    expected = _reference_global_sort_routes(*reference, experts)
    actual = global_sort_routes(*reference, experts)
    for reference_tensor, candidate in zip(expected, actual, strict=True):
        assert torch.equal(candidate, reference_tensor)


@pytest.mark.cpu
def test_routing_rejects_malformed_inputs() -> None:
    logits = torch.randn(2, 8, 16)
    with pytest.raises(ValueError, match="heads,tokens,experts"):
        compute_topk_probs_and_indices(logits[0], 4)
    with pytest.raises(ValueError, match="top_k"):
        compute_topk_probs_and_indices(logits, 17)
    with pytest.raises(ValueError, match="top_k"):
        compute_topk_probs_and_indices(logits, 0)
    with pytest.raises(ValueError, match="unsupported routing score function"):
        compute_topk_probs_and_indices(logits, 4, score_func="argmax")  # type: ignore[arg-type]
    probs, indices = compute_topk_probs_and_indices(logits, 4)
    with pytest.raises(ValueError, match=r"\[H,S,K\] shape"):
        global_sort_routes(probs, indices[..., :2], 16)


# Route construction is deterministic in its output but not in its latency, and
# CI spans L4, H100 and B200, so an absolute microsecond baseline would be
# either meaningless or flaky.  Gate on the speedup over the reference measured
# in the same process on the same device instead: that cancels the hardware
# difference.  Observed on a B200 (torch 2.13, Triton 3.7.1): 6.1x at 4096
# tokens and 11.1x at 29184.  The floor below leaves roughly half of that as
# headroom, so it catches a real regression -- or the fused path silently
# falling back -- without tracking normal run-to-run spread.
ROUTING_SPEEDUP_FLOOR = 3.0


def _median_microseconds(operation, *, warmup: int = 10, iterations: int = 30) -> float:
    import statistics

    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.accelerator.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


@pytest.mark.benchmark
@hardware_test(res={"cuda": ["B200", "H100", "L4"]}, num_cards=1)
def test_fused_routing_stays_faster_than_the_reference() -> None:
    """Regression gate on the M1 route-construction speedup."""

    heads, tokens, experts, top_k = 12, 4096, 256, 6
    logits, expert_bias = _router_inputs(heads, tokens, experts, device="cuda")

    reference_us = _median_microseconds(
        lambda: _reference_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias)
    )
    fused_us = _median_microseconds(lambda: compute_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias))
    topk_speedup = reference_us / fused_us

    probs, indices = compute_topk_probs_and_indices(logits, top_k, expert_bias=expert_bias)
    reference_layout_us = _median_microseconds(lambda: _reference_global_sort_routes(probs, indices, experts))
    fused_layout_us = _median_microseconds(lambda: global_sort_routes(probs, indices, experts))
    layout_speedup = reference_layout_us / fused_layout_us

    total_speedup = (reference_us + reference_layout_us) / (fused_us + fused_layout_us)
    assert total_speedup >= ROUTING_SPEEDUP_FLOOR, (
        f"route construction speedup {total_speedup:.2f}x fell below {ROUTING_SPEEDUP_FLOOR}x "
        f"(top-k {topk_speedup:.2f}x at {reference_us:.0f}->{fused_us:.0f} us, "
        f"layout {layout_speedup:.2f}x at {reference_layout_us:.0f}->{fused_layout_us:.0f} us)"
    )
