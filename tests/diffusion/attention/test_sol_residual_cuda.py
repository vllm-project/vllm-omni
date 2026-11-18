# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

if not current_omni_platform.is_cuda():
    pytest.skip("SOL residual fusion requires CUDA", allow_module_level=True)


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
@pytest.mark.parametrize(
    "batch,tokens,heads,kv_blocks",
    [(1, 1, 1, None), (1, 63, 2, None), (1, 64, 2, None), (2, 257, 2, None), (1, 2057, 2, None), (1, 65, 1, 2051)],
)
@pytest.mark.parametrize(
    "routing,lse_offset", [("exact", 0), ("residual", 0), ("mixed", 0), ("mixed", -80), ("mixed", 80)]
)
@pytest.mark.parametrize("chunks_per_launch", [1, 8])
@torch.inference_mode()
def test_fused_residual_matches_normalized_mixture(
    batch, tokens, heads, kv_blocks, routing, lse_offset, chunks_per_launch
):
    """Independent FP64 mixture oracle, including multiplicity and ragged tails."""
    from vllm_omni.diffusion.attention.ops.sol_residual import merge_sol_residual

    generator = torch.Generator(device="cuda").manual_seed(83)
    query_blocks = (tokens + 63) // 64
    # A short query prefix covers the wide-key reduction without a quadratic
    # FP64 oracle for an entire 131k-token sequence.
    blocks = kv_blocks or query_blocks
    kv_tokens = blocks * 64 - 1 if kv_blocks else tokens

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    query = rand(batch, tokens, heads, 128)
    if lse_offset:
        query.mul_(32)
    # Sliced caches and padded exact outputs must retain their actual strides.
    keys = rand(batch, blocks + 3, heads, 128)[:, :blocks]
    sizes = (kv_tokens - torch.arange(blocks, device="cuda", dtype=torch.int32) * 64).clamp(max=64)
    sums = (rand(batch, blocks + 3, heads, 128)[:, :blocks].float() * sizes[None, :, None, None]).bfloat16()
    exact = rand(batch, query_blocks * 64, heads, 128)
    lse = torch.randn(batch, heads, query_blocks * 64, device="cuda", generator=generator) + lse_offset
    selected = (torch.rand(batch, heads, query_blocks, blocks * 2, device="cuda", generator=generator) > 0.5)[..., ::2]
    if routing == "exact":
        selected.fill_(True)
    elif routing == "residual":
        selected.fill_(False)
        lse.fill_(-torch.inf)
        exact.zero_()
    else:
        # Noncontiguous protected sink blocks and different per-head routes.
        selected[:, :, 0, :] = True
        selected[:, :, :, -1] = True

    scale = 128**-0.5
    actual = merge_sol_residual(
        query, keys, sums, selected, sizes, exact, lse, scale, chunks_per_launch=chunks_per_launch
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", query.double(), keys.double()) * scale
    expanded_mask = selected.repeat_interleave(64, dim=-2)[:, :, :tokens]
    scores.masked_fill_(expanded_mask, -torch.inf)
    # One mixture component represents the exact attention partition; each
    # residual component represents n identical centroid/value-mean tokens.
    logits = torch.cat((lse[:, :, :tokens, None].double(), scores + sizes.double().log()), dim=-1)
    probabilities = logits.softmax(-1)
    expected = exact[:, :tokens].double().transpose(1, 2) * probabilities[..., :1]
    expected += torch.einsum(
        "bhqk,bhkd->bhqd", probabilities[..., 1:], (sums.double() / sizes[None, :, None, None]).transpose(1, 2)
    )
    expected = expected.transpose(1, 2)
    assert actual.shape == query.shape and actual.dtype == query.dtype
    assert torch.isfinite(actual).all()
    # BF16 output rounding, compared to an independently formed FP64 oracle.
    torch.testing.assert_close(actual.double(), expected, rtol=0.004, atol=2e-4)
    # Also guard the eager normalization's rounding boundaries. A combined
    # softmax over exact/residual partitions passes the FP64 tolerance above,
    # yet can change sparse routes after many BF16 transformer layers.
    eager = torch.empty_like(query)
    for start in range(0, tokens, 256):
        stop = min(start + 256, tokens)
        eager_scores = (
            query[:, start:stop].float().transpose(1, 2) @ keys.float().transpose(1, 2).transpose(-1, -2) * scale
        )
        eager_scores.masked_fill_(expanded_mask[:, :, start:stop], -torch.inf)
        eager_lse = (eager_scores + sizes.float().log()).logsumexp(-1)
        joint_lse = torch.logaddexp(lse[:, :, start:stop], eager_lse)
        chunk = exact[:, start:stop].float().transpose(1, 2) * (lse[:, :, start:stop] - joint_lse).exp().unsqueeze(-1)
        chunk += (eager_scores - joint_lse.unsqueeze(-1)).exp() @ sums.float().transpose(1, 2)
        eager[:, start:stop] = chunk.transpose(1, 2).to(query.dtype)
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    if routing == "exact":
        torch.testing.assert_close(actual, exact[:, :tokens], rtol=0, atol=0)
    elif routing == "mixed":
        torch.testing.assert_close(actual[:, : min(tokens, 64)], exact[:, : min(tokens, 64)], rtol=0, atol=0)
