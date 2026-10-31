# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def _bottom_right_causal_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    key_keep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    query_positions = torch.arange(query.shape[1]).unsqueeze(1)
    key_positions = torch.arange(key.shape[1]).unsqueeze(0)
    keep_mask = key_positions <= query_positions + (key.shape[1] - query.shape[1])
    if key_keep_mask is not None:
        assert query.shape[0] == 1
        keep_mask = keep_mask & key_keep_mask[0].bool().unsqueeze(0)
    valid_rows = keep_mask.any(dim=-1)

    output = torch.zeros(
        query.shape[0],
        query.shape[1],
        query.shape[2],
        value.shape[-1],
        dtype=torch.float32,
    )
    valid_scores = scores[:, :, valid_rows].masked_fill(~keep_mask[valid_rows], float("-inf"))
    probabilities = torch.softmax(valid_scores, dim=-1)
    output[:, valid_rows] = torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())
    return output


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="Native causal attention requires Ascend NPU")
def test_npu_bottom_right_causal_fully_masked_rows_are_zero():
    torch.manual_seed(0)
    batch_size, query_length, key_length = 1, 4, 2
    num_heads, head_size = 2, 64
    scale = head_size**-0.5

    query_cpu = torch.randn(batch_size, query_length, num_heads, head_size).to(torch.bfloat16)
    key_cpu = torch.randn(batch_size, key_length, num_heads, head_size).to(torch.bfloat16)
    value_cpu = torch.randn(batch_size, key_length, num_heads, head_size).to(torch.bfloat16)
    expected = _bottom_right_causal_reference(query_cpu, key_cpu, value_cpu, scale)

    device = torch.device(current_omni_platform.device_type)
    query = query_cpu.to(device)
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    impl = FlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=scale,
        causal=True,
    )

    output = impl.forward_fa_npu(query, key, value)
    output_cpu = output.float().cpu()

    assert torch.isfinite(output_cpu).all()
    assert torch.equal(output_cpu[:, : query_length - key_length], torch.zeros_like(output_cpu[:, :2]))
    torch.testing.assert_close(
        output_cpu[:, query_length - key_length :],
        expected[:, query_length - key_length :],
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="Native causal attention requires Ascend NPU")
def test_npu_bottom_right_causal_composes_explicit_keep_mask():
    torch.manual_seed(1)
    batch_size, sequence_length = 1, 4
    num_heads, head_size = 2, 64
    scale = head_size**-0.5
    key_keep_mask_cpu = torch.tensor([[True, False, True, True]])

    query_cpu = torch.randn(batch_size, sequence_length, num_heads, head_size).to(torch.bfloat16)
    key_cpu = torch.randn(batch_size, sequence_length, num_heads, head_size).to(torch.bfloat16)
    value_cpu = torch.randn(batch_size, sequence_length, num_heads, head_size).to(torch.bfloat16)
    expected = _bottom_right_causal_reference(
        query_cpu,
        key_cpu,
        value_cpu,
        scale,
        key_keep_mask_cpu,
    )

    device = torch.device(current_omni_platform.device_type)
    query = query_cpu.to(device)
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    impl = FlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=scale,
        causal=True,
    )

    output = impl.forward_fa_npu(
        query,
        key,
        value,
        AttentionMetadata(attn_mask=key_keep_mask_cpu.to(device)),
    )
    output_cpu = output.float().cpu()

    assert torch.isfinite(output_cpu).all()
    torch.testing.assert_close(output_cpu, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="Native GQA attention requires Ascend NPU")
@pytest.mark.parametrize("q_len,kv_len", [(1, 17), (5, 17), (17, 5)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "keep", "bias"])
def test_npu_gqa_matches_independent_reference(q_len, kv_len, causal, mask_kind):
    # Independent CPU float64 reference; the backend executes on the real NPU.
    from tests.diffusion.attention.test_flash_attn_npu_fallback import _reference

    torch.manual_seed(92)
    q = torch.randn(1, q_len, 8, 128).to(torch.bfloat16)
    k = torch.randn(1, kv_len, 2, 128).to(torch.bfloat16)
    v = torch.randn_like(k)
    mask = None
    if mask_kind == "keep":
        mask = torch.ones(1, kv_len, dtype=torch.bool)
        mask[:, 1] = False
    elif mask_kind == "bias":
        mask = torch.zeros(1, 1, q_len, kv_len, dtype=torch.bfloat16)
        mask[..., 1] = -torch.inf
        mask[..., -1] = 0.375
    scale = 0.23
    expected = _reference(q, k, v, mask, causal, scale)
    device = torch.device(current_omni_platform.device_type)
    impl = FlashAttentionImpl(num_heads=8, num_kv_heads=2, head_size=128, softmax_scale=scale, causal=causal)
    metadata = AttentionMetadata(attn_mask=mask.to(device)) if mask is not None else None
    got = impl.forward_fa_npu(q.to(device), k.to(device), v.to(device), metadata)
    torch.testing.assert_close(got.cpu().double(), expected, atol=3e-2, rtol=3e-2)
