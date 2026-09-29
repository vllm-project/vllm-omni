# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]


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
