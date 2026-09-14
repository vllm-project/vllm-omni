# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan block residual contracts, independent of attention kernels and weights.

Run the production forward with deterministic child modules so modulation
indexing, broadcasting, and eager rounding remain observable at both updates.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformerBlock

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _RecordingIdentity(nn.Module):
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        self.input = x.detach().clone()
        return x


class _FixedBranch(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, *args) -> torch.Tensor:
        return self.output


def _make_block(
    table: torch.Tensor,
    attention: torch.Tensor,
    cross_attention: torch.Tensor,
    feed_forward: torch.Tensor,
) -> WanTransformerBlock:
    # Bypass TP/kernel initialization, but keep WanTransformerBlock.forward.
    block = WanTransformerBlock.__new__(WanTransformerBlock)
    nn.Module.__init__(block)
    block.scale_shift_table = nn.Parameter(table, requires_grad=False)
    block.norm1 = _RecordingIdentity()
    block.norm2 = _RecordingIdentity()
    block.norm3 = _RecordingIdentity()
    block.attn1 = _FixedBranch(attention)
    block.attn2 = _FixedBranch(cross_attention)
    block.ffn = _FixedBranch(feed_forward)
    return block


def _run_block(block: WanTransformerBlock, residual: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
    unused = torch.empty(0)
    with torch.inference_mode():
        return block(residual, encoder_hidden_states=unused, temb=temb, rotary_emb=(unused, unused))


@pytest.mark.parametrize("token_wise", [False, True], ids=["standard", "ti2v"])
def test_gates_preserve_batch_and_token_conditions(token_wise: bool) -> None:
    residual = torch.tensor([10.0, 20.0]).expand(2, 3, 2).clone()
    table = torch.zeros(1, 6, 2)
    table[0, 2] = torch.tensor([1.0, -1.0])
    table[0, 5] = torch.tensor([0.0, 2.0])
    block = _make_block(
        table,
        attention=torch.tensor([2.0, 3.0]).expand_as(residual).clone(),
        cross_attention=torch.tensor([5.0, 7.0]).expand_as(residual).clone(),
        feed_forward=torch.tensor([4.0, 6.0]).expand_as(residual).clone(),
    )

    if token_wise:
        temb = torch.zeros(2, 3, 6, 2)
        temb[:, :, 2] = torch.tensor([[[0, 1], [1, 0], [-1, 2]], [[2, 0], [-2, 1], [0, -1]]])
        temb[:, :, 5] = torch.tensor([[[2, -1], [0, 0], [-1, -2]], [[-1, 0], [1, -3], [3, -1]]])
        expected_attention = torch.tensor([[[12, 20], [14, 17], [10, 23]], [[16, 17], [8, 20], [12, 14]]])
        expected_output = torch.tensor([[[25, 33], [19, 36], [11, 30]], [[17, 36], [17, 21], [29, 27]]])
    else:
        temb = torch.zeros(2, 6, 2)
        temb[:, 2] = torch.tensor([[0, 1], [2, 0]])
        temb[:, 5] = torch.tensor([[2, -1], [-1, 0]])
        expected_attention = torch.tensor([[[12, 20]], [[16, 17]]]).expand_as(residual)
        expected_output = torch.tensor([[[25, 33]], [[17, 36]]]).expand_as(residual)

    output = _run_block(block, residual, temb)

    # Distinct gates catch batch/token broadcasting and attention/FFN mix-ups.
    # A nonzero cross-attention branch also catches accidentally gating that update.
    torch.testing.assert_close(block.norm2.input, expected_attention.float(), rtol=0, atol=0)
    torch.testing.assert_close(output, expected_output.float(), rtol=0, atol=0)


@pytest.mark.parametrize("token_wise", [False, True], ids=["standard", "ti2v"])
def test_fp32_gates_restore_bf16_residual_after_each_update(token_wise: bool) -> None:
    residual = torch.full((2, 3, 2), -1.0, dtype=torch.bfloat16)
    block = _make_block(
        torch.zeros(1, 6, 2, dtype=torch.float32),
        attention=torch.ones_like(residual),
        cross_attention=torch.zeros_like(residual),
        feed_forward=torch.ones_like(residual),
    )
    temb = torch.zeros((2, 3, 6, 2) if token_wise else (2, 6, 2), dtype=torch.float32)
    temb[..., 2, :] = 1.00390625
    temb[..., 5, :] = 1.001953125

    output = _run_block(block, residual, temb)

    # Casting either gate to BF16 before multiplying loses its fractional part.
    # The first residual is 1/256; the second rounds 1.005859375 to 1.0078125.
    expected_attention = torch.full_like(residual, 0.00390625)
    expected_output = torch.full_like(residual, 1.0078125)
    torch.testing.assert_close(block.norm2.input, expected_attention, rtol=0, atol=0)
    torch.testing.assert_close(block.norm3.input, expected_attention, rtol=0, atol=0)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize("gate_index", [2, 5], ids=["attention", "ffn"])
def test_bf16_product_rounds_before_residual_addition(gate_index: int) -> None:
    residual = torch.full((2, 3, 2), -1.015625, dtype=torch.bfloat16)
    branch = torch.full_like(residual, 1.0078125)
    block = _make_block(
        torch.zeros(1, 6, 2, dtype=torch.bfloat16),
        attention=branch,
        cross_attention=torch.zeros_like(residual),
        feed_forward=branch,
    )
    temb = torch.zeros(2, 6, 2, dtype=torch.bfloat16)
    temb[:, gate_index] = 1.0078125

    output = _run_block(block, residual, temb)

    # BF16 rounds 1.0078125**2 to 1.015625 before the subtraction, giving zero.
    # Accumulating everything in FP32 and casting only once instead gives 2**-14.
    torch.testing.assert_close(output, torch.zeros_like(residual), rtol=0, atol=0)
