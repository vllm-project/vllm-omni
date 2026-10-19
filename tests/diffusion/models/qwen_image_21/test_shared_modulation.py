# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21TransformerBlock,
    _select_modulation_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class ToyAttention(nn.Module):
    def forward(self, values, *args, **kwargs):
        return values * 0.37


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask", [False, True])
@torch.no_grad()
def test_shared_modulation_is_exact_and_request_owned(dtype, mask):
    block = QwenImage21TransformerBlock.__new__(QwenImage21TransformerBlock)
    nn.Module.__init__(block)
    block.img_norm1 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    block.img_norm2 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    block.attn = ToyAttention()
    block.img_mlp = nn.SiLU()
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(2, 17, 128, generator=generator).to(dtype)
    modulation = torch.randn(3 if mask else 2, 512, generator=generator).to(dtype)
    local_mask = torch.tensor([False] * 5 + [True] * 12) if mask else None
    scales_and_gates = modulation.chunk(4, dim=-1)
    prepared = tuple(
        1 + _select_modulation_rows(x, local_mask) if i % 2 == 0 else _select_modulation_rows(x, local_mask).tanh()
        for i, x in enumerate(scales_and_gates)
    )
    originals = [x.clone() for x in prepared]
    expected = block(values, modulation, None, target_token_mask=local_mask)
    actual = block(values, modulation, None, target_token_mask=local_mask, prepared_modulation=prepared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for original, result in zip(originals, prepared):
        torch.testing.assert_close(original, result, rtol=0, atol=0)
