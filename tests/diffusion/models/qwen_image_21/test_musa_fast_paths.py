# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MUSA paths of the Qwen-Image-2.1 transformer fast paths."""

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.qwen_image_21 import (
    qwen_image_21_transformer as transformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _require_musa() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")


class _ScaledAttention(nn.Module):
    def forward(self, values, *args, **kwargs):
        return values * 0.37


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_shared_modulation_matches_per_block_modulation():
    _require_musa()
    block = transformer.QwenImage21TransformerBlock.__new__(transformer.QwenImage21TransformerBlock)
    nn.Module.__init__(block)
    block.img_norm1 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    block.img_norm2 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    block.attn = _ScaledAttention()
    block.img_mlp = nn.SiLU()
    block.to("musa")
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(2, 17, 128, generator=generator).to("musa", torch.bfloat16)
    modulation = torch.randn(2, 512, generator=generator).to("musa", torch.bfloat16)
    select = transformer._select_modulation_rows
    prepared = tuple(
        1 + select(x, None) if i % 2 == 0 else select(x, None).tanh() for i, x in enumerate(modulation.chunk(4, dim=-1))
    )
    expected = block(values, modulation, None, target_token_mask=None)
    actual = block(values, modulation, None, target_token_mask=None, prepared_modulation=prepared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
