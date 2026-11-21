# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import ErnieImageSharedAdaLNBlock

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ViewAttention(nn.Module):
    def forward(self, hidden_states, *, attention_mask, image_rotary_emb):
        # The online FP8 linear kernel flattens its input with this view.
        hidden_states.view(-1, hidden_states.shape[-1])
        return hidden_states


@pytest.mark.parametrize("batch_size", [1, 2])
def test_attention_input_is_compatible_with_fp8_linear_view(batch_size):
    block = ErnieImageSharedAdaLNBlock.__new__(ErnieImageSharedAdaLNBlock)
    nn.Module.__init__(block)
    block.adaLN_sa_ln = nn.Identity()
    block.adaLN_mlp_ln = nn.Identity()
    block.self_attention = _ViewAttention()
    block.mlp = nn.Identity()

    hidden_states = torch.randn(3, batch_size, 8)
    zero = torch.zeros(1, 1, 8)
    one = torch.ones_like(zero)
    temb = (zero, zero, one, zero, zero, zero)

    output = block(hidden_states, rotary_pos_emb=None, temb=temb)

    torch.testing.assert_close(output, 2 * hidden_states)
