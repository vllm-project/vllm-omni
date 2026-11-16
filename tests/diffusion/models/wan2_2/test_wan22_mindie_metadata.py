# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan integration contracts required by quantized dense/sparse attention."""

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformerBlock

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_block_publishes_unpadded_grid_without_partial_packed_metadata():
    block = object.__new__(WanTransformerBlock)
    torch.nn.Module.__init__(block)
    block.scale_shift_table = torch.zeros(1, 6, 4)
    block.norm1 = lambda x, *a: x
    block.norm2 = lambda x: x
    block.norm3 = lambda x, *a: x
    block.attn1 = Mock(side_effect=lambda x, *a: torch.zeros_like(x))
    block.attn2 = Mock(side_effect=lambda x, *a: torch.zeros_like(x))
    block.ffn = lambda x: torch.zeros_like(x)
    # Local SP tokens include alignment padding; grid describes the full valid video.
    x = torch.zeros(1, 1152, 4)
    block(
        x,
        x,
        torch.zeros(1, 6, 4),
        None,
        hidden_states_mask=torch.ones(1, 1152, dtype=torch.bool),
        vsa_dit_seq_shape=(5, 16, 27),
    )
    metadata = block.attn1.call_args.args[2]
    assert metadata.video_layout.latent_grid == (5, 16, 27)
    assert metadata.video_layout.used_len == 2160
    assert metadata.video_layout.prefix_len == 0
    assert "max_seqlen_q" not in metadata.extra  # CUDA must not infer incomplete packed varlen.
    assert block.attn2.call_args.args[2] is None
