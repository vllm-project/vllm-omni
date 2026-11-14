# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Rope,
    QwenImage21SequencePrepare,
    _apply_qwen_image21_rotary_emb_native,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_mixed_prompt_padding_keeps_each_image_frame_position():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = QwenImage21SequencePrepare(nn.Identity(), rope)
    layout = [(1, 2, 2)]
    image_slots = torch.tensor([[False] * 6 + [True]] * 2)
    text_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1] * 6])

    _, _, _, target_freqs = prepare(
        torch.zeros(2, 4, 64),
        torch.zeros(2, 6, 64),
        layout,
        image_slots,
        False,
        text_mask,
    )

    short = rope(layout, torch.tensor([False] * 4 + [True] * 4), torch.device("cpu"))[-4:]
    long = rope(layout, torch.tensor([False] * 6 + [True] * 4), torch.device("cpu"))[-4:]
    assert target_freqs.shape == (2, 4, 64)
    assert torch.equal(target_freqs[0], short)
    assert torch.equal(target_freqs[1], long)


def test_mixed_prompt_padding_with_reference_image_keeps_target_frame_position():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = QwenImage21SequencePrepare(nn.Identity(), rope)
    layout = [(1, 2, 2), (1, 2, 2)]
    image_slots = torch.tensor([[False] * 2 + [True] + [False] * 6 + [True]] * 2)
    text_mask = torch.tensor([[0] + [1] * 8, [1] * 9])

    _, _, _, target_freqs = prepare(
        torch.zeros(2, 8, 64),
        torch.zeros(2, 9, 64),
        layout,
        image_slots,
        False,
        text_mask,
    )

    short_mask = torch.tensor([False, True] + [False] * 6 + [True])
    short = rope(layout, prepare.expand_image_pad_mask(short_mask), torch.device("cpu"))[-4:]
    long = rope(layout, prepare.expand_image_pad_mask(image_slots[0]), torch.device("cpu"))[-4:]
    assert torch.equal(target_freqs[0], short)
    assert torch.equal(target_freqs[1], long)


def test_batched_rotary_matches_individual_requests():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.tensor([False] * 6 + [True] * 4)
    text_valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1] * 10])
    freqs = rope([(1, 2, 2)], image_pad_mask, torch.device("cpu"), text_valid)
    x = torch.randn(2, 10, 1, 128, dtype=torch.float32)

    actual = _apply_qwen_image21_rotary_emb_native(x, freqs)
    expected = torch.cat([_apply_qwen_image21_rotary_emb_native(x[i : i + 1], freqs[i]) for i in range(2)])
    assert torch.equal(actual, expected)
