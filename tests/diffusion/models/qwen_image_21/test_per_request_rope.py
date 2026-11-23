# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Rope,
    QwenImage21SequencePrepare,
    _apply_qwen_image21_rotary_emb_native,
    _qwen_image21_rope_text_mask,
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


def test_singleton_decode_after_mixed_prefill_keeps_target_frame_position():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = QwenImage21SequencePrepare(nn.Identity(), rope)
    layout = [(1, 2, 2)]
    image_slots = torch.tensor([[False] * 6 + [True]] * 2)
    text_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1] * 6])

    _, _, _, prefill_target = prepare(
        torch.zeros(2, 4, 64), torch.zeros(2, 6, 64), layout, image_slots, False, text_mask
    )
    short_request_mask = text_mask[:1]
    short_mask = _qwen_image21_rope_text_mask(short_request_mask, torch.device("cuda"), sp_active=False)
    assert short_mask is short_request_mask
    prepare._request_geometry = {}
    prefix, _, _, decode_target = prepare(torch.zeros(1, 4, 64), None, layout, image_slots[:1], True, short_mask)

    assert prefix.shape[1] == 0
    assert decode_target.shape == (1, 4, 64)
    assert torch.equal(decode_target[0], prefill_target[0])
    assert torch.equal(prepare._request_geometry["rotary_emb"][0, -4:], prefill_target[0])


@pytest.mark.parametrize("device,sp_active", [("cpu", False), ("cuda", True)])
def test_unsupported_mixed_padding_is_rejected(device, sp_active):
    text_mask = torch.tensor([[0, 1, 1]])
    with pytest.raises(ValueError, match="mixed-length text padding requires CUDA or NPU without sequence parallelism"):
        _qwen_image21_rope_text_mask(text_mask, torch.device(device), sp_active)


def test_batched_rotary_matches_individual_requests():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.tensor([False] * 6 + [True] * 4)
    text_valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1] * 10])
    freqs = rope([(1, 2, 2)], image_pad_mask, torch.device("cpu"), text_valid)
    x = torch.randn(2, 10, 1, 128, dtype=torch.float32)

    actual = _apply_qwen_image21_rotary_emb_native(x, freqs)
    expected = torch.cat([_apply_qwen_image21_rotary_emb_native(x[i : i + 1], freqs[i]) for i in range(2)])
    assert torch.equal(actual, expected)
