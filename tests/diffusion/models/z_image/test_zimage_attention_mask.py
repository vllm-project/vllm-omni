# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.models.z_image import z_image_transformer
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    ZImageAttention,
    ZImageTransformerBlock,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _IdentityQKV(nn.Module):
    num_heads = 1
    num_kv_heads = 1

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.cat([hidden_states, hidden_states, hidden_states], dim=-1), None


class _RecordingMaskedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masks: list[torch.Tensor | None] = []
        self.use_ring = False
        self._no_parallel_strategy = object()
        self.active_parallel_strategy = self._no_parallel_strategy

    def _get_active_parallel_strategy(self):
        return self.active_parallel_strategy

    def forward(self, query, key, value, attn_metadata=None):
        attention_mask = None if attn_metadata is None else attn_metadata.attn_mask
        self.masks.append(attention_mask)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        )
        return output.transpose(1, 2)


class _ZeroFeedForward(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class _MainTransformerModulation(nn.Module):
    def forward(self, adaln_input: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(adaln_input)
        return torch.cat([zeros, torch.ones_like(adaln_input), zeros, zeros], dim=-1)


def _make_attention() -> ZImageAttention:
    attention = ZImageAttention.__new__(ZImageAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 4
    attention.to_qkv = _IdentityQKV()
    attention.norm_q = nn.Identity()
    attention.norm_k = nn.Identity()
    attention.rope = object()
    attention.attn = _RecordingMaskedAttention()
    attention.to_out = nn.ModuleList([nn.Identity()])
    return attention


def _make_block(*, modulation: bool) -> ZImageTransformerBlock:
    block = ZImageTransformerBlock.__new__(ZImageTransformerBlock)
    nn.Module.__init__(block)
    block.modulation = modulation
    block.attention = _make_attention()
    block.feed_forward = _ZeroFeedForward()
    block.attention_norm1 = nn.Identity()
    block.attention_norm2 = nn.Identity()
    block.ffn_norm1 = nn.Identity()
    block.ffn_norm2 = nn.Identity()
    if modulation:
        block.adaLN_modulation = _MainTransformerModulation()
    return block


@pytest.mark.parametrize(
    "modulation",
    [
        pytest.param(False, id="context-refiner"),
        pytest.param(True, id="main-transformer"),
    ],
)
def test_valid_tokens_are_invariant_to_other_prompt_padding(monkeypatch, modulation: bool):
    """A short sample must not attend to padding introduced by a longer peer."""
    monkeypatch.setattr(
        z_image_transformer,
        "apply_rope_to_qk",
        lambda _rope, query, key, _freqs: (query, key),
    )
    block = _make_block(modulation=modulation)

    short = torch.tensor([[[1.0, 0.5, 0.25, 0.75], [0.5, 1.0, 0.75, 0.25], [0.25, 0.75, 1.0, 0.5]]])
    valid_length = short.shape[1]
    padded_length = 6
    padded_short = torch.cat(
        [short, torch.full((1, padded_length - valid_length, short.shape[-1]), 10.0)],
        dim=1,
    )
    long_peer = torch.linspace(0.1, 2.4, padded_length * short.shape[-1]).reshape(1, padded_length, short.shape[-1])
    batched = torch.cat([padded_short, long_peer], dim=0)

    single_mask = torch.ones((1, valid_length), dtype=torch.bool)
    batched_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
        ]
    )
    single_rope = torch.zeros((1, valid_length, 1))
    batched_rope = torch.zeros((2, padded_length, 1))
    single_adaln = torch.zeros((1, short.shape[-1])) if modulation else None
    batched_adaln = torch.zeros((2, short.shape[-1])) if modulation else None

    single_output = block(short, single_mask, single_rope, single_rope, single_adaln)
    batched_output = block(batched, batched_mask, batched_rope, batched_rope, batched_adaln)

    torch.testing.assert_close(batched_output[0, :valid_length], single_output[0])
    recorded_masks = block.attention.attn.masks
    assert len(recorded_masks) == 2
    assert all(mask is not None and mask.ndim == 2 for mask in recorded_masks)
    assert torch.equal(recorded_masks[0], single_mask)
    assert torch.equal(recorded_masks[1], batched_mask)


def test_variable_padding_mask_rejects_active_ring_attention(monkeypatch):
    monkeypatch.setattr(
        z_image_transformer,
        "apply_rope_to_qk",
        lambda _rope, query, key, _freqs: (query, key),
    )
    attention = _make_attention()
    attention.attn.use_ring = True
    attention.attn.active_parallel_strategy = object()

    hidden_states = torch.ones((2, 4, 4))
    attention_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    rope = torch.zeros((2, 4, 1))

    with pytest.raises(ValueError, match="not supported with ring sequence parallelism"):
        attention(hidden_states, attention_mask, rope, rope)
