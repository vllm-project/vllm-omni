# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed import sp_sharding
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    _pad_text_inputs_for_sp,
    _split_text_embed_in_sp_from_extras,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_EXTRA_NAME = "qwen_image_split_text_embed_in_sp"


def test_split_text_embed_extra_defaults_to_false():
    assert not _split_text_embed_in_sp_from_extras(OmniDiffusionConfig(extras={}))


@pytest.mark.parametrize("value", [True, False])
def test_split_text_embed_extra_accepts_booleans(value):
    config = OmniDiffusionConfig(extras={_EXTRA_NAME: value})

    assert _split_text_embed_in_sp_from_extras(config) is value


@pytest.mark.parametrize("value", ["true", 1, None])
def test_split_text_embed_extra_rejects_non_bool(value):
    config = OmniDiffusionConfig(extras={_EXTRA_NAME: value})

    with pytest.raises(TypeError, match=rf"{_EXTRA_NAME} must be a bool"):
        _split_text_embed_in_sp_from_extras(config)


def test_text_sp_padding_aligns_embeddings_mask_and_rope(monkeypatch):
    sp_size = 8
    encoder_hidden_states = torch.randn(1, 10, 4)
    encoder_hidden_states_mask = torch.tensor([[True] * 6 + [False] * 4])

    padded_hidden_states, padded_mask, txt_seq_lens = _pad_text_inputs_for_sp(
        encoder_hidden_states,
        encoder_hidden_states_mask,
        [encoder_hidden_states.shape[1]],
        sp_size,
    )

    assert padded_hidden_states.shape == (1, 16, 4)
    assert padded_mask is not None
    torch.testing.assert_close(padded_mask[:, :10], encoder_hidden_states_mask)
    assert padded_mask[:, 10:].all()
    assert txt_seq_lens == [16]

    monkeypatch.setattr(sp_sharding, "get_sequence_parallel_world_size", lambda: sp_size)
    monkeypatch.setattr(sp_sharding, "get_sequence_parallel_rank", lambda: sp_size - 1)
    txt_freqs = torch.randn(16, 2)

    hidden_states_shard = sp_sharding.sp_shard(padded_hidden_states, dim=1)
    mask_shard = sp_sharding.sp_shard(padded_mask, dim=1)
    txt_freqs_shard = sp_sharding.sp_shard(txt_freqs, dim=0)

    assert hidden_states_shard.shape[1] == mask_shard.shape[1] == txt_freqs_shard.shape[0] == 2
    assert not hidden_states_shard.any()
    assert mask_shard.all()


def test_text_sp_padding_is_noop_when_divisible():
    encoder_hidden_states = torch.randn(1, 16, 4)
    encoder_hidden_states_mask = torch.ones(1, 16, dtype=torch.bool)
    txt_seq_lens = [16]

    padded_hidden_states, padded_mask, padded_txt_seq_lens = _pad_text_inputs_for_sp(
        encoder_hidden_states,
        encoder_hidden_states_mask,
        txt_seq_lens,
        sp_size=8,
    )

    assert padded_hidden_states is encoder_hidden_states
    assert padded_mask is encoder_hidden_states_mask
    assert padded_txt_seq_lens is txt_seq_lens
