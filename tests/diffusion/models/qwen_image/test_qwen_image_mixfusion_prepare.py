# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for ``prepare_mixfusion_inputs`` and its step-invariant layout cache (CPU).

``prepare_mixfusion_inputs`` owns all packed-layout preparation: row expansion,
chunk split, RoPE tables and the block-agnostic packing plan. Within one
generation's denoise loop only the latents change, so the bundle from the
previous step can be handed back as ``cached``: the plan, RoPE tables and
chunk size are reused verbatim while the fresh latents are re-split and
re-projected.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

ROPE_DIM = 32


class _PlainBlock(nn.Module):
    def forward(self, x):
        return x


def _transformer() -> tuple[QwenImageTransformer2DModel, list]:
    """A transformer with only the prepare path's dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed;
    ``pos_embed`` / ``img_in`` are stubbed with plain attributes so the RoPE
    tables are zeros of the right shape and ``img_in`` is an identity (the
    prepared ``hidden_states`` then equals the chunk-stacked latents).
    ``pos_embed_calls`` records how often the (step-invariant) tables were
    rebuilt.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    object.__setattr__(transformer, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    object.__setattr__(transformer, "transformer_blocks", [_PlainBlock()])
    object.__setattr__(transformer, "img_in", nn.Identity())
    pos_embed_calls: list[int] = [0]

    def pos_embed(img_shapes, txt_seq_lens, device):
        # img_shapes entry: (img_token_count, ...); freq rows must cover the
        # image row length and at least the text length.
        pos_embed_calls[0] += 1
        img_tokens = int(img_shapes[0])
        txt_len = int(txt_seq_lens[0])
        return torch.zeros(img_tokens, ROPE_DIM), torch.zeros(txt_len + 4, ROPE_DIM)

    object.__setattr__(transformer, "pos_embed", pos_embed)
    return transformer, pos_embed_calls


def _prepare(transformer, latents, *, prompt_rows, seq_len_txt=4, cached=None, img_shapes=None):
    if img_shapes is None:
        img_shapes = [(int(latent.shape[1]), 2, 2) for latent in latents]
    return transformer.prepare_mixfusion_inputs(
        hidden_states=latents,
        encoder_hidden_states=torch.zeros(prompt_rows, seq_len_txt, 8),
        encoder_hidden_states_mask=None,
        img_shapes=img_shapes,
        txt_seq_lens=[seq_len_txt] * len(latents),
        cached=cached,
    )


def test_prepare_builds_packed_inputs():
    transformer, _ = _transformer()
    latents = [torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)]
    prepared = _prepare(transformer, latents, prompt_rows=1)

    # img_in is an identity: hidden_states is the chunk-stacked latents.
    assert torch.equal(prepared.hidden_states, torch.cat([latents[0]], dim=0))
    assert prepared.chunk_size == 8
    assert prepared.image_row_counts == [1]
    assert prepared.plan is not None
    assert prepared.plan.chunk_to_request.tolist() == [0]
    # No mask given: the plan sees the full text length per row.
    assert prepared.encoder_hidden_states_mask is None
    # RoPE tables are stacked per chunk row / per image row.
    assert prepared.image_rotary_emb[0].shape == (1, 8, ROPE_DIM)
    assert prepared.image_rotary_emb[1].shape == (1, 4, ROPE_DIM)


def test_prepare_normalizes_all_valid_mask():
    transformer, _ = _transformer()
    latents = [torch.zeros(1, 8, 2)]
    prepared = transformer.prepare_mixfusion_inputs(
        hidden_states=latents,
        encoder_hidden_states=torch.zeros(1, 4, 8),
        encoder_hidden_states_mask=torch.ones(1, 4),
        img_shapes=[(8, 2, 2)],
        txt_seq_lens=[4],
    )
    assert prepared.encoder_hidden_states_mask is None


def test_prepare_reuses_step_invariant_layout_from_cached():
    transformer, pos_embed_calls = _transformer()
    latents_step_1 = [torch.ones(1, 8, 2), 2 * torch.ones(1, 8, 2)]
    latents_step_2 = [3 * torch.ones(1, 8, 2), 4 * torch.ones(1, 8, 2)]

    first = _prepare(transformer, latents_step_1, prompt_rows=2)
    assert pos_embed_calls[0] == 2  # one RoPE build per image row

    second = _prepare(transformer, latents_step_2, prompt_rows=2, cached=first)
    # Step-invariant layout reused verbatim: no RoPE or plan rebuild.
    assert pos_embed_calls[0] == 2
    assert second.plan is first.plan
    assert second.image_rotary_emb is first.image_rotary_emb
    assert second.row_vid_freqs is first.row_vid_freqs
    assert second.text_freqs is first.text_freqs
    assert second.chunk_size == first.chunk_size
    assert second.key == first.key
    # Per-step latents are re-split and re-projected.
    assert second.hidden_states is not first.hidden_states
    assert torch.equal(second.hidden_states, torch.cat(latents_step_2, dim=0))


def test_prepare_rebuilds_layout_when_shapes_change():
    transformer, pos_embed_calls = _transformer()
    first = _prepare(transformer, [torch.ones(1, 8, 2)], prompt_rows=1)
    assert pos_embed_calls[0] == 1

    # A different request mix (two rows, different length) must not reuse
    # the cached single-row layout.
    latents = [torch.ones(1, 8, 2), torch.ones(1, 8, 2)]
    second = _prepare(transformer, latents, prompt_rows=2, cached=first)

    assert pos_embed_calls[0] == 3  # rebuild: +2 rows
    assert second.plan is not first.plan
    assert second.key != first.key
    assert second.image_row_counts == [1, 1]
    assert second.hidden_states.shape == (2, 8, 2)


def test_prepare_splits_mixed_lengths_into_equal_chunks():
    transformer, _ = _transformer()
    latents = [torch.ones(1, 8, 2), torch.ones(1, 12, 2)]
    prepared = _prepare(transformer, latents, prompt_rows=2)

    # gcd(8, 12) = 4: 2 + 3 chunks, stacked on the batch dimension.
    assert prepared.chunk_size == 4
    assert prepared.hidden_states.shape == (5, 4, 2)
    assert prepared.plan.chunk_to_request.tolist() == [0, 0, 1, 1, 1]
