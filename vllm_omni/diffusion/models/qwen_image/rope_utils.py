# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def txt_seq_lens_from_embeds(prompt_embeds: torch.Tensor | None) -> list[int] | None:
    """Return per-row RoPE text lengths from padded encoder hidden-state width.

    Diffusers derives Qwen-Image text RoPE length from
    ``encoder_hidden_states.shape[1]``, not from the count of valid (non-padding)
    tokens in the attention mask. Callers that right-pad prompt embeddings for
    batching must pass the padded width here so ``QwenEmbedRope`` builds a table
    long enough for every text position.
    """
    if prompt_embeds is None:
        return None
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds.ndim != 3:
        raise ValueError(f"prompt_embeds must be 2D or 3D, got shape={tuple(prompt_embeds.shape)}")
    seq_len = int(prompt_embeds.shape[1])
    batch_size = int(prompt_embeds.shape[0])
    return [seq_len] * batch_size
