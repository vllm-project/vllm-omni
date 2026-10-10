# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original T3 conditioning, matching Resemble AI's MIT-licensed T3 modules."""

from collections.abc import Iterable

import torch
from torch import nn


class OriginalAttention(nn.Module):
    """Shared cross/self attention with upstream checkpoint parameter names."""

    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query_norm, context_norm = self.norm(queries), self.norm(context)
        q, k, v = [
            projection(value).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
            for projection, value in ((self.to_q, query_norm), (self.to_k, context_norm), (self.to_v, context_norm))
        ]
        attended = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.2 if self.training else 0.0)
        return queries + self.proj_out(attended.transpose(1, 2).flatten(2))


class OriginalPerceiver(nn.Module):
    """The same attention parameters resample and then mix 32 learned queries."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, 32, hidden_size))
        nn.init.uniform_(self.pre_attention_query, -((3 / 32) ** 0.5), (3 / 32) ** 0.5)
        self.attn = OriginalAttention(hidden_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        queries = self.pre_attention_query.tile(context.shape[0], 1, 1)
        attended = self.attn(queries, context)
        return self.attn(attended, attended)


class OriginalConditioning(nn.Module):
    """Original speaker, reference speech and emotion projections."""

    def __init__(self, hidden_size: int, speaker_embed_size: int) -> None:
        super().__init__()
        self.spkr_enc = nn.Linear(speaker_embed_size, hidden_size)
        self.emotion_adv_fc = nn.Linear(1, hidden_size, bias=False)
        self.perceiver = OriginalPerceiver(hidden_size)


class OriginalPositions(nn.Module):
    """Preserve the upstream learned-position checkpoint namespace."""

    def __init__(self, length: int, hidden_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(length, hidden_size)


class OriginalHeads(nn.Module):
    """Original T3 modules outside the LLaMA backbone.

    Config supplies hidden_size, text_vocab_size, speech_vocab_size and
    speaker_embed_size. Learned positions and resampler length are fixed by
    the English Original checkpoint.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.text_emb = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, config.hidden_size)
        self.speech_head = nn.Linear(config.hidden_size, config.speech_vocab_size, bias=False)
        self.text_pos_emb = OriginalPositions(2050, config.hidden_size)
        self.speech_pos_emb = OriginalPositions(4100, config.hidden_size)
        self.cond_enc = OriginalConditioning(config.hidden_size, config.speaker_embed_size)


def original_speech_embeds(heads: OriginalHeads, ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Embed generated tokens with speech-local positions (first token is 1)."""
    return heads.speech_emb(ids) + heads.speech_pos_emb.emb(positions)


def original_prefill_embeds(
    heads: OriginalHeads,
    text_ids: torch.Tensor,
    cond_tokens: torch.Tensor,
    speaker_emb: torch.Tensor,
    exaggeration: float = 0.5,
    unconditional: bool = False,
) -> torch.Tensor:
    """Build one raw-text request's exact Original prompt, shape (T + 38, H).

    Inputs have shapes (T,), (C,), and (1, speaker_embed_size). CFG retains
    speaker, reference and emotion; only text content embeddings are zeroed.
    Both initial speech tokens use learned speech position zero upstream.
    """
    device = text_ids.device
    text_ids = torch.cat((text_ids.new_tensor([255]), text_ids, text_ids.new_tensor([0])))
    text = heads.text_emb(text_ids)
    if unconditional:
        text = torch.zeros_like(text)
    text = text + heads.text_pos_emb.emb(torch.arange(text_ids.shape[0], device=device))
    reference = original_speech_embeds(heads, cond_tokens, torch.arange(cond_tokens.shape[0], device=device))
    reference = heads.cond_enc.perceiver(reference[None])[0]
    emotion = heads.cond_enc.emotion_adv_fc(text.new_tensor([[exaggeration]]))
    bos = original_speech_embeds(heads, text_ids.new_tensor([6561]), text_ids.new_tensor([0])).tile(2, 1)
    return torch.cat((heads.cond_enc.spkr_enc(speaker_emb), reference, emotion, text, bos))


def split_original_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Split checkpoint keys, rejecting every unknown inference parameter."""
    head_names = {
        "text_emb.weight",
        "speech_emb.weight",
        "speech_head.weight",
        "text_pos_emb.emb.weight",
        "speech_pos_emb.emb.weight",
        "cond_enc.spkr_enc.weight",
        "cond_enc.spkr_enc.bias",
        "cond_enc.emotion_adv_fc.weight",
        "cond_enc.perceiver.pre_attention_query",
    }
    head_names.update(
        f"cond_enc.perceiver.attn.{module}.{parameter}"
        for module in ("norm", "to_q", "to_k", "to_v", "proj_out")
        for parameter in ("weight", "bias")
    )
    backbone, heads = [], {}
    for name, tensor in weights:
        if name.startswith("tfmr."):
            backbone.append((name.removeprefix("tfmr."), tensor))
        elif name in head_names:
            heads[name] = tensor
        elif name != "text_head.weight":
            raise KeyError(f"unexpected key {name!r} in the Original T3 checkpoint")
    return backbone, heads
