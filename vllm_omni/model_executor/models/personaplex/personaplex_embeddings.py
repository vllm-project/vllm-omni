# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex input embeddings (Moshi ``embed_codes``).

The temporal transformer (:class:`HeliumForCausalLM`) consumes precomputed
``inputs_embeds`` rather than token ids. Those embeddings come from Moshi's
``embed_codes``: a per-frame ``[B, 1 + n_q, S]`` token stack (row 0 = text,
rows 1..n_q = audio codebooks) is mapped through one text embedding plus ``n_q``
audio-codebook embeddings and summed into ``[B, S, hidden]``.

This module owns exactly those embedding tables (Moshi ``text_emb`` and
``emb.0..n_q-1``), which ``modeling_helium`` deliberately omits. The talker
(Task 2) uses it to build the temporal ``inputs_embeds`` for both the system /
voice-prompt prefill and the per-frame decode step.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    _ScaledEmbedding,
)

__all__ = ["PersonaPlexInputEmbeddings"]


class PersonaPlexInputEmbeddings(nn.Module):
    """Moshi ``embed_codes``: summed text + audio-codebook embeddings.

    Args:
        config: the top-level PersonaPlex config (supplies the temporal hidden
            size, the audio cardinality / codebook count, and the text table size).
    """

    def __init__(self, config: PersonaPlexConfig) -> None:
        super().__init__()
        dim = config.temporal_config.hidden_size
        self.num_audio_codebooks = config.num_audio_codebooks  # n_q (16)
        # Text table has one extra row for the special token (text_card + 1).
        self.text_emb = _ScaledEmbedding(config.text_embedding_rows, dim)
        # Each audio table is card + 1 rows; ``embed_codes`` uses all n_q codebooks.
        self.audio_emb = nn.ModuleList(
            [_ScaledEmbedding(config.audio_vocab_size + 1, dim) for _ in range(self.num_audio_codebooks)]
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Map a ``[B, 1 + n_q, S]`` token stack to ``[B, S, hidden]``.

        Row 0 is the text stream; rows ``1..n_q`` are the audio codebooks (in
        Moshi's layout the audio rows carry both the agent and user streams).
        """
        if sequence.dim() != 3 or sequence.shape[1] != self.num_audio_codebooks + 1:
            raise ValueError(f"sequence must be [B, {self.num_audio_codebooks + 1}, S]; got {tuple(sequence.shape)}")
        emb = self.text_emb(sequence[:, 0])  # [B, S, hidden]
        for cb in range(self.num_audio_codebooks):
            emb = emb + self.audio_emb[cb](sequence[:, cb + 1])
        return emb

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]] | Mapping[str, torch.Tensor],
    ) -> set[str]:
        """Load Moshi ``text_emb`` and ``emb.0..n_q-1`` embedding tables."""
        src = dict(weights.items() if isinstance(weights, Mapping) else weights)
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        plan: dict[str, str] = {"text_emb.weight": "text_emb.weight"}
        for cb in range(self.num_audio_codebooks):
            plan[f"audio_emb.{cb}.weight"] = f"emb.{cb}.weight"

        for tgt, key in plan.items():
            if key not in src:
                continue
            param = params[tgt]
            tensor = src[key]
            if tensor.shape != param.shape:
                raise ValueError(
                    f"shape mismatch for {tgt}: checkpoint {tuple(tensor.shape)} vs param {tuple(param.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(tgt)
        return loaded
