# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 NetEase Youdao
"""Confucius4-TTS Stage 1 latent recompute over a finished semantic sequence.

The S2A decoder consumes one hidden row per semantic token. Upstream derives
those rows by running the text-to-semantic talker once over the completed
``[BOS | codes | EOS]`` sequence and slicing out the semantic span, so the
latent is a pure function of the final token sequence. This module reproduces
that pass in float32 without paged attention or a KV cache.

The parameter layout mirrors the official ``Text2Semantic`` checkpoint
(``transformer.h.N`` / ``transformer.ln_f``, GPT2 ``Conv1D`` weights stored as
``[in, out]``), so checkpoint tensors load without renaming or transposition.
The GPT2 blocks are written in plain torch to stay independent of
``transformers`` internals; tests pin them against ``transformers.GPT2Model``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from .configuration_confucius4_tts import Confucius4TTSConfig
from .position_embeddings import LearnedPositionalEmbedding
from .speaker_encoder import build_speaker_encoder
from .text_encoder import TextEmbeddingProjector

logger = init_logger(__name__)

_LAYER_NORM_EPS = 1e-5


class _Conv1D(nn.Module):
    """GPT2 ``Conv1D``: a linear layer whose weight is stored as ``[in, out]``."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).reshape(*x.shape[:-1], -1)


class _Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.c_attn = _Conv1D(hidden_size, 3 * hidden_size)
        self.c_proj = _Conv1D(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=-1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, length, self.num_heads, -1).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(batch, length, width))


class _MLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.c_fc = _Conv1D(hidden_size, 4 * hidden_size)
        self.c_proj = _Conv1D(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``gelu_new`` in transformers is the tanh approximation.
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class _Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=_LAYER_NORM_EPS)
        self.attn = _Attention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=_LAYER_NORM_EPS)
        self.mlp = _MLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Confucius4TTST2SRecompute(nn.Module):
    """Float32 mirror of the talker used to rebuild Stage 1 latents."""

    def __init__(self, config: Confucius4TTSConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config
        self.text_projector = TextEmbeddingProjector(
            config.text_vocab_size,
            config.text_embedding_dim,
            config.hidden_size,
        )
        self.semantic_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_position_embedding = LearnedPositionalEmbedding(
            config.max_text_tokens,
            config.hidden_size,
        )
        self.semantic_position_embedding = LearnedPositionalEmbedding(
            config.max_semantic_tokens,
            config.hidden_size,
        )
        self.speaker_encoder = build_speaker_encoder(config)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            _Block(config.hidden_size, config.num_attention_heads) for _ in range(config.num_layers)
        )
        self.transformer.ln_f = nn.LayerNorm(config.hidden_size, eps=_LAYER_NORM_EPS)

    @torch.no_grad()
    def load_state(self, state: dict[str, torch.Tensor]) -> set[str]:
        """Load talker weights, ignoring the training-only heads.

        Returns the names of the parameters that were loaded.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        unexpected: list[str] = []
        for name, weight in state.items():
            if name not in params:
                if not name.startswith(("final_norm.", "semantic_head.")):
                    unexpected.append(name)
                continue
            if tuple(weight.shape) != tuple(params[name].shape):
                raise RuntimeError(
                    f"Confucius4-TTS latent recompute shape mismatch for {name}: "
                    f"{tuple(weight.shape)} != {tuple(params[name].shape)}"
                )
            params[name].copy_(weight)
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                "Confucius4-TTS latent recompute checkpoint is missing parameters: " + ", ".join(missing[:8])
            )
        if unexpected:
            logger.warning(
                "Ignored %d unexpected Confucius4-TTS recompute keys: %s",
                len(unexpected),
                ", ".join(sorted(unexpected)[:8]),
            )
        return loaded

    @torch.no_grad()
    def forward(
        self,
        text_token_ids: torch.Tensor,
        condition_vector: torch.Tensor,
        semantic_codes: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Return the latent rows aligned with ``semantic_codes``.

        Args:
            text_token_ids: Formatted prompt token ids, shape (T_text,).
            condition_vector: Wav2Vec2-BERT reference features, shape (T_cond, D)
                or (1, T_cond, D).
            semantic_codes: Generated semantic ids without BOS/EOS, shape (T_sem,).
            bos_token_id: Semantic BOS id placed before the codes.
            eos_token_id: Semantic EOS id placed after the codes.

        Returns:
            Latent of shape (T_sem, hidden_size). Row ``i`` is the hidden state
            that predicted ``semantic_codes[i]``.
        """
        cfg = self.config
        device = next(self.parameters()).device
        text_ids = torch.as_tensor(text_token_ids, dtype=torch.long, device=device).reshape(1, -1)
        codes = torch.as_tensor(semantic_codes, dtype=torch.long, device=device).reshape(1, -1)
        if text_ids.shape[1] > cfg.max_text_tokens:
            raise ValueError(f"Confucius4-TTS formatted prompt exceeds {cfg.max_text_tokens} tokens")
        # BOS and EOS occupy semantic positions 0 and T + 1.
        if codes.shape[1] + 2 > cfg.max_semantic_tokens:
            raise ValueError(f"Confucius4-TTS semantic sequence exceeds {cfg.max_semantic_tokens - 2} tokens")

        condition = torch.as_tensor(condition_vector, dtype=torch.float32, device=device)
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)

        speaker_emb = self.speaker_encoder(condition).unsqueeze(1)
        text_emb = self.text_position_embedding(self.text_projector(text_ids))
        bos = torch.full((1, 1), int(bos_token_id), dtype=torch.long, device=device)
        eos = torch.full((1, 1), int(eos_token_id), dtype=torch.long, device=device)
        semantic_emb = self.semantic_position_embedding(self.semantic_embedding(torch.cat([bos, codes, eos], dim=1)))
        hidden = torch.cat([speaker_emb, text_emb, semantic_emb], dim=1)

        for block in self.transformer.h:
            hidden = block(hidden)
        hidden = self.transformer.ln_f(hidden)

        # Sequence: [speaker | text | BOS | c_1 .. c_T | EOS]. Keeping the BOS
        # position through c_{T-1} yields the T states that predicted c_1 .. c_T.
        start = 1 + text_ids.shape[1]
        return hidden[0, start:-2].contiguous()


__all__ = ["Confucius4TTST2SRecompute"]
