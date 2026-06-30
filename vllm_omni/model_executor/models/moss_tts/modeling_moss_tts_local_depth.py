# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Per-frame depth transformer for MossTTSLocalModel (MOSS-TTS-Local-Transformer-v1.5).

A 1-layer GPT2-style block that decodes the ``n_vq`` audio codebook codes for
one audio frame, run inside the talker's ``make_omni_output`` independent of
vLLM's main scheduler -- mirrors ``MossTTSRealtimeLocalTransformer``
(``modeling_moss_tts_local.py``) in role, but the algorithm and numerics are
faithful to the official ``gpt2_decoder.py`` / ``modeling_moss_tts.py``
(GPT2-style LayerNorm + bias, SiLU MLP, **interleaved/GPT-J-style RoPE** --
not vLLM's neox-style concat-half rotation) rather than Realtime's
Qwen3-style ``CodePredictorBaseModel``.

Its KV cache and positions reset to 0 every audio frame: there is no
cross-frame state. Submodule names (``h.0.*`` / ``ln_f``) match the
checkpoint 1:1 so ``load_weights()`` needs no remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
    _sample_token,
)


class _MossTTSLocalAttention(nn.Module):
    """GPT2-style fused-QKV self-attention with interleaved RoPE.

    Faithful to the official ``MossTTSNanoGPT2Attention``: ``rotate_half``
    operates on even/odd index pairs (GPT-J style), and ``cos``/``sin`` are
    built via ``repeat_interleave(2, dim=-1)`` rather than the neox-style
    concat-half construction vLLM uses elsewhere.
    """

    def __init__(self, hidden_size: int, n_head: int, rope_base: float) -> None:
        super().__init__()
        if hidden_size % n_head != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.embed_dim = hidden_size
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("s,d->sd", position_ids, self.inv_freq.to(device=device))
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype)
        return cos.view(1, seq_len, 1, self.head_dim), sin.view(1, seq_len, 1, self.head_dim)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        even = x[..., ::2]
        odd = x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).reshape_as(x)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states``: ``(B, S, H)``. Re-prefills with a fresh causal mask over ``[0, S)`` every call."""
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim)

        cos, sin = self._rope_cos_sin(seq_len, hidden_states.device, hidden_states.dtype)
        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.c_proj(attn_output)


class _MossTTSLocalMLP(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, inner_size, bias=True)
        self.fc_out = nn.Linear(inner_size, hidden_size, bias=True)
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.fc_in(hidden_states)))


class _MossTTSLocalBlock(nn.Module):
    def __init__(self, hidden_size: int, n_head: int, inner_size: int, rope_base: float, eps: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attn = _MossTTSLocalAttention(hidden_size, n_head, rope_base)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = _MossTTSLocalMLP(hidden_size, inner_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class MossTTSLocalDepthTransformer(nn.Module):
    """Per-frame depth transformer for MOSS-TTS-Local-Transformer-v1.5.

    Per frame:
      - position 0's input is the backbone's last hidden state; its output
        feeds BOTH the binary continue/stop head (``local_text_lm_head``)
        and codebook-0's head (``audio_lm_heads[0]``) simultaneously.
      - codebooks 1..n_vq-1 are sampled sequentially: each sampled code is
        re-embedded (``audio_embeddings[c]``) and appended as the next
        position, re-prefilling the block over the growing (<=n_vq) sequence
        with a fresh causal mask each call -- mathematically identical to
        incremental KV-cache decoding since attention is strictly causal and
        only the last position is ever read, but avoids any cache plumbing
        given the trivially short sequence length.
    """

    def __init__(self, gpt2_config, hidden_size: int | None = None) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size if hidden_size is not None else gpt2_config.n_embd)
        n_head = int(gpt2_config.n_head)
        inner_size = int(gpt2_config.n_inner)
        eps = float(getattr(gpt2_config, "layer_norm_epsilon", 1e-5))
        rope_base = float(getattr(gpt2_config, "rope_base", 1_000_000.0))
        self.h = nn.ModuleList([_MossTTSLocalBlock(self.hidden_size, n_head, inner_size, rope_base, eps)])
        self.ln_f = nn.LayerNorm(self.hidden_size, eps=eps)

    def _forward_prefix(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = seq_embeds
        for block in self.h:
            hidden_states = block(hidden_states)
        return self.ln_f(hidden_states)

    @torch.no_grad()
    def generate_frame(
        self,
        backbone_last_hidden: torch.Tensor,  # (B, H)
        audio_lm_heads: nn.ModuleList,  # n_vq x Linear(H -> audio_vocab_size)
        audio_embeddings: nn.ModuleList,  # n_vq x Embedding(audio_vocab_size, H)
        local_text_lm_head: nn.Module,  # Linear(H -> 2): [continue, stop]
        *,
        n_vq: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        text_temperature: float = 1.0,
        text_top_k: int = 50,
        text_top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        history_per_codebook: list[list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one audio frame for batch B.

        Returns ``(should_continue, codes)``: ``should_continue`` is a
        ``(B,)`` bool tensor (``True`` iff the binary head picked the
        "continue" candidate, i.e. logits index 0); ``codes`` is a
        ``(B, n_vq)`` LongTensor of sampled codebook indices.

        ``history_per_codebook[c]`` is a list of recently-emitted token ids
        for codebook ``c``; when ``repetition_penalty != 1.0`` those tokens'
        logits get scaled down before sampling (mirrors upstream's
        ``_apply_repetition_penalty``).
        """
        batch_size = backbone_last_hidden.shape[0]
        dtype = self.ln_f.weight.dtype

        embeds = backbone_last_hidden.new_zeros((batch_size, n_vq, self.hidden_size), dtype=dtype)
        embeds[:, 0, :] = backbone_last_hidden.to(dtype)

        hidden = self._forward_prefix(embeds[:, :1, :])
        local_hidden = hidden[:, 0, :]

        binary_logits = local_text_lm_head(local_hidden).float()
        binary_choice = _sample_token(binary_logits, text_temperature, text_top_k, text_top_p, do_sample)
        should_continue = binary_choice.eq(0)
        import os as _os

        if _os.environ.get("MOSS_TTS_DEBUG_STOP"):
            import logging as _logging

            _logging.getLogger("moss_tts_debug").warning(
                "binary_logits=%s choice=%s", binary_logits.tolist(), binary_choice.tolist()
            )

        codes = backbone_last_hidden.new_zeros((batch_size, n_vq), dtype=torch.long)
        for channel_index in range(n_vq):
            channel_logits = audio_lm_heads[channel_index](local_hidden).float()
            if (
                repetition_penalty != 1.0
                and history_per_codebook is not None
                and channel_index < len(history_per_codebook)
            ):
                hist = history_per_codebook[channel_index]
                if hist:
                    hist_t = torch.tensor(hist, dtype=torch.long, device=channel_logits.device)
                    sel = channel_logits.index_select(-1, hist_t)
                    pos = sel > 0
                    sel = torch.where(pos, sel / repetition_penalty, sel * repetition_penalty)
                    channel_logits.index_copy_(-1, hist_t, sel)
            channel_token = _sample_token(channel_logits, temperature, top_k, top_p, do_sample)
            codes[:, channel_index] = channel_token

            if channel_index + 1 < n_vq:
                embeds[:, channel_index + 1, :] = audio_embeddings[channel_index](channel_token).to(dtype)
                hidden = self._forward_prefix(embeds[:, : channel_index + 2, :])
                local_hidden = hidden[:, channel_index + 1, :]

        return should_continue, codes


__all__ = ["MossTTSLocalDepthTransformer"]
