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
from vllm.logger import init_logger

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
    _sample_token,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


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

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        query = query.view(batch_size, self.n_head, self.head_dim)
        key = key.view(batch_size, self.n_head, self.head_dim)
        value = value.view(batch_size, self.n_head, self.head_dim)

        position_ids = position_ids.to(dtype=torch.float32)
        freqs = torch.einsum("s,d->sd", position_ids, self.inv_freq.to(device=hidden_states.device))
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(hidden_states.dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(hidden_states.dtype)
        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin
        return query, key, value

    def forward_token(
        self,
        hidden_states: torch.Tensor,
        position: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Eager single-token path using the exact cached prefix length."""
        batch_size = hidden_states.shape[0]
        position_ids = torch.arange(
            position,
            position + 1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        query, key, value = self._project_qkv(hidden_states, position_ids)
        key_cache[:batch_size, :, position].copy_(key)
        value_cache[:batch_size, :, position].copy_(value)
        attn_output = F.scaled_dot_product_attention(
            query.unsqueeze(2),
            key_cache[:batch_size, :, : position + 1],
            value_cache[:batch_size, :, : position + 1],
            is_causal=False,
        )
        attn_output = attn_output.squeeze(2).reshape(batch_size, self.embed_dim)
        return self.c_proj(attn_output)

    def forward_token_static(
        self,
        hidden_states: torch.Tensor,
        position: torch.Tensor,
        cache_positions: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape token path that keeps ``position`` dynamic for compile."""
        batch_size = hidden_states.shape[0]
        query, key, value = self._project_qkv(hidden_states, position.reshape(1))

        cache_index = position.reshape(1)
        key_cache[:batch_size].index_copy_(2, cache_index, key.unsqueeze(2))
        value_cache[:batch_size].index_copy_(2, cache_index, value.unsqueeze(2))
        valid_prefix = cache_positions <= position
        attn_output = F.scaled_dot_product_attention(
            query.unsqueeze(2),
            key_cache[:batch_size],
            value_cache[:batch_size],
            attn_mask=valid_prefix.view(1, 1, 1, -1),
            is_causal=False,
        )
        attn_output = attn_output.squeeze(2).reshape(batch_size, self.embed_dim)
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

    def forward_token(
        self,
        hidden_states: torch.Tensor,
        position: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn.forward_token(
            self.ln_1(hidden_states),
            position,
            key_cache,
            value_cache,
        )
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states

    def forward_token_static(
        self,
        hidden_states: torch.Tensor,
        position: torch.Tensor,
        cache_positions: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn.forward_token_static(
            self.ln_1(hidden_states),
            position,
            cache_positions,
            key_cache,
            value_cache,
        )
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
        position. With ``use_static_local_kv_cache=True``, each step computes
        only that position and reuses the K/V written earlier in the frame;
        otherwise vLLM retains its growing-prefix compatibility path.
    """

    def __init__(
        self,
        gpt2_config,
        hidden_size: int | None = None,
        max_positions: int = 12,
        use_static_local_kv_cache: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size if hidden_size is not None else gpt2_config.n_embd)
        self.n_head = int(gpt2_config.n_head)
        self.head_dim = self.hidden_size // self.n_head
        self.max_positions = int(max_positions)
        self.use_static_local_kv_cache = bool(use_static_local_kv_cache)
        inner_size = int(gpt2_config.n_inner)
        eps = float(getattr(gpt2_config, "layer_norm_epsilon", 1e-5))
        rope_base = float(getattr(gpt2_config, "rope_base", 1_000_000.0))
        self.h = nn.ModuleList([_MossTTSLocalBlock(self.hidden_size, self.n_head, inner_size, rope_base, eps)])
        self.ln_f = nn.LayerNorm(self.hidden_size, eps=eps)
        self._compiled_forward_prefix = None
        self._compiled_forward_token = None
        self.register_buffer(
            "_cache_positions",
            torch.arange(self.max_positions, dtype=torch.long),
            persistent=False,
        )
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._kv_capacity = 0
        self._kv_frozen = False

    def _forward_prefix(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """Full-prefix reference path retained for parity tests."""
        hidden_states = seq_embeds
        for block in self.h:
            hidden_states = block(hidden_states)
        return self.ln_f(hidden_states)

    def _ensure_kv_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self._kv_capacity >= batch_size
            and self._kv_cache
            and self._kv_cache[0][0].device == device
            and self._kv_cache[0][0].dtype == dtype
        ):
            return
        if self._kv_frozen:
            raise RuntimeError(
                "MOSS-TTS local KV cache is frozen after CUDA graph setup "
                f"(capacity={self._kv_capacity}, requested={batch_size}, "
                f"device={device}, dtype={dtype})"
            )
        capacity = max(batch_size, self._kv_capacity, 1)
        shape = (capacity, self.n_head, self.max_positions, self.head_dim)
        self._kv_cache = [
            (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in self.h
        ]
        self._kv_capacity = capacity

    def prepare_kv_cache(self, max_batch_size: int) -> None:
        """Allocate the fixed-address workspace before CUDA graph capture."""
        if not self.use_static_local_kv_cache:
            return
        self._ensure_kv_cache(
            max_batch_size,
            self.ln_f.weight.device,
            self.ln_f.weight.dtype,
        )
        self._kv_frozen = True

    def _forward_token(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        x = hidden_states
        for layer_idx, block in enumerate(self.h):
            key_cache, value_cache = self._kv_cache[layer_idx]
            x = block.forward_token(x, position, key_cache, value_cache)
        return self.ln_f(x)

    def _forward_token_static(
        self,
        hidden_states: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states
        for layer_idx, block in enumerate(self.h):
            key_cache, value_cache = self._kv_cache[layer_idx]
            x = block.forward_token_static(
                x,
                position,
                self._cache_positions,
                key_cache,
                value_cache,
            )
        return self.ln_f(x)

    def setup_compile(self) -> None:
        if not self.use_static_local_kv_cache:
            if self._compiled_forward_prefix is not None:
                return
            if not current_omni_platform.supports_torch_inductor():
                self._compiled_forward_prefix = self._forward_prefix
                logger.warning_once("MOSS-TTS local depth torch.compile disabled on this platform")
                return
            self._compiled_forward_prefix = torch.compile(
                self._forward_prefix,
                dynamic=False,
                options={"epilogue_fusion": False},
            )
            logger.info("MOSS-TTS local depth prefix enabled with torch.compile")
            return
        if self._compiled_forward_token is not None:
            return
        if not current_omni_platform.supports_torch_inductor():
            self._compiled_forward_token = self._forward_token_static
            logger.warning_once("MOSS-TTS local depth torch.compile disabled on this platform")
            return
        self._compiled_forward_token = torch.compile(
            self._forward_token_static,
            dynamic=False,
            options={"epilogue_fusion": False},
        )
        logger.info("MOSS-TTS local depth incremental step enabled with torch.compile")

    def _run_prefix(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        forward_prefix = self._compiled_forward_prefix or self._forward_prefix
        return forward_prefix(seq_embeds)

    def _run_token(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        if not 0 <= position < self.max_positions:
            raise ValueError(f"local position {position} out of range [0, {self.max_positions})")
        self._ensure_kv_cache(hidden_states.shape[0], hidden_states.device, hidden_states.dtype)
        if self._compiled_forward_token is not None:
            return self._compiled_forward_token(hidden_states, self._cache_positions[position])
        return self._forward_token(hidden_states, position)

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
        generator: torch.Generator | None = None,
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

        if self.use_static_local_kv_cache:
            if n_vq > self.max_positions:
                raise ValueError(f"n_vq={n_vq} exceeds local KV capacity {self.max_positions}")
            local_hidden = self._run_token(backbone_last_hidden.to(dtype), 0)
            embeds = None
        else:
            embeds = backbone_last_hidden.new_zeros((batch_size, n_vq, self.hidden_size), dtype=dtype)
            embeds[:, 0, :] = backbone_last_hidden.to(dtype)
            local_hidden = self._run_prefix(embeds[:, :1, :])[:, 0, :]

        binary_logits = local_text_lm_head(local_hidden).float()
        # This is a binary continue/stop gate. The checkpoint expects sampling
        # here; greedy argmax is biased toward "continue" and may never stop.
        binary_choice = _sample_token(
            binary_logits,
            text_temperature,
            text_top_k,
            text_top_p,
            do_sample,
            generator=generator,
        )
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
            channel_token = _sample_token(
                channel_logits,
                temperature,
                top_k,
                top_p,
                do_sample,
                generator=generator,
            )
            codes[:, channel_index] = channel_token

            if channel_index + 1 < n_vq:
                next_embed = audio_embeddings[channel_index](channel_token).to(dtype)
                if self.use_static_local_kv_cache:
                    local_hidden = self._run_token(next_embed, channel_index + 1)
                else:
                    assert embeds is not None
                    embeds[:, channel_index + 1, :] = next_embed
                    local_hidden = self._run_prefix(embeds[:, : channel_index + 2, :])[:, channel_index + 1, :]

        return should_continue, codes


__all__ = ["MossTTSLocalDepthTransformer"]
