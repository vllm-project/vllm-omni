# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Per-frame depth transformer for MossTTSLocalModel (MOSS-TTS-Local-Transformer-v1.5).

A 1-layer GPT2-style block that decodes the ``n_vq`` audio codebook codes for
one audio frame, run inside the talker's ``talker_mtp`` independent of
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
        self.register_buffer("_rope_cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin_cache", torch.empty(0), persistent=False)

    def prepare_rope_cache(
        self,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Materialize frame-local RoPE values once per device/dtype.

        Local Depth positions always restart at zero and never exceed the
        number of RVQ codebooks (12 for MOSS-TTS Local v1.5).  Building
        arange/einsum/cos/sin inside every one of the 12 depth steps creates
        many tiny kernels and also bloats the enclosing talker CUDA graph.
        Keep one non-persistent cache and slice it for prefix execution.
        """
        if max_seq_len <= 0:
            return
        cache = self._rope_cos_cache
        if cache.numel() > 0 and cache.device == device and cache.dtype == dtype and int(cache.shape[1]) >= max_seq_len:
            return

        position_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.einsum("s,d->sd", position_ids, inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype)
        self._rope_cos_cache = cos.view(1, max_seq_len, 1, self.head_dim)
        self._rope_sin_cache = sin.view(1, max_seq_len, 1, self.head_dim)

    def _rope_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.prepare_rope_cache(seq_len, device, dtype)
        return (
            self._rope_cos_cache[:, :seq_len],
            self._rope_sin_cache[:, :seq_len],
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        even = x[..., ::2]
        odd = x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).reshape_as(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position: int = 0,
    ) -> torch.Tensor:
        """Run a causal prefix, or one new token using frame-local K/V."""
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim)

        cos, sin = self._rope_cos_sin(position + seq_len, hidden_states.device, hidden_states.dtype)
        cos, sin = cos[:, position:], sin[:, position:]
        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if kv_cache is not None:
            assert seq_len == 1, "Cached Local Depth execution consumes one token at a time"
            k_cache, v_cache = kv_cache
            k_cache[:, :, position : position + 1].copy_(key)
            v_cache[:, :, position : position + 1].copy_(value)
            key = k_cache[:, :, : position + 1]
            value = v_cache[:, :, : position + 1]
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=kv_cache is None)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position: int = 0,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), kv_cache, position)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class MossTTSLocalDepthTransformer(nn.Module):
    """Per-frame depth transformer for MOSS-TTS-Local-Transformer-v1.5.

    Per frame:
      - position 0's input is the backbone's last hidden state; its output
        feeds BOTH the binary continue/stop head (``local_text_lm_head``)
        and codebook-0's head (``audio_lm_heads[0]``) simultaneously.
      - codebooks 1..n_vq-1 are sampled sequentially. When lookup tables are
        prepared, their first-layer normalized QKV (including RoPE) are
        gathered by token ID, with attention using a frame-local KV cache.
        This is valid because there is exactly one block: its input embedding
        has no context dependence. The original prefix path remains available
        as a reference and for running without prepared lookup tables.
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
        self._compiled_forward_prefix = None
        self._compiled_forward_lookup = None
        self.register_buffer("_qkv_lookup", torch.empty(0), persistent=False)
        self._lookup_n_vq = 0
        self._fused_attention = False
        self._fused_linear = False
        self._auto_fused_linear = False
        self._fused_sampling = False

    @torch.no_grad()
    def prepare_qkv_lookup(self, audio_embeddings: nn.ModuleList, n_vq: int) -> None:
        """Build derived inference tables after all weights have been loaded.

        Rebuild after changing weights, before compilation/graph capture.
        Do not replace these buffers while captured graphs still reference them.
        """
        if len(self.h) != 1:
            raise ValueError("MOSS local QKV lookup requires exactly one transformer block")
        if n_vq < 1 or len(audio_embeddings) < n_vq:
            raise ValueError("n_vq must be positive and covered by audio_embeddings")
        block = self.h[0]
        attn = block.attn
        device, dtype = block.ln_1.weight.device, block.ln_1.weight.dtype
        attn.prepare_rope_cache(n_vq, device, dtype)
        vocab_size = audio_embeddings[0].num_embeddings
        table = torch.empty((n_vq - 1, vocab_size, 3 * self.hidden_size), device=device, dtype=dtype)
        for channel in range(n_vq - 1):
            if audio_embeddings[channel].num_embeddings != vocab_size:
                raise ValueError("MOSS local QKV lookup requires equal codebook vocabulary sizes")
            # Bound temporary activations during model loading.
            for start in range(0, vocab_size, 256):
                embeds = audio_embeddings[channel].weight[start : start + 256].to(dtype)
                qkv = attn.c_attn(block.ln_1(embeds))
                q, k, v = qkv.split(self.hidden_size, dim=-1)
                q = q.reshape(-1, attn.n_head, attn.head_dim)
                k = k.reshape_as(q)
                cos = attn._rope_cos_cache[0, channel + 1]
                sin = attn._rope_sin_cache[0, channel + 1]
                q = q * cos + attn._rotate_half(q) * sin
                k = k * cos + attn._rotate_half(k) * sin
                table[channel, start : start + 256].copy_(torch.cat((q.flatten(1), k.flatten(1), v), dim=-1))
        self._qkv_lookup = table
        self._lookup_n_vq = n_vq
        cuda_kernels = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
        self._fused_attention = cuda_kernels and attn.head_dim <= 128 and n_vq <= 16
        self._fused_linear = False
        self._auto_fused_linear = (
            cuda_kernels
            and dtype == torch.bfloat16
            and self.hidden_size == 2560
            and block.mlp.fc_in.out_features == 9728
            and torch.cuda.get_device_capability(device) == (9, 0)
        )
        self._fused_sampling = cuda_kernels
        logger.info("MOSS-TTS local QKV lookup prepared: %.1f MiB", table.numel() * table.element_size() / 2**20)
        logger.info("MOSS local fused kernels: attention=%s sampling=%s", self._fused_attention, self._fused_sampling)

    def _forward_lookup(
        self,
        hidden: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        block = self.h[0]
        batch = hidden.shape[0]
        auto_proj = self._auto_fused_linear and batch <= 2
        auto_down = self._auto_fused_linear and batch == 1
        if self._fused_linear or auto_proj or auto_down:
            from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_linear

            if self._fused_linear or auto_proj:
                hidden = fused_linear(attn_output, block.attn.c_proj.weight, block.attn.c_proj.bias, hidden)
            else:
                hidden = hidden + block.attn.c_proj(attn_output)
            if self._fused_linear:
                intermediate = fused_linear(
                    block.ln_2(hidden), block.mlp.fc_in.weight, block.mlp.fc_in.bias, activation=True
                )
            else:
                intermediate = block.mlp.act(block.mlp.fc_in(block.ln_2(hidden)))
            if self._fused_linear or auto_down:
                hidden = fused_linear(intermediate, block.mlp.fc_out.weight, block.mlp.fc_out.bias, hidden)
            else:
                hidden = hidden + block.mlp.fc_out(intermediate)
            return self.ln_f(hidden)
        hidden = hidden + block.attn.c_proj(attn_output)
        hidden = hidden + block.mlp(block.ln_2(hidden))
        return self.ln_f(hidden)

    def _run_lookup_step(
        self,
        hidden: torch.Tensor,
        qkv: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        attn = self.h[0].attn
        if self._fused_attention:
            from vllm_omni.model_executor.models.moss_tts.local_kernels import lookup_attention

            out, _ = lookup_attention(qkv, key, value, position)
            forward = self._compiled_forward_lookup or self._forward_lookup
            return forward(hidden, out)
        q, k, v = (t.reshape(-1, attn.n_head, 1, attn.head_dim) for t in qkv.split(self.hidden_size, dim=-1))
        key[:, :, position : position + 1].copy_(k)
        value[:, :, position : position + 1].copy_(v)
        # Slice to the valid prefix so SDPA can use an unmasked backend.
        # is_causal=True would incorrectly align the length-1 query at zero.
        out = F.scaled_dot_product_attention(
            q, key[:, :, : position + 1], value[:, :, : position + 1], is_causal=False
        ).reshape_as(hidden)
        # Keep variable-length attention outside this compiled function (the
        # enclosing CUDA graph still captures it). The remaining projections /
        # MLP have identical shapes at every depth, without 12 specializations.
        forward = self._compiled_forward_lookup or self._forward_lookup
        return forward(hidden, out)

    def _forward_prefix(
        self,
        seq_embeds: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position: int = 0,
    ) -> torch.Tensor:
        return self.ln_f(self.h[0](seq_embeds, kv_cache, position))

    def setup_compile(self) -> None:
        if self._lookup_n_vq > 0:
            if self._compiled_forward_lookup is None and current_omni_platform.supports_torch_inductor():
                self._compiled_forward_lookup = torch.compile(
                    self._forward_lookup, dynamic=False, options={"epilogue_fusion": False}
                )
                logger.info("MOSS-TTS local QKV lookup enabled with torch.compile")
            return
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

    def _run_prefix(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        forward_prefix = self._compiled_forward_prefix or self._forward_prefix
        return forward_prefix(seq_embeds)

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
        fused_sampling = (
            self._lookup_n_vq > 0
            and self._fused_sampling
            and do_sample
            and temperature > 0
            and 0 < top_k <= min(32, audio_lm_heads[0].weight.shape[0])
            and 0 < top_p <= 1
            and repetition_penalty == 1.0
        )
        if fused_sampling:
            from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_sample

            # Philox state remains owned by PyTorch, including graph replay.
            # Uniforms are independent across requests and codebooks.
            uniforms = torch.rand((batch_size, n_vq), device=backbone_last_hidden.device, generator=generator)

        # Populate the complete [0, n_vq) table before torch.compile traces
        # _forward_prefix or the runner captures talker_mtp.  The compiled /
        # captured body then contains only fixed-address cache slices rather
        # than cache construction or buffer replacement.
        for block in self.h:
            block.attn.prepare_rope_cache(n_vq, backbone_last_hidden.device, dtype)

        use_lookup = self._lookup_n_vq > 0
        if use_lookup:
            if self._lookup_n_vq != n_vq:
                raise ValueError("n_vq differs from the prepared MOSS local QKV lookup")
            attn = self.h[0].attn
            # Invocation-local storage is private to each captured graph and
            # eager call. Each slot is overwritten before the valid-prefix
            # attention reads it, so no full-buffer clearing is needed.
            key = backbone_last_hidden.new_empty((batch_size, attn.n_head, n_vq, attn.head_dim), dtype=dtype)
            value = torch.empty_like(key)
            first_hidden = backbone_last_hidden.to(dtype)
            # Position-zero RoPE is identity; only this input depends on the
            # backbone and therefore needs a runtime LN/QKV projection.
            qkv = attn.c_attn(self.h[0].ln_1(first_hidden))
            local_hidden = self._run_lookup_step(first_hidden, qkv, key, value, 0)
        else:
            embeds = backbone_last_hidden.new_zeros((batch_size, n_vq, self.hidden_size), dtype=dtype)
            embeds[:, 0, :] = backbone_last_hidden.to(dtype)
            hidden = self._run_prefix(embeds[:, :1, :])
            local_hidden = hidden[:, 0, :]

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
            if fused_sampling:
                channel_token = fused_sample(
                    channel_logits, uniforms[:, channel_index], top_k=top_k, temperature=temperature, top_p=top_p
                )
            else:
                channel_token = _sample_token(channel_logits, temperature, top_k, top_p, do_sample, generator=generator)
            codes[:, channel_index] = channel_token

            if channel_index + 1 < n_vq:
                if use_lookup and self._fused_attention:
                    from vllm_omni.model_executor.models.moss_tts.local_kernels import lookup_attention

                    out, next_embed = lookup_attention(
                        self._qkv_lookup[channel_index],
                        key,
                        value,
                        channel_index + 1,
                        tokens=channel_token,
                        embedding=audio_embeddings[channel_index].weight,
                    )
                    forward = self._compiled_forward_lookup or self._forward_lookup
                    local_hidden = forward(next_embed, out)
                    continue
                next_embed = audio_embeddings[channel_index](channel_token).to(dtype)
                if use_lookup:
                    qkv = F.embedding(channel_token, self._qkv_lookup[channel_index])
                    local_hidden = self._run_lookup_step(next_embed, qkv, key, value, channel_index + 1)
                else:
                    embeds[:, channel_index + 1, :] = next_embed
                    hidden = self._run_prefix(embeds[:, : channel_index + 2, :])
                    local_hidden = hidden[:, channel_index + 1, :]

        return should_continue, codes


__all__ = ["MossTTSLocalDepthTransformer"]
