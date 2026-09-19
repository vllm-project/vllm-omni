# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a causal prefix, or one new token using frame-local K/V.

        When ``attn_mask`` is provided (NPUGraph path), the full ``kv_cache``
        tensors are passed to SDPA as-is with the mask zeroing positions
        ``[position+1, n_vq)`` -- fixed shapes for graph capture. Without
        ``attn_mask`` (eager path), a dynamic slice ``[:position+1]`` is
        used instead.
        """
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
            if attn_mask is not None:
                key = k_cache
                value = v_cache
            else:
                key = k_cache[:, :, : position + 1]
                value = v_cache[:, :, : position + 1]
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=kv_cache is None and attn_mask is None
        )
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
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), kv_cache, position, attn_mask)
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
        position. K/V for earlier positions are reused within the frame, so
        each position's projections and MLP run once. A new frame overwrites
        every cache position before reading it, including CUDA Graph replay.
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
        # NPU whole-loop NPUGraph infra: the entire n_vq-iteration autoregressive
        # loop (KV-cache incremental decode + Gumbel-max sampling with
        # externally-generated noise) is captured as ONE NPUGraph per
        # power-of-2 batch bucket and replayed once per frame -- amortising the
        # NPUGraph replay overhead over the whole loop instead of paying it
        # n_vq times.  Populated lazily: ``setup_compile`` sets up the infra,
        # ``generate_frame`` captures per bucket on first use.  None / empty
        # on GPU and until capture.
        self._n_vq: int | None = None
        self._npu_graph_enabled: bool = False
        self._npu_buckets: list[int] = []
        self._npu_whole_graphs: dict[int, tuple] = {}
        self._npu_graph_pool = None
        self._npu_captured_modules: tuple | None = None
        self._npu_captured_params: tuple | None = None

    def _forward_prefix(
        self,
        seq_embeds: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.ln_f(self.h[0](seq_embeds, kv_cache, position, attn_mask))

    def setup_compile(self, n_vq: int, max_num_seqs: int = 64) -> None:
        """Compile the prefix forward for fast per-frame decoding.

        On inductor-capable platforms (GPU) this applies ``torch.compile``
        unchanged. On NPU, ``torch.compile`` is unavailable
        (``supports_torch_inductor() is False``) and per-call ``NPUGraph``
        capture/replay does not pay off for this 1-layer block (replay overhead
        ~0.45ms/call exceeds the dispatch it saves). The win that does work on
        NPU is to capture the ENTIRE n_vq-iteration autoregressive loop of
        ``generate_frame`` as ONE NPUGraph (KV-cache incremental decode so each
        step computes only 1 new position, + Gumbel-max sampling with
        externally-generated noise so the graph has no RNG), replayed once per
        frame. That amortises the replay overhead over the whole loop and cuts
        compute ~6x vs re-prefill.

        Graph capture itself is deferred to the first ``generate_frame`` call
        (per batch bucket) because it needs the
        ``audio_lm_heads``/``audio_embeddings``/``local_text_lm_head`` args.
        """
        if self._compiled_forward_prefix is not None:
            return
        self._n_vq = int(n_vq)

        if not current_omni_platform.supports_torch_inductor():
            self._compiled_forward_prefix = self._forward_prefix
            if current_omni_platform.is_npu():
                mns = max(1, int(max_num_seqs))
                buckets = [1 << i for i in range(mns.bit_length()) if (1 << i) <= mns]
                if mns not in buckets:
                    buckets.append(mns)
                self._npu_buckets = sorted(set(buckets))
                self._npu_graph_enabled = True
                logger.info(
                    "MOSS-TTS local depth NPU whole-loop graph enabled "
                    "(buckets=%s, n_vq=%d); lazy capture on first generate_frame",
                    self._npu_buckets,
                    n_vq,
                )
            else:
                logger.warning_once("MOSS-TTS local depth torch.compile disabled on this platform")
            return

        self._compiled_forward_prefix = torch.compile(
            self._forward_prefix,
            # Share batch/position shapes instead of exhausting Dynamo's
            # recompilation budget while warming request-size graph buckets.
            dynamic=True,
            options={"epilogue_fusion": False},
        )
        logger.info("MOSS-TTS local depth frame-local KV execution enabled with torch.compile")

    def _run_prefix(
        self,
        seq_embeds: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        position: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_prefix = self._compiled_forward_prefix or self._forward_prefix
        return forward_prefix(seq_embeds, kv_cache, position, attn_mask)

    # ------------------------------------------------------------------
    # NPU whole-loop NPUGraph path (#02)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_topk_topp_mask(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Top-k/top-p masking as pure tensor ops (graph-capturable).

        Mirrors ``_sample_token``'s masking minus the final ``multinomial`` --
        the whole-loop graph samples via Gumbel-max instead (a captured
        NPUGraph cannot run ``multinomial`` with a per-request generator on
        NPU).
        """
        if top_k and 0 < top_k < logits.shape[-1]:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits = torch.where(
                logits < top_vals[..., -1:],
                torch.full_like(logits, float("-inf")),
                logits,
            )
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            drop = cum > top_p
            drop[..., 1:] = drop[..., :-1].clone()
            drop[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
        return logits

    def _fwd_incremental_graph(
        self,
        x_c: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        c: int,
    ) -> torch.Tensor:
        """One-position KV-cache forward for position ``c`` (graph-captured).

        Reuses the existing ``forward(kv_cache=...)`` path but passes an
        ``attn_mask`` so the full ``cache_k`` (fixed shape) is used instead
        of a dynamic ``[:c+1]`` slice -- required for NPUGraph capture.
        """
        n_vq = cache_k.shape[2]
        mask = torch.ones(1, n_vq, dtype=torch.bool, device=x_c.device)
        mask[:, c + 1 :] = False
        return self.ln_f(self.h[0](x_c, (cache_k, cache_v), c, mask))

    def _whole_loop(
        self,
        audio_lm_heads: nn.ModuleList,
        audio_embeddings: nn.ModuleList,
        local_text_lm_head: nn.Module,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        text_temperature: float,
        text_top_k: int,
        text_top_p: float,
        embeds_buf: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        gumb_codes: torch.Tensor,
        gumb_bin: torch.Tensor,
        codes_buf: torch.Tensor,
        cont_buf: torch.Tensor,
        n_vq: int,
    ) -> None:
        """The full n_vq-iteration autoregressive loop (captured as ONE graph).

        ``embeds_buf`` ``(B, n_vq, H)``: position 0 prefilled (backbone hidden);
        positions 1..n_vq-1 filled in-graph from sampled-code embeddings.
        ``gumb_codes`` ``(B, n_vq, V)`` / ``gumb_bin`` ``(B, 2)``: Gumbel noise
        generated eagerly per frame (with the request generator) -- the graph
        has NO RNG. ``do_sample``/``temperature``/``top_k``/``top_p``/``text_*``
        are captured Python constants (the graph is only used when the call's
        params match the captured ones -- see ``generate_frame``).
        """
        h0 = self._fwd_incremental_graph(embeds_buf[:, 0:1], cache_k, cache_v, 0)
        lh = h0[:, 0]
        bl = local_text_lm_head(lh).float()
        if do_sample:
            bl = self._apply_topk_topp_mask(bl / text_temperature, text_top_k, text_top_p)
            bin_tok = (bl + gumb_bin).argmax(-1)
        else:
            bin_tok = bl.argmax(-1)
        cont_buf.copy_(bin_tok)

        cl = audio_lm_heads[0](lh).float()
        if do_sample:
            cl = self._apply_topk_topp_mask(cl / temperature, top_k, top_p)
            tok = (cl + gumb_codes[:, 0]).argmax(-1)
        else:
            tok = cl.argmax(-1)
        codes_buf[:, 0].copy_(tok)
        embeds_buf[:, 1].copy_(audio_embeddings[0](tok))

        for c in range(1, n_vq):
            hc = self._fwd_incremental_graph(embeds_buf[:, c : c + 1], cache_k, cache_v, c)
            lh = hc[:, 0]
            cl = audio_lm_heads[c](lh).float()
            if do_sample:
                cl = self._apply_topk_topp_mask(cl / temperature, top_k, top_p)
                tok = (cl + gumb_codes[:, c]).argmax(-1)
            else:
                tok = cl.argmax(-1)
            codes_buf[:, c].copy_(tok)
            if c + 1 < n_vq:
                embeds_buf[:, c + 1].copy_(audio_embeddings[c](tok))

    def _make_gumbel_noise(
        self, batch_size: int, n_vq: int, vocab: int, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gumbel(0,1) noise ``(B, n_vq, V)`` and ``(B, 2)`` seeded by ``generator``.

        Generated eagerly per frame (outside the captured graph) so the graph
        has no RNG and per-request reproducibility is preserved (same request
        seed -> same noise -> same Gumbel-max -> same codes).
        """
        device = next(self.parameters()).device
        dtype = self.ln_f.weight.dtype
        if not hasattr(self, "_gumbel_u") or self._gumbel_u is None or self._gumbel_u.shape[0] < batch_size:
            self._gumbel_u = torch.empty(max(batch_size, 64), n_vq, vocab, device=device, dtype=torch.float32)
            self._gumbel_ub = torch.empty(max(batch_size, 64), 2, device=device, dtype=torch.float32)
        u = self._gumbel_u[:batch_size].uniform_(1e-6, 1 - 1e-6, generator=generator)
        gc = -torch.log(-torch.log(u))
        ub = self._gumbel_ub[:batch_size].uniform_(1e-6, 1 - 1e-6, generator=generator)
        gb = -torch.log(-torch.log(ub))
        return gc.to(dtype), gb.to(dtype)

    def _capture_whole_graph(
        self,
        bucket: int,
        audio_lm_heads: nn.ModuleList,
        audio_embeddings: nn.ModuleList,
        local_text_lm_head: nn.Module,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        text_temperature: float,
        text_top_k: int,
        text_top_p: float,
    ) -> tuple | None:
        """Allocate static buffers + capture the whole-loop NPUGraph for ``bucket``."""
        device = next(self.parameters()).device
        dtype = self.ln_f.weight.dtype
        attn = self.h[0].attn
        n_vq = self._n_vq
        vocab = audio_lm_heads[0].out_features
        embeds_buf = torch.zeros(bucket, n_vq, self.hidden_size, device=device, dtype=dtype)
        cache_k = torch.zeros(bucket, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)
        cache_v = torch.zeros(bucket, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)
        gumb_codes = torch.zeros(bucket, n_vq, vocab, device=device, dtype=dtype)
        gumb_bin = torch.zeros(bucket, 2, device=device, dtype=dtype)
        codes_buf = torch.zeros(bucket, n_vq, device=device, dtype=torch.long)
        cont_buf = torch.zeros(bucket, device=device, dtype=torch.long)
        with torch.no_grad():
            for _ in range(3):
                self._whole_loop(
                    audio_lm_heads,
                    audio_embeddings,
                    local_text_lm_head,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    text_temperature,
                    text_top_k,
                    text_top_p,
                    embeds_buf,
                    cache_k,
                    cache_v,
                    gumb_codes,
                    gumb_bin,
                    codes_buf,
                    cont_buf,
                    n_vq,
                )
        torch.npu.synchronize()
        try:
            if self._npu_graph_pool is None:
                self._npu_graph_pool = torch.npu.graph_pool_handle()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=self._npu_graph_pool):
                self._whole_loop(
                    audio_lm_heads,
                    audio_embeddings,
                    local_text_lm_head,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    text_temperature,
                    text_top_k,
                    text_top_p,
                    embeds_buf,
                    cache_k,
                    cache_v,
                    gumb_codes,
                    gumb_bin,
                    codes_buf,
                    cont_buf,
                    n_vq,
                )
        except Exception as exc:
            logger.warning_once(
                "MOSS-TTS local depth whole-loop graph capture (bucket=%d) failed: %s; eager fallback",
                bucket,
                exc,
            )
            return None
        self._npu_whole_graphs[bucket] = (
            graph,
            embeds_buf,
            cache_k,
            cache_v,
            gumb_codes,
            gumb_bin,
            codes_buf,
            cont_buf,
        )
        self._npu_captured_modules = (
            id(audio_lm_heads),
            id(audio_embeddings),
            id(local_text_lm_head),
        )
        self._npu_captured_params = (
            do_sample,
            temperature,
            top_k,
            top_p,
            text_temperature,
            text_top_k,
            text_top_p,
        )
        logger.info(
            "MOSS-TTS local depth captured whole-loop graph (bucket=%d, do_sample=%s)",
            bucket,
            do_sample,
        )
        return self._npu_whole_graphs[bucket]

    @torch.no_grad()
    def _generate_frame_graph(
        self,
        backbone_last_hidden: torch.Tensor,
        audio_lm_heads: nn.ModuleList,
        audio_embeddings: nn.ModuleList,
        local_text_lm_head: nn.Module,
        *,
        n_vq: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        text_temperature: float,
        text_top_k: int,
        text_top_p: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Whole-loop graph path. Returns ``(should_continue, codes)`` or None on miss."""
        B = int(backbone_last_hidden.shape[0])
        bucket = next((b for b in self._npu_buckets if B <= b), None)
        if bucket is None:
            return None
        params = (do_sample, temperature, top_k, top_p, text_temperature, text_top_k, text_top_p)
        modules = (id(audio_lm_heads), id(audio_embeddings), id(local_text_lm_head))
        entry = self._npu_whole_graphs.get(bucket)
        if entry is None or self._npu_captured_modules != modules or self._npu_captured_params != params:
            entry = self._capture_whole_graph(
                bucket,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample,
                temperature,
                top_k,
                top_p,
                text_temperature,
                text_top_k,
                text_top_p,
            )
            if entry is None:
                return None
        graph, embeds_buf, cache_k, cache_v, gumb_codes, gumb_bin, codes_buf, cont_buf = entry
        dtype = self.ln_f.weight.dtype
        embeds_buf.zero_()
        embeds_buf[:B, 0, :].copy_(backbone_last_hidden.to(dtype))
        cache_k.zero_()
        cache_v.zero_()
        if do_sample:
            vocab = audio_lm_heads[0].out_features
            gc, gb = self._make_gumbel_noise(B, self._n_vq, vocab, generator)
            gumb_codes[:B].copy_(gc)
            gumb_bin[:B].copy_(gb)
        graph.replay()
        should_continue = cont_buf[:B].eq(0)
        codes = codes_buf[:B].clone()
        return should_continue, codes

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

        On NPU, dispatches to the whole-loop NPUGraph path
        (``_generate_frame_graph``) when the call matches the captured config
        (``do_sample``/``temperature``/``top_k``/``top_p``/``text_*``/``n_vq``,
        ``repetition_penalty==1.0``, no history, not inside an outer capture)
        -- the fast path. Otherwise falls back to the eager KV-cache path below.

        The graph path's sampling is Gumbel-max: distributionally identical to
        ``multinomial`` (both sample from ``softmax(logits)``) and itself
        deterministic given the request generator (it seeds the noise), but NOT
        bit-identical to the old multinomial output -- ``multinomial`` with a
        per-request generator cannot be captured in an NPUGraph on this torch_npu.
        """
        batch_size = backbone_last_hidden.shape[0]
        dtype = self.ln_f.weight.dtype

        # Populate the complete [0, n_vq) table before torch.compile traces
        # _forward_prefix or the runner captures talker_mtp.  The compiled /
        # captured body then contains only fixed-address cache slices rather
        # than cache construction or buffer replacement.
        for block in self.h:
            block.attn.prepare_rope_cache(n_vq, backbone_last_hidden.device, dtype)

        # NPU whole-loop NPUGraph fast path: dispatch to the captured graph
        # when the call matches the captured config (no repetition penalty,
        # no history, not inside an outer graph capture). Falls back to the
        # eager KV-cache path below on any miss.
        if (
            self._npu_graph_enabled
            and n_vq == self._n_vq
            and repetition_penalty == 1.0
            and history_per_codebook is None
            and backbone_last_hidden.device.type == "npu"
            and not torch.npu.is_current_stream_capturing()
        ):
            try:
                out = self._generate_frame_graph(
                    backbone_last_hidden,
                    audio_lm_heads,
                    audio_embeddings,
                    local_text_lm_head,
                    n_vq=n_vq,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    text_temperature=text_temperature,
                    text_top_k=text_top_k,
                    text_top_p=text_top_p,
                    generator=generator,
                )
                if out is not None:
                    return out
            except Exception as exc:
                logger.warning_once(
                    "MOSS-TTS local depth NPU graph path failed: %s; eager fallback",
                    exc,
                )

        attn = self.h[0].attn
        cache_shape = (batch_size, attn.n_head, n_vq, attn.head_dim)
        kv_cache = (
            backbone_last_hidden.new_empty(cache_shape, dtype=dtype),
            backbone_last_hidden.new_empty(cache_shape, dtype=dtype),
        )
        hidden = self._run_prefix(backbone_last_hidden[:, None, :].to(dtype), kv_cache, 0)
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
                embeds = audio_embeddings[channel_index](channel_token).to(dtype)[:, None, :]
                hidden = self._run_prefix(embeds, kv_cache, channel_index + 1)
                local_hidden = hidden[:, 0, :]

        return should_continue, codes


__all__ = ["MossTTSLocalDepthTransformer"]
