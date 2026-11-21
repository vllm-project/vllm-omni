# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 Talker model and MRv2 ModelState for Google Magenta RealTime 2."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.magenta_rt2.conditioning import (
    DEFAULT_CFG_SCALES,
    ENCODER_NUM_CHANNELS,
    MagentaRT2ConditioningEncoder,
    build_conditioning_tokens,
)
from vllm_omni.model_executor.models.magenta_rt2.spectrostream import _unflatten_nested
from vllm_omni.model_executor.models.magenta_rt2.transformer import (
    DepthTransformerBlock,
    TemporalTransformerBlock,
    load_flax_attention_weights,
    load_flax_ffn_weights,
)
from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot, pack_output_snapshot
from vllm_omni.transformers_utils.configs.magenta_rt2 import (
    MagentaRT2Config,
    magenta_rt2_allow_patterns,
    resolve_magenta_rt2_checkpoint,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.warmup import is_warmup_request

logger = init_logger(__name__)

_LARGE_NEG = -1e9


def _sample_gumbel(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=torch.float32, generator=generator).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


class MagentaRT2Depthformer(nn.Module):
    """Autoregressive 12-codebook Depthformer with CUDA-graph-friendly step decoding."""

    def __init__(self, config: MagentaRT2Config) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = int(config.num_codebooks)
        self.codebook_size = int(config.codebook_size)
        self.num_extra_tokens = int(config.num_extra_tokens)
        self.vocab_size = int(config.vocab_size)
        self.temporal_hidden_size = int(config.temporal_hidden_size)
        self.depth_hidden_size = int(config.depth_hidden_size)
        self.logits_soft_cap = float(config.logits_soft_cap)
        self._embed_scale = math.sqrt(float(self.temporal_hidden_size))

        self.decoder_embedding = nn.Embedding(self.vocab_size, self.temporal_hidden_size)
        if self.temporal_hidden_size != self.depth_hidden_size:
            self.depth_input_adapter: nn.Module = nn.Linear(
                self.temporal_hidden_size, self.depth_hidden_size, bias=False
            )
        else:
            self.depth_input_adapter = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                DepthTransformerBlock(
                    hidden_size=self.depth_hidden_size,
                    ffn_dim=int(config.depth_ffn_dim),
                    num_heads=int(config.depth_num_heads),
                    head_dim=int(config.depth_head_dim),
                    max_past_horizon=int(config.depth_past_horizon),
                    norm_eps=float(config.norm_eps),
                )
                for _ in range(int(config.depth_num_layers))
            ]
        )
        self.final_ln = nn.LayerNorm(self.depth_hidden_size, eps=float(config.norm_eps))
        self.to_logits = nn.Linear(self.depth_hidden_size, self.vocab_size, bias=True)

    def embed_codes(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder_embedding(tokens.to(dtype=torch.long)) * self._embed_scale

    def forward_teacher_forced(
        self, temporal_hidden: torch.Tensor, prev_codes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Parallel depth forward ``[B, Q, vocab_size]`` given temporal state + optional prior codes ``[B, Q-1]``."""
        temporal_seq = temporal_hidden.unsqueeze(1)
        if prev_codes is None or self.num_codebooks == 1:
            seq = temporal_seq
        else:
            seq = torch.cat([temporal_seq, self.embed_codes(prev_codes[:, : self.num_codebooks - 1])], dim=1)
        h = self.depth_input_adapter(seq)
        for blk in self.blocks:
            h = blk.forward_sequence(h)
        h = F.layer_norm(
            h.float(),
            (self.depth_hidden_size,),
            self.final_ln.weight.float(),
            self.final_ln.bias.float(),
            float(self.config.norm_eps),
        ).to(dtype=h.dtype)
        logits = self.to_logits(h)
        if self.logits_soft_cap > 0:
            logits = torch.tanh(logits / self.logits_soft_cap) * self.logits_soft_cap
        return logits

    @torch.inference_mode()
    def sample_codes(
        self,
        temporal_hidden: torch.Tensor,
        *,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        gumbel_noise: torch.Tensor,
        out_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Run all 12 depth steps using pre-allocated static buffers (CUDA-graph capturable).

        Args:
            temporal_hidden: ``[B, temporal_hidden_size]``
            k_caches: ``[L, B, Q, H, D]`` scratch cache
            v_caches: ``[L, B, Q, H, D]`` scratch cache
            temperature: ``[B]`` float32 sampling temperatures (``0`` for greedy)
            top_k: ``[B]`` int64 top-k values (``<= 0`` clamps to full ``codebook_size``)
            gumbel_noise: ``[B, Q, codebook_size]`` float32 Gumbel(0, 1) noise
            out_codes: ``[B, Q]`` int64 destination tensor
        """
        cur = temporal_hidden
        k_clamped = torch.where(top_k > 0, top_k, self.codebook_size).clamp(1, self.codebook_size)
        k_idx = (k_clamped - 1).unsqueeze(-1)  # [B, 1]
        temp_col = temperature.clamp(min=0.0).unsqueeze(-1)  # [B, 1]

        for q in range(self.num_codebooks):
            h = self.depth_input_adapter(cur)
            for layer_idx, blk in enumerate(self.blocks):
                h = blk.forward_step(h, k_caches[layer_idx], v_caches[layer_idx], q)
            h = F.layer_norm(
                h.float(),
                (self.depth_hidden_size,),
                self.final_ln.weight.float(),
                self.final_ln.bias.float(),
                float(self.config.norm_eps),
            ).to(dtype=h.dtype)
            logits = self.to_logits(h).float()
            if self.logits_soft_cap > 0:
                logits = torch.tanh(logits / self.logits_soft_cap) * self.logits_soft_cap

            q_start = self.num_extra_tokens + q * self.codebook_size
            q_end = q_start + self.codebook_size
            valid_logits = logits[:, q_start:q_end]  # [B, 1024]

            # Dynamic per-request top-k filtering inside static CUDA graph.
            sorted_vals, _ = torch.sort(valid_logits, dim=-1, descending=True)
            kth = sorted_vals.gather(-1, k_idx)
            filtered = torch.where(
                valid_logits >= kth,
                valid_logits,
                torch.full_like(valid_logits, _LARGE_NEG),
            )
            noisy = filtered + gumbel_noise[:, q, :] * temp_col
            sampled_local = torch.argmax(noisy, dim=-1)
            sampled_global = sampled_local + q_start
            out_codes[:, q].copy_(sampled_global)

            if q + 1 < self.num_codebooks:
                cur = self.embed_codes(sampled_global)
        return out_codes


class DepthformerGraphRunner:
    """Bucketed CUDA graph runner for ``MagentaRT2Depthformer.sample_codes``."""

    def __init__(
        self,
        depthformer: MagentaRT2Depthformer,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_batch_size: int,
    ) -> None:
        self.depthformer = depthformer
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max(1, int(max_batch_size))
        self.num_layers = len(depthformer.blocks)
        self.num_codebooks = depthformer.num_codebooks
        self.num_heads = int(depthformer.config.depth_num_heads)
        self.head_dim = int(depthformer.config.depth_head_dim)
        self.codebook_size = depthformer.codebook_size
        self.temporal_hidden_size = depthformer.temporal_hidden_size

        # Bucket sizes (powers of 2 up to 16, then multiples of 8)
        buckets: list[int] = [1, 2, 4, 8]
        b = 16
        while b < self.max_batch_size:
            buckets.append(b)
            b += 8
        if self.max_batch_size not in buckets:
            buckets.append(self.max_batch_size)
        self.buckets = sorted({bs for bs in buckets if bs <= self.max_batch_size})
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, dict[str, torch.Tensor]]] = {}

    def _alloc_buffers(self, bsz: int) -> dict[str, torch.Tensor]:
        with torch.inference_mode(False):
            return {
                "temporal": torch.zeros((bsz, self.temporal_hidden_size), dtype=self.dtype, device=self.device),
                "k_caches": torch.zeros(
                    (self.num_layers, bsz, self.num_codebooks, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v_caches": torch.zeros(
                    (self.num_layers, bsz, self.num_codebooks, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "temp": torch.ones((bsz,), dtype=torch.float32, device=self.device),
                "top_k": torch.full((bsz,), 40, dtype=torch.long, device=self.device),
                "gumbel": torch.zeros(
                    (bsz, self.num_codebooks, self.codebook_size), dtype=torch.float32, device=self.device
                ),
                "out_codes": torch.zeros((bsz, self.num_codebooks), dtype=torch.long, device=self.device),
            }

    def capture_all(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        for bsz in self.buckets:
            if bsz in self._graphs:
                continue
            bufs = self._alloc_buffers(bsz)
            # Warmup
            for _ in range(2):
                self.depthformer.sample_codes(
                    bufs["temporal"],
                    k_caches=bufs["k_caches"],
                    v_caches=bufs["v_caches"],
                    temperature=bufs["temp"],
                    top_k=bufs["top_k"],
                    gumbel_noise=bufs["gumbel"],
                    out_codes=bufs["out_codes"],
                )
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.depthformer.sample_codes(
                    bufs["temporal"],
                    k_caches=bufs["k_caches"],
                    v_caches=bufs["v_caches"],
                    temperature=bufs["temp"],
                    top_k=bufs["top_k"],
                    gumbel_noise=bufs["gumbel"],
                    out_codes=bufs["out_codes"],
                )
            self._graphs[bsz] = (graph, bufs)

    def run(
        self,
        temporal_hidden: torch.Tensor,
        *,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        gumbel_noise: torch.Tensor,
    ) -> torch.Tensor:
        bsz = temporal_hidden.shape[0]
        bucket = next((b for b in self.buckets if b >= bsz), None) if self._graphs else None
        if bucket is not None and bucket in self._graphs:
            graph, bufs = self._graphs[bucket]
            bufs["temporal"][:bsz].copy_(temporal_hidden)
            bufs["temp"][:bsz].copy_(temperature)
            bufs["top_k"][:bsz].copy_(top_k)
            bufs["gumbel"][:bsz].copy_(gumbel_noise)
            graph.replay()
            return bufs["out_codes"][:bsz].clone()

        bufs = self._alloc_buffers(bsz)
        self.depthformer.sample_codes(
            temporal_hidden,
            k_caches=bufs["k_caches"],
            v_caches=bufs["v_caches"],
            temperature=temperature,
            top_k=top_k,
            gumbel_noise=gumbel_noise,
            out_codes=bufs["out_codes"],
        )
        return bufs["out_codes"]


class _CodeRowsSnapshot(PackedOutputSnapshot):
    """Async D2H-safe snapshot wrapping ``{"codes": {"audio": codes}}``."""


class MagentaRT2ForConditionalGeneration(nn.Module):
    """Stage-0 vLLM MRv2 Talker model for Google Magenta RealTime 2."""

    # Omni AR runner contract
    has_preprocess = False
    has_postprocess = True
    have_multimodal_outputs = True
    requires_raw_input_tokens = True
    omni_pooler_payload_include_hidden = False
    mrv2_forward_returns_tensor = True
    mm_outputs_fresh_per_step = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.config: MagentaRT2Config = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self._dtype: torch.dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
        self.allow_patterns_overrides = magenta_rt2_allow_patterns(
            getattr(vllm_config.model_config, "model", None), self.config
        )

        max_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 8) or 8)
        duplex_max = int(getattr(vllm_config.model_config, "duplex_max_sessions", 1) or 1)
        self.slot_capacity = max(max_seqs, duplex_max)
        self.scratch_slot = self.slot_capacity
        self.total_slots = self.slot_capacity + 1
        self.max_num_tokens = int(getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 512) or 512)

        self.self_ring_cap = int(self.config.temporal_self_past_horizon) + 1
        self.cross_ring_cap = int(self.config.temporal_cross_past_horizon) + 1

        self.encoder = MagentaRT2ConditioningEncoder(self.config)
        self.depthformer = MagentaRT2Depthformer(self.config)

        self.temporal_layers = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    hidden_size=int(self.config.temporal_hidden_size),
                    ffn_dim=int(self.config.temporal_ffn_dim),
                    num_heads=int(self.config.temporal_num_heads),
                    head_dim=int(self.config.temporal_head_dim),
                    encoder_dim=int(self.config.encoder_dim),
                    self_past_horizon=int(self.config.temporal_self_past_horizon),
                    cross_past_horizon=int(self.config.temporal_cross_past_horizon),
                    num_sinks=int(self.config.temporal_num_sinks),
                    norm_eps=float(self.config.norm_eps),
                    max_slots=self.total_slots,
                    dtype=self._dtype,
                )
                for _ in range(int(self.config.temporal_num_layers))
            ]
        )

        # Persistent fixed-address GPU slot + step buffers (shared with FULL CUDA graphs).
        self.register_buffer(
            "_slot_prev_codes",
            torch.zeros((self.total_slots, int(self.config.num_codebooks)), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_active_slot_ids",
            torch.full((self.max_num_tokens,), self.scratch_slot, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_active_cond_tokens",
            torch.zeros((self.max_num_tokens, ENCODER_NUM_CHANNELS), dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_active_write_pos",
            torch.zeros((self.max_num_tokens,), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_active_attn_mask",
            torch.zeros((self.max_num_tokens, self.self_ring_cap), dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_active_cross_write_pos",
            torch.zeros((self.max_num_tokens,), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_active_cross_attn_mask",
            torch.zeros((self.max_num_tokens, self.cross_ring_cap), dtype=torch.bool),
            persistent=False,
        )
        self._single_token_per_slot_hint = True
        self._multi_token_pending: bool = False
        self._multi_token_spans: list[tuple[int, int]] = []

    @classmethod
    def create_mrv2_model_state(
        cls,
        model: MagentaRT2ForConditionalGeneration,
        vllm_config: VllmConfig,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> MagentaRT2ModelState:
        return MagentaRT2ModelState(vllm_config, model, encoder_cache, device)

    def ensure_slot_capacity(self, slot_capacity: int) -> None:
        if slot_capacity <= self.slot_capacity:
            return
        device = self._slot_prev_codes.device
        self.slot_capacity = int(slot_capacity)
        self.scratch_slot = self.slot_capacity
        self.total_slots = self.slot_capacity + 1
        with torch.inference_mode(False):
            self._slot_prev_codes = torch.zeros(
                (self.total_slots, int(self.config.num_codebooks)), dtype=torch.long, device=device
            )
            for layer in self.temporal_layers:
                layer.self_attention.ensure_slot_capacity(self.total_slots)
                layer.cross_attention.ensure_slot_capacity(self.total_slots)

    def reset_slot(self, slot_id: int) -> None:
        if 0 <= slot_id < self.total_slots:
            self._slot_prev_codes[slot_id].zero_()
            for layer in self.temporal_layers:
                layer.self_attention.reset_slot(slot_id)
                layer.cross_attention.reset_slot(slot_id)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        th = int(self.config.temporal_hidden_size)
        if input_ids.numel() == 0:
            return torch.empty((0, th), device=input_ids.device, dtype=self._dtype)
        return torch.zeros((input_ids.shape[0], th), device=input_ids.device, dtype=self._dtype)

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        del sampling_metadata
        h = self.depthformer.final_ln(self.depthformer.depth_input_adapter(hidden_states))
        return self.depthformer.to_logits(h)

    def _ensure_token_capacity(self, num_tokens: int) -> None:
        if num_tokens <= self._active_slot_ids.shape[0]:
            return
        device = self._active_slot_ids.device
        self.max_num_tokens = int(num_tokens)
        with torch.inference_mode(False):
            self._active_slot_ids = torch.full((num_tokens,), self.scratch_slot, dtype=torch.long, device=device)
            self._active_cond_tokens = torch.zeros((num_tokens, ENCODER_NUM_CHANNELS), dtype=torch.int32, device=device)
            self._active_write_pos = torch.zeros((num_tokens,), dtype=torch.long, device=device)
            self._active_attn_mask = torch.zeros((num_tokens, self.self_ring_cap), dtype=torch.bool, device=device)
            self._active_cross_write_pos = torch.zeros((num_tokens,), dtype=torch.long, device=device)
            self._active_cross_attn_mask = torch.zeros(
                (num_tokens, self.cross_ring_cap), dtype=torch.bool, device=device
            )

    @torch.inference_mode()
    def forward_single_step_tokens(
        self,
        num_tokens: int,
        *,
        prev_codes_override: torch.Tensor | None = None,
        start: int = 0,
    ) -> torch.Tensor:
        """CUDA-graph-safe forward for ``[start : start + num_tokens]`` using fixed-address slot buffers."""
        end = start + num_tokens
        slot_ids = self._active_slot_ids[start:end]
        cond_tokens = self._active_cond_tokens[start:end]
        write_pos = self._active_write_pos[start:end]
        attn_mask = self._active_attn_mask[start:end]
        cross_write_pos = self._active_cross_write_pos[start:end]
        cross_attn_mask = self._active_cross_attn_mask[start:end]

        prev_codes = (
            self._slot_prev_codes.index_select(0, slot_ids) if prev_codes_override is None else prev_codes_override
        )
        # Mean over the 12 codebook embeddings in float32 matching JAX MultivariateDecoder.
        x = self.depthformer.embed_codes(prev_codes).float().mean(dim=-2).to(dtype=self._dtype)
        enc_hidden = self.encoder(cond_tokens).to(dtype=self._dtype)

        for layer in self.temporal_layers:
            x = layer(
                x,
                enc_hidden,
                slot_ids=slot_ids,
                self_write_pos=write_pos,
                self_attn_mask=attn_mask,
                cross_write_pos=cross_write_pos,
                cross_attn_mask=cross_attn_mask,
            )
        return x

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del positions, intermediate_tensors, kwargs
        th = int(self.config.temporal_hidden_size)
        if inputs_embeds is not None:
            num_tokens = int(inputs_embeds.shape[0])
        elif input_ids is not None:
            num_tokens = int(input_ids.shape[0])
        else:
            return torch.empty((0, th), dtype=self._dtype, device=self._slot_prev_codes.device)
        if num_tokens == 0:
            return torch.empty((0, th), dtype=self._dtype, device=self._slot_prev_codes.device)
        if self._multi_token_pending and not torch.cuda.is_current_stream_capturing():
            # Eager multi-token prefill for a request: step causal frames in order so
            # ring writes never race within the same slot.
            spans = self._multi_token_spans
            self._multi_token_pending = False
            self._multi_token_spans = []
            out = torch.zeros((num_tokens, th), dtype=self._dtype, device=self._slot_prev_codes.device)
            max_span = max((end - start for start, end in spans), default=0)
            for k in range(max_span):
                for start, end in spans:
                    idx = start + k
                    if idx < end:
                        out[idx : idx + 1] = self.forward_single_step_tokens(1, start=idx)
            return out
        return self.forward_single_step_tokens(num_tokens)

    # ------------------------------------------------------------------
    # Weight loading (Flax safetensors & HF state dicts)
    # ------------------------------------------------------------------

    def load_flax_params(self, jax_params: dict[str, Any]) -> None:
        jax_df = jax_params["depthformer"]
        jax_enc_body = jax_df["encoder"]["body"]

        if "layers_1" in jax_enc_body:
            branched = jax_enc_body["layers_1"]
            mulan = branched["branched_mulan_embedder"]["mulan_embedder"]
            self.encoder.mulan_dequantizer.weight.data.copy_(
                torch.as_tensor(
                    mulan["mulan_dequantizer"]["embedding"], dtype=self.encoder.mulan_dequantizer.weight.dtype
                )
            )
            self.encoder.depth_input_adapter.weight.data.copy_(
                torch.as_tensor(
                    mulan["depth_input_adapter"]["kernel"], dtype=self.encoder.depth_input_adapter.weight.dtype
                ).t()
            )
            reg = branched["branched_regular_embedder"]["regular_embedder"]
            self.encoder.regular_embedder.weight.data.copy_(
                torch.as_tensor(reg["embedding"], dtype=self.encoder.regular_embedder.weight.dtype)
            )

        enc_ln = jax_enc_body["encoder_ln"]
        self.encoder.encoder_ln.weight.data.copy_(
            torch.as_tensor(enc_ln["scale"], dtype=self.encoder.encoder_ln.weight.dtype)
        )
        if "bias" in enc_ln:
            self.encoder.encoder_ln.bias.data.copy_(
                torch.as_tensor(enc_ln["bias"], dtype=self.encoder.encoder_ln.bias.dtype)
            )

        jax_dec = jax_df["decoder"]
        self.depthformer.decoder_embedding.weight.data.copy_(
            torch.as_tensor(
                jax_dec["decoder_embedding"]["embedding"]["embedding"],
                dtype=self.depthformer.decoder_embedding.weight.dtype,
            )
        )

        jax_temporal = jax_dec["temporal_body"]["transformer"]
        for i, layer in enumerate(self.temporal_layers):
            jblk = jax_temporal[f"x_layers_{i}"]
            load_flax_attention_weights(layer.self_attention, jblk["self_attention"])
            load_flax_attention_weights(layer.cross_attention, jblk["cross_attention"])
            load_flax_ffn_weights(layer.ffn, jblk["ffn"])

        jax_depth = jax_dec["depth_body"]
        if isinstance(self.depthformer.depth_input_adapter, nn.Linear) and "depth_input_adapter" in jax_depth:
            self.depthformer.depth_input_adapter.weight.data.copy_(
                torch.as_tensor(
                    jax_depth["depth_input_adapter"]["kernel"],
                    dtype=self.depthformer.depth_input_adapter.weight.dtype,
                ).t()
            )
        jax_depth_tf = jax_depth["transformer"]
        for i, blk in enumerate(self.depthformer.blocks):
            jblk = jax_depth_tf[f"x_layers_{i}"]
            load_flax_attention_weights(blk.self_attention, jblk["self_attention"])
            load_flax_ffn_weights(blk.ffn, jblk["ffn"])

        fin_ln = jax_depth["final_ln"]
        self.depthformer.final_ln.weight.data.copy_(
            torch.as_tensor(fin_ln["scale"], dtype=self.depthformer.final_ln.weight.dtype)
        )
        if "bias" in fin_ln:
            self.depthformer.final_ln.bias.data.copy_(
                torch.as_tensor(fin_ln["bias"], dtype=self.depthformer.final_ln.bias.dtype)
            )
        to_logits = jax_depth["to_logits"]
        self.depthformer.to_logits.weight.data.copy_(
            torch.as_tensor(to_logits["kernel"], dtype=self.depthformer.to_logits.weight.dtype).t()
        )
        if self.depthformer.to_logits.bias is not None and "bias" in to_logits:
            self.depthformer.to_logits.bias.data.copy_(
                torch.as_tensor(to_logits["bias"], dtype=self.depthformer.to_logits.bias.dtype)
            )

    def _find_flax_checkpoint(self) -> Path | None:
        model_path = getattr(self.vllm_config.model_config, "model", None)
        return resolve_magenta_rt2_checkpoint(model_path, self.config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from either vLLM safetensors iterator (Flax or PyTorch keys)."""
        flat_flax: dict[str, torch.Tensor] = {}
        state = self.state_dict()
        loaded: set[str] = set()

        for name, tensor in weights:
            if name.startswith("params/"):
                flat_flax[name] = tensor
                continue
            if name in state and state[name].shape == tensor.shape:
                state[name].data.copy_(tensor.to(dtype=state[name].dtype))
                loaded.add(name)

        if not flat_flax and not loaded:
            ckpt = self._find_flax_checkpoint()
            if ckpt is not None:
                with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as f:
                    for key in f.keys():  # noqa: SIM118
                        if key.startswith("params/depthformer/"):
                            flat_flax[key] = f.get_tensor(key)

        if flat_flax:
            nested = _unflatten_nested({k: v for k, v in flat_flax.items() if k.startswith("params/depthformer/")})
            if "params" in nested and "depthformer" in nested["params"]:
                self.load_flax_params(nested["params"])
                loaded.update(self.state_dict().keys())
                logger.info("Loaded MagentaRT2 Talker Flax weights (%d parameters)", len(loaded))
        return loaded


class MagentaRT2ModelState(OmniModelState):
    """MRv2 ModelState owning slot allocation, conditioning updates, and Depthformer sampling."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: MagentaRT2ForConditionalGeneration,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.mrt_model = model
        self.max_slots = model.slot_capacity
        self.free_slots: list[int] = list(range(self.max_slots - 1, -1, -1))
        self.req_to_slot: dict[str, int] = {}
        self.slot_steps: dict[int, int] = {}
        self.slot_conditioning: dict[int, np.ndarray] = {}
        self.slot_temperature: dict[int, float] = {}
        self.slot_top_k: dict[int, int] = {}
        self.slot_generators: dict[int, torch.Generator | None] = {}
        self._last_codes: torch.Tensor | None = None
        self._snapshot_slots: list[dict[tuple[Any, ...], torch.Tensor]] = [{} for _ in range(4)]
        self._snapshot_cursor = 0

        use_graphs = True
        hf_cfg = getattr(vllm_config.model_config, "hf_config", None)
        if hf_cfg is not None:
            use_graphs = bool(getattr(hf_cfg, "depthformer_cuda_graphs", True))
        self.depth_graph_runner = (
            DepthformerGraphRunner(
                model.depthformer,
                device=device,
                dtype=model._dtype,
                max_batch_size=self.max_slots,
            )
            if use_graphs and device.type == "cuda"
            else None
        )
        # Reusable pinned/CPU staging buffers for non-blocking batched H2D.
        self._max_tokens = int(model.max_num_tokens)
        self._default_cond = build_conditioning_tokens(cfg_scales=DEFAULT_CFG_SCALES)

    def capture_inner_cudagraphs(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if self.depth_graph_runner is not None:
            self.depth_graph_runner.capture_all()

    def _alloc_slot(self, req_id: str) -> int:
        if req_id in self.req_to_slot:
            return self.req_to_slot[req_id]
        if not self.free_slots:
            raise RuntimeError(f"MagentaRT2 slot pool exhausted (capacity={self.max_slots})")
        slot = self.free_slots.pop()
        self.req_to_slot[req_id] = slot
        self.slot_steps[slot] = 0
        self.slot_conditioning[slot] = self._default_cond.copy()
        self.slot_temperature[slot] = float(getattr(self.mrt_model.config, "default_temperature", 1.3))
        self.slot_top_k[slot] = int(getattr(self.mrt_model.config, "default_top_k", 40))
        self.slot_generators[slot] = None
        self.mrt_model.reset_slot(slot)
        return slot

    def _free_slot(self, req_id: str) -> None:
        slot = self.req_to_slot.pop(req_id, None)
        if slot is None:
            return
        self.slot_steps.pop(slot, None)
        self.slot_conditioning.pop(slot, None)
        self.slot_temperature.pop(slot, None)
        self.slot_top_k.pop(slot, None)
        self.slot_generators.pop(slot, None)
        self.mrt_model.reset_slot(slot)
        self.free_slots.append(slot)

    @staticmethod
    def _extract_conditioning_from_buf(
        buf: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray | None, float | None, int | None]:
        if not isinstance(buf, Mapping):
            return None, None, None
        sources: list[Mapping[str, Any]] = [buf]
        duplex = buf.get("duplex")
        if isinstance(duplex, Mapping):
            rc = duplex.get("runtime_config")
            if isinstance(rc, Mapping):
                sources.append(rc)
            pl = duplex.get("payload")
            if isinstance(pl, Mapping):
                sources.append(pl)

        raw = None
        mulan = None
        piano = None
        drum = None
        cfg_scales: dict[str, float] | None = None
        temp: float | None = None
        top_k: int | None = None

        for src in sources:
            if "raw_conditioning_tokens" in src and src["raw_conditioning_tokens"] is not None:
                raw = src["raw_conditioning_tokens"]
            if "mulan_tokens" in src and src["mulan_tokens"] is not None:
                mulan = src["mulan_tokens"]
            if "pianoroll_tokens" in src and src["pianoroll_tokens"] is not None:
                piano = src["pianoroll_tokens"]
            if "drum_tokens" in src and src["drum_tokens"] is not None:
                drum = src["drum_tokens"]
            if isinstance(src.get("cfg_scales"), Mapping):
                cfg_scales = {str(k): float(v) for k, v in src["cfg_scales"].items() if v is not None}
            if src.get("temperature") is not None:
                temp = float(src["temperature"])
            if src.get("top_k") is not None:
                top_k = int(src["top_k"])

        if any(x is not None for x in (raw, mulan, piano, drum, cfg_scales)):
            cond = build_conditioning_tokens(
                mulan_tokens=mulan,
                pianoroll_tokens=piano,
                drum_tokens=drum,
                cfg_scales=cfg_scales,
                raw_conditioning_tokens=raw,
                already_offset=bool(buf.get("conditioning_already_offset", False)),
            )
            return cond, temp, top_k
        return None, temp, top_k

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        req_id = new_req_data.req_id
        if not is_warmup_request(req_id):
            slot = self._alloc_slot(req_id)
            sp = new_req_data.sampling_params
            if sp is not None:
                if getattr(sp, "temperature", None) is not None:
                    self.slot_temperature[slot] = float(sp.temperature)
                if getattr(sp, "top_k", None) is not None and int(sp.top_k) > 0:
                    self.slot_top_k[slot] = int(sp.top_k)
                seed = getattr(sp, "seed", None)
                if seed is not None:
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(int(seed))
                    self.slot_generators[slot] = gen
            buf = self.intermediate_buffer.buffers[req_index]
            cond, temp, top_k = self._extract_conditioning_from_buf(buf)
            if cond is not None:
                self.slot_conditioning[slot] = cond
            if temp is not None:
                self.slot_temperature[slot] = temp
            if top_k is not None:
                self.slot_top_k[slot] = top_k

    def remove_request(self, req_index_or_id: int | str) -> None:
        req_idx = self._resolve_req_index(req_index_or_id)
        req_id: str | None = req_index_or_id if isinstance(req_index_or_id, str) else None
        if req_idx is not None and 0 <= req_idx < len(self.intermediate_buffer.buffers):
            buf_req_id = self.intermediate_buffer.buffers[req_idx].get("req_id")
            if isinstance(buf_req_id, str):
                req_id = buf_req_id
        super().remove_request(req_index_or_id)
        if isinstance(req_id, str):
            self._free_slot(req_id)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        parent_fn = getattr(super(), "on_requests_finished", None)
        if callable(parent_fn):
            parent_fn(req_ids)
        for req_id in req_ids:
            self._free_slot(req_id)

    def _stage_slot_tensors(
        self,
        slot_ids_cpu: np.ndarray,
        steps_cpu: np.ndarray,
        cond_cpu: np.ndarray,
        total_tokens: int,
    ) -> None:
        self.mrt_model._ensure_token_capacity(total_tokens)
        device = self.device
        self_cap = self.mrt_model.self_ring_cap
        cross_cap = self.mrt_model.cross_ring_cap

        self_pos = steps_cpu % self_cap
        cross_pos = steps_cpu % cross_cap

        # Valid time positions are t in [max(0, step - horizon), step], mapped to ring physical indices.
        phys_self = np.arange(self_cap, dtype=np.int64)[None, :]
        dist_self = (self_pos[:, None] - phys_self + self_cap) % self_cap
        t_self = steps_cpu[:, None] - dist_self
        mask_self = (t_self >= 0) & (dist_self < self_cap)

        phys_cross = np.arange(cross_cap, dtype=np.int64)[None, :]
        dist_cross = (cross_pos[:, None] - phys_cross + cross_cap) % cross_cap
        t_cross = steps_cpu[:, None] - dist_cross
        mask_cross = (t_cross >= 0) & (dist_cross < cross_cap)

        self.mrt_model._active_slot_ids[:total_tokens].copy_(
            torch.from_numpy(slot_ids_cpu).to(device=device), non_blocking=True
        )
        self.mrt_model._active_cond_tokens[:total_tokens].copy_(
            torch.from_numpy(cond_cpu).to(device=device), non_blocking=True
        )
        self.mrt_model._active_write_pos[:total_tokens].copy_(
            torch.from_numpy(self_pos).to(device=device), non_blocking=True
        )
        self.mrt_model._active_attn_mask[:total_tokens].copy_(
            torch.from_numpy(mask_self).to(device=device), non_blocking=True
        )
        self.mrt_model._active_cross_write_pos[:total_tokens].copy_(
            torch.from_numpy(cross_pos).to(device=device), non_blocking=True
        )
        self.mrt_model._active_cross_attn_mask[:total_tokens].copy_(
            torch.from_numpy(mask_cross).to(device=device), non_blocking=True
        )

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        base = super().prepare_dummy_inputs(num_reqs, num_tokens)
        self.mrt_model._multi_token_pending = False
        self.mrt_model._multi_token_spans = []
        if num_tokens > 0:
            slot_ids = np.full((num_tokens,), self.mrt_model.scratch_slot, dtype=np.int64)
            steps = np.zeros((num_tokens,), dtype=np.int64)
            cond = np.broadcast_to(self._default_cond[None, :], (num_tokens, ENCODER_NUM_CHANNELS)).copy()
            self._stage_slot_tensors(slot_ids, steps, cond, num_tokens)
        return base

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        base = super().prepare_inputs(input_batch, req_states)
        self.mrt_model._multi_token_pending = False
        self.mrt_model._multi_token_spans = []
        total_tokens = int(getattr(input_batch, "num_tokens_after_padding", input_batch.num_tokens))
        if total_tokens <= 0:
            return base

        slot_ids = np.full((total_tokens,), self.mrt_model.scratch_slot, dtype=np.int64)
        steps = np.zeros((total_tokens,), dtype=np.int64)
        cond = np.broadcast_to(self._default_cond[None, :], (total_tokens, ENCODER_NUM_CHANNELS)).copy()
        spans: list[tuple[int, int]] = []
        has_multi_tok = False

        for i in range(input_batch.num_reqs):
            req_id = input_batch.req_ids[i]
            if is_warmup_request(req_id):
                continue
            req_idx = int(input_batch.idx_mapping_np[i])
            buf = self.intermediate_buffer.buffers[req_idx]
            slot = self.req_to_slot.get(req_id)
            if slot is None:
                slot = self._alloc_slot(req_id)
            new_cond, temp, top_k = self._extract_conditioning_from_buf(buf)
            if new_cond is not None:
                self.slot_conditioning[slot] = new_cond
            if temp is not None:
                self.slot_temperature[slot] = temp
            if top_k is not None:
                self.slot_top_k[slot] = top_k

            start = int(input_batch.query_start_loc_np[i])
            n_tok = int(input_batch.num_scheduled_tokens[i])
            if n_tok > 1:
                has_multi_tok = True
            if n_tok > 0:
                spans.append((start, start + n_tok))
            base_step = self.slot_steps.get(slot, 0)
            req_cond = self.slot_conditioning.get(slot, self._default_cond)
            for k in range(n_tok):
                idx = start + k
                slot_ids[idx] = slot
                steps[idx] = base_step + k
                cond[idx] = req_cond

        if has_multi_tok:
            self.mrt_model._multi_token_pending = True
            self.mrt_model._multi_token_spans = spans
        self._stage_slot_tensors(slot_ids, steps, cond, total_tokens)
        return base

    def run_postprocess(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        multimodal_outputs: Any | None = None,
    ) -> None:
        del multimodal_outputs
        self._last_codes = None
        if input_batch.num_reqs == 0:
            return

        total_real = int(np.asarray(input_batch.num_scheduled_tokens[: input_batch.num_reqs]).sum())
        if total_real == 0:
            return

        real_indices: list[int] = []
        temps: list[float] = []
        top_ks: list[int] = []
        row_gens: list[torch.Generator | None] = []
        last_slots: list[int] = []
        last_code_rows: list[int] = []

        for i in range(input_batch.num_reqs):
            req_id = input_batch.req_ids[i]
            n_tok = int(input_batch.num_scheduled_tokens[i])
            start = int(input_batch.query_start_loc_np[i])
            slot = self.req_to_slot.get(req_id)
            warmup = is_warmup_request(req_id)
            for k in range(n_tok):
                real_indices.append(start + k)
                if slot is not None and not warmup:
                    temps.append(self.slot_temperature.get(slot, 1.3))
                    top_ks.append(self.slot_top_k.get(slot, 40))
                    row_gens.append(self.slot_generators.get(slot))
                else:
                    temps.append(0.0)
                    top_ks.append(1)
                    row_gens.append(None)
            if slot is not None and not warmup and n_tok > 0:
                last_slots.append(slot)
                last_code_rows.append(len(real_indices) - 1)
                self.slot_steps[slot] = self.slot_steps.get(slot, 0) + n_tok

        if real_indices == list(range(len(real_indices))):
            h_real = hidden_states[: len(real_indices)]
        else:
            real_idx_t = torch.tensor(real_indices, dtype=torch.long, device=hidden_states.device)
            h_real = hidden_states.index_select(0, real_idx_t)

        bsz = h_real.shape[0]
        temp_t = torch.tensor(temps, dtype=torch.float32, device=h_real.device)
        topk_t = torch.tensor(top_ks, dtype=torch.long, device=h_real.device)
        q_dim = self.mrt_model.depthformer.num_codebooks
        v_dim = self.mrt_model.depthformer.codebook_size
        if all(g is None for g in row_gens):
            gumbel = _sample_gumbel((bsz, q_dim, v_dim), h_real.device)
        else:
            gumbel = torch.empty((bsz, q_dim, v_dim), dtype=torch.float32, device=h_real.device)
            unseeded_rows = [idx for idx, g in enumerate(row_gens) if g is None]
            if unseeded_rows:
                unseeded_noise = _sample_gumbel((len(unseeded_rows), q_dim, v_dim), h_real.device)
                for src_i, dst_i in enumerate(unseeded_rows):
                    gumbel[dst_i].copy_(unseeded_noise[src_i])
            for row_idx, row_gen in enumerate(row_gens):
                if row_gen is not None:
                    gumbel[row_idx].copy_(_sample_gumbel((q_dim, v_dim), h_real.device, generator=row_gen))

        if self.depth_graph_runner is not None:
            codes = self.depth_graph_runner.run(h_real, temperature=temp_t, top_k=topk_t, gumbel_noise=gumbel)
        else:
            out_codes = torch.empty(
                (bsz, self.mrt_model.depthformer.num_codebooks), dtype=torch.long, device=h_real.device
            )
            num_layers = len(self.mrt_model.depthformer.blocks)
            k_c = torch.zeros(
                (
                    num_layers,
                    bsz,
                    self.mrt_model.depthformer.num_codebooks,
                    int(self.mrt_model.config.depth_num_heads),
                    int(self.mrt_model.config.depth_head_dim),
                ),
                dtype=h_real.dtype,
                device=h_real.device,
            )
            v_c = torch.zeros_like(k_c)
            codes = self.mrt_model.depthformer.sample_codes(
                h_real,
                k_caches=k_c,
                v_caches=v_c,
                temperature=temp_t,
                top_k=topk_t,
                gumbel_noise=gumbel,
                out_codes=out_codes,
            )

        # Update slot prev_codes on GPU directly for the last frame of each request.
        if last_slots:
            slot_idx_t = torch.tensor(last_slots, dtype=torch.long, device=codes.device)
            row_idx_t = torch.tensor(last_code_rows, dtype=torch.long, device=codes.device)
            self.mrt_model._slot_prev_codes[slot_idx_t] = codes[row_idx_t]

        self._last_codes = codes.to(dtype=torch.int32)

    def postprocess_model_output(
        self,
        model_output: Any,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, dict]:
        text_hidden, _ = super().postprocess_model_output(model_output, input_batch, req_states)
        if self._last_codes is None or self._last_codes.numel() == 0:
            return text_hidden, {}
        codes = self._last_codes
        self._last_codes = None

        payload: dict[str, Any] = {"codes": {"audio": codes}}
        if codes.device.type == "cuda":
            slot_idx = self._snapshot_cursor
            self._snapshot_cursor = (self._snapshot_cursor + 1) % len(self._snapshot_slots)
            snap = pack_output_snapshot(
                payload,
                self._snapshot_slots[slot_idx],
                max_buckets=4,
                reuse_existing_storage=True,
            )
            if snap is not None:
                return text_hidden, _CodeRowsSnapshot(snap._leaves, snap._spec, snap._groups)
        return text_hidden, payload

    def sample_determined_tokens(self, input_batch: InputBatch, sampler: Any) -> SamplerOutput | None:
        """Bypass lm_head + Sampler on Stage 0: audio codebooks were already sampled by Depthformer."""
        del sampler
        num_reqs = input_batch.num_reqs
        if num_reqs <= 0:
            return None
        device = self.device
        return SamplerOutput(
            sampled_token_ids=torch.zeros((num_reqs, 1), dtype=torch.int32, device=device),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones((num_reqs,), dtype=torch.int32, device=device),
            num_rejected=torch.zeros((num_reqs,), dtype=torch.int32, device=device),
        )


__all__ = [
    "DepthformerGraphRunner",
    "MagentaRT2Depthformer",
    "MagentaRT2ForConditionalGeneration",
    "MagentaRT2ModelState",
]
