# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 codebook tower: predicts codes 1..num_codebooks-1 of each frame.

Every tower layer attends causally over the whole sequence, so each layer
keeps its own key/value history per request. The tower at sequence position
``p`` reads the backbone's final-norm hidden state at ``p`` and the frame at
``p`` (input), and predicts the codes of frame ``p + 1``.

Module names follow the checkpoint (``audio_decoder*``) so weights load by name.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import CUDAGraph
from vllm.platforms import current_platform

from .configuration_fish_omni import FishOmniAudioDecoderConfig


@functools.cache
def paged_flash_attention() -> tuple[Callable[..., Any], int] | None:
    """vLLM's paged varlen flash attention and the version to use, or None when
    this build has no flash-attention extension (for example on ROCm)."""
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported
    except ImportError:
        return None
    # FA3 has native Hopper kernels; the FA2 build needs a slow driver JIT there.
    version = 3 if current_platform.is_device_capability(90) and is_fa_version_supported(3) else 2
    return flash_attn_varlen_func, version


class GemmaRMSNorm(nn.Module):
    """RMSNorm with a zero-centered gain (scale = 1 + weight); with ``residual``
    it normalizes ``x + residual`` and also returns that sum."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (h * (1.0 + self.weight.float())).to(x.dtype)
        return out if residual is None else (out, residual)


def interleaved_rope(states: torch.Tensor, positions: torch.Tensor, rotary_dim: int, base: float) -> torch.Tensor:
    """GPT-J rope on the first ``rotary_dim`` channels of [rows, heads, head_dim]."""
    inv_freq = 1.0 / base ** (torch.arange(0, rotary_dim, 2, device=states.device, dtype=torch.float32) / rotary_dim)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    rotary = states[..., :rotary_dim].float()
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((rotated.to(states.dtype), states[..., rotary_dim:]), dim=-1)


class TowerKVCache:
    """Per-request key/value history for every tower layer.

    Each layer owns one key and one value tensor [slots, max_len, kv_heads,
    head_dim]; flash-attention reads it as a paged cache with one block per
    slot. Allocated on first use; a request takes a slot at prefill and keeps
    it until the slot is reclaimed for a newer request (least recently used).
    """

    def __init__(self, *, slots: int, layers: int, max_len: int, kv_heads: int, head_dim: int) -> None:
        self.slots = slots
        self.layers = layers
        # Paged flash-attention needs the block size to be a multiple of 256.
        self.max_len = -(-max_len // 256) * 256
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self._owner: list[object | None] = [None] * slots
        self._last_use = [0] * slots
        self._clock = 0

    @property
    def scratch_slot(self) -> int:
        """A slot no request owns; padding rows of captured graphs write here."""
        return self.slots

    def ensure(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self.keys:
            shape = (self.slots + 1, self.max_len, self.kv_heads, self.head_dim)
            self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.layers)]
            self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.layers)]

    def acquire(self, owner: object, in_use: set[int]) -> int:
        self._clock += 1
        free = [s for s in range(self.slots) if self._owner[s] is None]
        if not free:
            free = sorted((s for s in range(self.slots) if s not in in_use), key=lambda s: self._last_use[s])
        if not free:
            raise RuntimeError(f"all {self.slots} tower cache slots are in use")
        slot = free[0]
        self._owner[slot] = owner
        self._last_use[slot] = self._clock
        return slot

    def release(self, slot: int, owner: object) -> None:
        if 0 <= slot < self.slots and self._owner[slot] is owner:
            self._owner[slot] = None

    def owns(self, slot: int, owner: object) -> bool:
        return self._owner[slot] is owner

    def touch(self, slot: int) -> None:
        self._clock += 1
        self._last_use[slot] = self._clock


def to_device(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy small host metadata to ``device`` without stalling the host on CUDA."""
    if device.type == "cuda":
        return host.pin_memory().to(device, non_blocking=True)
    return host.to(device)


@dataclass(frozen=True)
class TowerBatch:
    """Where this call's rows sit in the cache.

    Rows are grouped by request: decode has one row per request, prefill one
    request with many rows. ``seq_used`` counts each group's cached keys after
    this call's rows are written.
    """

    slots: torch.Tensor  # [rows] cache slot per row
    positions: torch.Tensor  # [rows]
    block_table: torch.Tensor  # [groups, 1] int32 slot per group
    cu_seqlens_q: torch.Tensor  # [groups + 1] int32
    seq_used: torch.Tensor  # [groups] int32
    max_rows: int

    @staticmethod
    def decode(slots: torch.Tensor, positions: torch.Tensor) -> TowerBatch:
        rows = slots.shape[0]
        return TowerBatch(
            slots=slots,
            positions=positions,
            block_table=slots.to(torch.int32).view(rows, 1),
            cu_seqlens_q=torch.arange(rows + 1, device=slots.device, dtype=torch.int32),
            seq_used=(positions + 1).to(torch.int32),
            max_rows=1,
        )

    @staticmethod
    def extend(slot: int, first: int, end: int, device: torch.device) -> TowerBatch:
        """Rows ``first..end-1`` of one request; earlier positions are already cached."""
        rows = end - first
        return TowerBatch(
            slots=torch.full((rows,), slot, device=device, dtype=torch.long),
            positions=torch.arange(first, end, device=device),
            block_table=torch.full((1, 1), slot, device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, rows], device=device, dtype=torch.int32),
            seq_used=torch.tensor([end], device=device, dtype=torch.int32),
            max_rows=rows,
        )

    @staticmethod
    def groups(slots: list[int], firsts: list[int], ends: list[int], device: torch.device) -> TowerBatch:
        """Rows ``firsts[g]..ends[g]-1`` of request ``g``, for several requests at once."""
        counts = [end - first for first, end in zip(firsts, ends)]
        row_slots = [slot for slot, count in zip(slots, counts) for _ in range(count)]
        row_positions = [p for first, end in zip(firsts, ends) for p in range(first, end)]
        bounds = [0]
        for count in counts:
            bounds.append(bounds[-1] + count)
        rows = len(row_slots)
        row_data = to_device(torch.tensor(row_slots + row_positions, dtype=torch.long), device)
        meta = to_device(torch.tensor(slots + bounds + ends, dtype=torch.int32), device)
        groups = len(slots)
        return TowerBatch(
            slots=row_data[:rows],
            positions=row_data[rows:],
            block_table=meta[:groups].view(groups, 1),
            cu_seqlens_q=meta[groups : 2 * groups + 1],
            seq_used=meta[2 * groups + 1 :],
            max_rows=max(counts),
        )

    @staticmethod
    def prefill(slot: int, rows: int, device: torch.device) -> TowerBatch:
        return TowerBatch.extend(slot, 0, rows, device)


@dataclass(eq=False)
class TowerState:
    """Per-request tower progress, kept in the request's intermediate buffer."""

    slot: int
    next_position: int
    input_frame: torch.Tensor  # [num_codebooks] codes of the frame at next_position
    input_is_semantic: torch.Tensor  # [] bool, on device
    # Prompt rows still to teacher-force through the tower: (hidden [n, dim], frames [n + 1, 1 + Q]).
    pending: tuple[torch.Tensor, torch.Tensor] | None = None
    # Codes [Q] of every generated frame so far, to rebuild the rows after a preemption.
    generated: list[torch.Tensor] = field(default_factory=list)


class TowerAttention(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv = config.n_local_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_base = config.rope_base
        q_width = self.n_head * self.head_dim
        kv_width = self.n_kv * self.head_dim
        # Checkpoint rows: [query | output gate | key | value].
        self.wqkv = nn.Linear(config.dim, 2 * q_width + 2 * kv_width, bias=False)
        self.wo = nn.Linear(q_width, config.dim, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        batch: TowerBatch,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Write this call's keys/values at ``batch`` and attend causally over each
        request's history. x [rows, dim] -> [rows, dim]."""
        rows = x.shape[0]
        q_width = self.n_head * self.head_dim
        kv_width = self.n_kv * self.head_dim
        query, gate, key, value = self.wqkv(x).split([q_width, q_width, kv_width, kv_width], dim=-1)
        query = self.q_norm(query.reshape(-1, self.head_dim)).view(rows, self.n_head, self.head_dim)
        key = self.k_norm(key.reshape(-1, self.head_dim)).view(rows, self.n_kv, self.head_dim)
        query = interleaved_rope(query, batch.positions, self.rotary_dim, self.rope_base)
        key = interleaved_rope(key, batch.positions, self.rotary_dim, self.rope_base)
        value = value.reshape(rows, self.n_kv, self.head_dim)
        key_cache[batch.slots, batch.positions] = key.to(key_cache.dtype)
        value_cache[batch.slots, batch.positions] = value.to(value_cache.dtype)
        flash = paged_flash_attention() if key_cache.is_cuda else None
        if flash is not None:
            flash_attn_varlen_func, fa_version = flash
            attended = flash_attn_varlen_func(
                q=query.contiguous(),
                k=key_cache,
                v=value_cache,
                max_seqlen_q=batch.max_rows,
                cu_seqlens_q=batch.cu_seqlens_q,
                max_seqlen_k=key_cache.shape[1],
                seqused_k=batch.seq_used,
                block_table=batch.block_table,
                softmax_scale=self.head_dim**-0.5,
                causal=True,
                fa_version=fa_version,
            )
        else:
            attended = self._attend_reference(query, batch, key_cache, value_cache)
        return self.wo(attended.reshape(rows, q_width) * torch.sigmoid(gate))

    @staticmethod
    def _attend_reference(
        query: torch.Tensor, batch: TowerBatch, key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> torch.Tensor:
        """Plain PyTorch causal attention over each group's history (no flash attention)."""
        outputs = []
        bounds = batch.cu_seqlens_q.tolist()
        for group, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            slot = int(batch.block_table[group, 0])
            used = int(batch.seq_used[group])
            keys = key_cache[slot, :used].transpose(0, 1).unsqueeze(0)
            values = value_cache[slot, :used].transpose(0, 1).unsqueeze(0)
            positions = batch.positions[start:stop]
            mask = torch.arange(used, device=query.device).unsqueeze(0) <= positions.unsqueeze(1)
            out = F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                keys,
                values,
                attn_mask=mask[None, None],
                enable_gqa=True,
            )
            outputs.append(out[0].transpose(0, 1))
        return torch.cat(outputs)


class TowerFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerBlock(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        self.attention = TowerAttention(config)
        self.feed_forward = TowerFeedForward(config.dim, config.intermediate_size)
        self.attention_norm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = GemmaRMSNorm(config.dim, config.norm_eps)


class TowerLayer(nn.Module):
    """Fuse a code embedding with the incoming hidden state (eh_proj), then apply
    one attention + feed-forward block."""

    def __init__(self, config: FishOmniAudioDecoderConfig, embedding_name: str) -> None:
        super().__init__()
        self.embedding_name = embedding_name
        self.add_module(embedding_name, nn.Embedding(config.vocab_size, config.dim))
        self.enorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.hnorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerBlock(config)
        self.final_layernorm = GemmaRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        code_ids: torch.Tensor,
        code_is_valid: torch.Tensor,
        extra_input: torch.Tensor | None,
        batch: TowerBatch,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        table: nn.Embedding = getattr(self, self.embedding_name)
        embedded = torch.where(
            code_is_valid.unsqueeze(-1),
            table(code_ids.clamp(0, table.num_embeddings - 1)),
            0.0,
        )
        if extra_input is not None:
            embedded = embedded + extra_input.to(embedded.dtype)
        mixed = self.eh_proj(torch.cat((self.enorm(embedded), self.hnorm(hidden)), dim=-1))
        block = self.transformer_layer
        attended = block.attention(block.attention_norm(mixed), batch, key_cache, value_cache)
        ffn_input, residual = block.ffn_norm(attended, mixed)
        out, _ = self.final_layernorm(block.feed_forward(ffn_input), residual)
        return out


class _LayerList(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _ResidualTower(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        count = config.num_codebooks - 1
        self.layers = nn.ModuleList(TowerLayer(config, "codebook_embedding") for _ in range(count))
        self.heads = nn.ModuleList(nn.Linear(config.dim, config.vocab_size, bias=False) for _ in range(count))


class _FrameEmbeddings(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        self.codebook_embeddings = nn.Embedding(config.vocab_size * config.num_codebooks, config.dim)


# Temperatures at or below this are greedy (vLLM marks greedy rows with -1).
_GREEDY_TEMPERATURE = 1e-5


def sample_rows(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generators: dict[int, torch.Generator] | None = None,
) -> torch.Tensor:
    """Sample a column index per row, with per-row temperature, top-k and top-p.

    A candidate survives when its rank is below top_k and the probability mass
    before and including it, over the whole row, stays within top_p; the most
    likely candidate is never filtered out. Greedy rows take the argmax. Rows
    listed in ``generators`` draw from their own generator.
    """
    greedy = temperature <= _GREEDY_TEMPERATURE
    scaled = logits.float() / temperature.float().clamp(min=_GREEDY_TEMPERATURE).unsqueeze(-1)
    sorted_logits, order = torch.sort(scaled, dim=-1, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    ranks = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    drop = (cumulative > top_p.float().unsqueeze(-1)) | (ranks >= top_k.unsqueeze(-1))
    drop[:, 0] = False
    probs = torch.softmax(sorted_logits.masked_fill(drop, float("-inf")), dim=-1)
    noise = torch.empty_like(probs).exponential_(1.0)
    for row, generator in (generators or {}).items():
        if row < noise.shape[0]:
            noise[row].exponential_(1.0, generator=generator)
    choice = torch.argmax(probs / noise, dim=-1, keepdim=True)
    choice = torch.where(greedy.unsqueeze(-1), torch.zeros_like(choice), choice)
    return torch.gather(order, -1, choice).squeeze(-1)


def sample_top_k_top_p(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """sample_rows with the same scalar parameters for every row."""
    rows, width = logits.shape
    k = top_k if 0 < top_k <= width else width
    return sample_rows(
        logits,
        torch.full((rows,), temperature, device=logits.device),
        torch.full((rows,), k, device=logits.device, dtype=torch.long),
        torch.full((rows,), top_p, device=logits.device),
    )


class CodebookTower(nn.Module):
    """Frame embeddings, 4 shared layers, and 9 residual layers with heads."""

    def __init__(self, config: FishOmniAudioDecoderConfig, *, semantic_begin_id: int, cache_slots: int, max_len: int):
        super().__init__()
        self.config = config
        self.semantic_begin_id = semantic_begin_id
        self.codebook_size = config.vocab_size
        self.num_codebooks = config.num_codebooks
        self.backbone_codebooks = config.vq_backbone_codebooks
        self.audio_decoder = _FrameEmbeddings(config)
        self.audio_decoder_mtp_shared = _LayerList(
            [TowerLayer(config, "semantic_embedding") for _ in range(config.shared_enrichment_layers)]
        )
        self.audio_decoder_mtp = _ResidualTower(config)
        self.register_buffer("row_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False)
        self.cache = TowerKVCache(
            slots=cache_slots,
            layers=config.shared_enrichment_layers + config.num_codebooks - 1,
            max_len=max_len,
            kv_heads=config.n_local_heads,
            head_dim=config.head_dim,
        )

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (token_ids < self.semantic_begin_id + self.codebook_size)

    def code_embedding_sum(self, codes: torch.Tensor, first: int, end: int) -> torch.Tensor:
        """Sum of codebook embeddings of codes[:, first:end] (codebook-offset rows)."""
        rows = codes[:, first:end].clamp(0, self.codebook_size - 1) + self.row_offsets[first:end]
        return F.embedding_bag(rows, self.audio_decoder.codebook_embeddings.weight, mode="sum")

    def backbone_embedding(self, token_embeds: torch.Tensor, codes: torch.Tensor, is_semantic: torch.Tensor):
        """Backbone input: for audio frames, the token embedding plus the embeddings of
        the first backbone_codebooks residual codes, rescaled."""
        extra = self.code_embedding_sum(codes, 1, 1 + self.backbone_codebooks).to(token_embeds.dtype)
        mixed = (token_embeds + extra) * (1.0 / math.sqrt(1.0 + self.backbone_codebooks))
        return torch.where(is_semantic.unsqueeze(-1), mixed, token_embeds)

    def _shared_input(self, input_codes: torch.Tensor, input_is_semantic: torch.Tensor) -> torch.Tensor:
        extra = self.code_embedding_sum(input_codes, 1 + self.backbone_codebooks, self.num_codebooks)
        return torch.where(input_is_semantic.unsqueeze(-1), extra, 0.0)

    def run(
        self,
        hidden: torch.Tensor,
        input_codes: torch.Tensor,
        input_is_semantic: torch.Tensor,
        next_main_tokens: torch.Tensor,
        next_codes: torch.Tensor | None,
        batch: TowerBatch,
        sampling: tuple[float, int, float],
    ) -> torch.Tensor:
        """Run all tower layers. With ``next_codes`` (prefill) the codes are teacher
        forced; without it (decode) codes 1.. are sampled. Returns next codes [rows, Q]."""
        self.cache.ensure(hidden.device, hidden.dtype)
        keys, values = self.cache.keys, self.cache.values
        label_is_semantic = self.is_semantic(next_main_tokens)
        code0 = (next_main_tokens - self.semantic_begin_id).clamp(0, self.codebook_size - 1)
        shared_input = self._shared_input(input_codes, input_is_semantic)
        layer_index = 0
        for i, layer in enumerate(self.audio_decoder_mtp_shared.layers):
            hidden = layer(
                hidden,
                code0,
                label_is_semantic,
                shared_input if i == 0 else None,
                batch,
                keys[layer_index],
                values[layer_index],
            )
            layer_index += 1
        codes = [code0]
        for r, (layer, head) in enumerate(zip(self.audio_decoder_mtp.layers, self.audio_decoder_mtp.heads)):
            current = codes[-1] if next_codes is None else next_codes[:, r]
            hidden = layer(hidden, current, label_is_semantic, None, batch, keys[layer_index], values[layer_index])
            layer_index += 1
            if next_codes is None:
                codes.append(sample_top_k_top_p(head(hidden), *sampling))
        if next_codes is not None:
            return next_codes
        return torch.stack(codes, dim=-1)


class TowerDecodeGraphs:
    """CUDA graphs of one tower decode step per batch-size bucket.

    Inputs are copied into static buffers; rows past the batch are padding that
    write to the tower cache's scratch slot, so no live request is touched.
    """

    def __init__(self, tower: CodebookTower, batch_sizes: tuple[int, ...], sampling: tuple[float, int, float]):
        self.tower = tower
        self.batch_sizes = tuple(sorted(set(batch_sizes)))
        self.sampling = sampling
        self._graphs: dict[int, tuple[CUDAGraph, dict[str, torch.Tensor], torch.Tensor]] = {}

    @torch.inference_mode()
    def capture(self, device: torch.device, dtype: torch.dtype) -> None:
        tower = self.tower
        tower.cache.ensure(device, dtype)
        scratch = tower.cache.scratch_slot
        dim = tower.config.dim
        pool = torch.cuda.graph_pool_handle()
        for size in self.batch_sizes:
            buffers = {
                "hidden": torch.zeros(size, dim, device=device, dtype=dtype),
                "input_codes": torch.zeros(size, tower.num_codebooks, device=device, dtype=torch.long),
                "input_is_semantic": torch.zeros(size, device=device, dtype=torch.bool),
                "main": torch.full((size,), tower.semantic_begin_id, device=device, dtype=torch.long),
                "slots": torch.full((size,), scratch, device=device, dtype=torch.long),
                "positions": torch.zeros(size, device=device, dtype=torch.long),
            }

            def step() -> torch.Tensor:
                return tower.run(
                    hidden=buffers["hidden"],
                    input_codes=buffers["input_codes"],
                    input_is_semantic=buffers["input_is_semantic"],
                    next_main_tokens=buffers["main"],
                    next_codes=None,
                    batch=TowerBatch.decode(buffers["slots"], buffers["positions"]),
                    sampling=self.sampling,
                )

            step()
            torch.accelerator.synchronize(device)
            graph = CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                output = step()
            self._graphs[size] = (graph, buffers, output)

    def covers(self, rows: int) -> bool:
        return bool(self._graphs) and rows <= self.batch_sizes[-1]

    @torch.inference_mode()
    def run(
        self,
        hidden: torch.Tensor,
        input_codes: torch.Tensor,
        input_is_semantic: torch.Tensor,
        next_main_tokens: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        rows = hidden.shape[0]
        size = next(s for s in self.batch_sizes if s >= rows)
        graph, buffers, output = self._graphs[size]
        buffers["hidden"][:rows].copy_(hidden)
        buffers["input_codes"][:rows].copy_(input_codes)
        buffers["input_is_semantic"][:rows].copy_(input_is_semantic)
        buffers["main"][:rows].copy_(next_main_tokens)
        buffers["slots"][:rows].copy_(slots)
        buffers["positions"][:rows].copy_(positions)
        if rows < size:
            buffers["slots"][rows:].fill_(self.tower.cache.scratch_slot)
            buffers["positions"][rows:].zero_()
        graph.replay()
        return output[:rows].clone()
