# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs of MiniCPM-o's streaming Whisper encoder for the steady unit shape."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from .streaming_audio_encoder import (
    DEFAULT_KV_PAGE_POSITIONS,
    StreamingAudioChunk,
    StreamingAudioKVCache,
    _attend,
    _extra_context_trim,
    encode_streaming_audio_batch,
)

if TYPE_CHECKING:
    from torch.cuda import CUDAGraph

_NEG_INF = float("-inf")

# stage0.py builds every unit but a session's first with these trims.
STEADY_PREFIX_EXTRA_FRAMES = 2
STEADY_SUFFIX_EXTRA_FRAMES = 2

DEFAULT_GRAPH_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
# Shared storage scales with the largest bucket only. 900 stays under ~3 GiB
# at batch 32 on MiniCPM-o 4.5 (bf16); history past it falls back to eager.
DEFAULT_GRAPH_CACHE_BUCKETS: tuple[int, ...] = (250, 500, 750, 900)


def batch_sizes_for_sessions(max_sessions: int) -> tuple[int, ...]:
    """Powers of two below ``max_sessions``, plus ``max_sessions`` itself."""
    top = max(1, int(max_sessions))
    sizes = [1 << i for i in range(top.bit_length()) if (1 << i) < top]
    return normalize_buckets([*sizes, top])


def select_bucket(value: int, buckets: Sequence[int]) -> int | None:
    return min((bucket for bucket in buckets if value <= bucket), default=None)


def normalize_buckets(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values if int(v) > 0}))


def steady_unit_length(unit_frames: int) -> int:
    conv_length = (int(unit_frames) - 1) // 2 + 1
    trim = _extra_context_trim(STEADY_PREFIX_EXTRA_FRAMES) + _extra_context_trim(STEADY_SUFFIX_EXTRA_FRAMES)
    return conv_length - trim


def row_is_steady(chunk: StreamingAudioChunk, *, expected_frames: int) -> bool:
    frames = int(chunk.features.shape[-1])
    return (
        chunk.cache is not None
        and chunk.use_extra_context
        and (int(chunk.prefix_extra_frames), int(chunk.suffix_extra_frames))
        == (STEADY_PREFIX_EXTRA_FRAMES, STEADY_SUFFIX_EXTRA_FRAMES)
        and frames == expected_frames
        and (chunk.feature_length is None or int(chunk.feature_length) == frames)
    )


@dataclass(slots=True)
class _Group:
    index: int
    cache_in: StreamingAudioKVCache
    cache_out: StreamingAudioKVCache
    past: int


class StreamingAudioGraphEncoder:
    """Pre-captured CUDA graphs of the streaming encoder's steady unit shape."""

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        pooler: nn.Module,
        *,
        unit_frames: int,
        pool_step: int,
        batch_sizes: Sequence[int] | None = None,
        cache_buckets: Sequence[int] | None = None,
        page_positions: int = DEFAULT_KV_PAGE_POSITIONS,
        pinned_h2d: bool = False,
    ) -> None:
        weight, config = encoder.conv1.weight, encoder.config
        if weight.device.type != "cuda":
            raise ValueError("StreamingAudioGraphEncoder requires a CUDA encoder")
        self.encoder, self.projection, self.pooler = encoder, projection, pooler
        self.device, self.dtype = weight.device, weight.dtype
        self.pool_step, self.page_positions, self.pinned_h2d = int(pool_step), int(page_positions), bool(pinned_h2d)
        self.embed_dim, self.num_heads = int(config.d_model), int(config.encoder_attention_heads)
        self.max_positions = int(encoder.embed_positions.weight.shape[0])
        self.implementation = getattr(config, "_attn_implementation", None) or "eager"
        self.unit_frames = int(unit_frames)
        self.unit_length = steady_unit_length(self.unit_frames)
        nominal_pooled = ((self.unit_frames - 1) // 2 + 1 - self.pool_step) // self.pool_step + 1
        self.pooled_length = min(nominal_pooled, self.unit_length // self.pool_step)
        if self.unit_length <= 0 or self.pooled_length <= 0:
            raise ValueError(f"unit_frames={unit_frames} pool_step={pool_step} leaves no steady unit output")
        self.batch_sizes = normalize_buckets(batch_sizes or DEFAULT_GRAPH_BATCH_SIZES)
        self.cache_buckets = normalize_buckets(cache_buckets or DEFAULT_GRAPH_CACHE_BUCKETS)
        if not self.batch_sizes or not self.cache_buckets:
            raise ValueError("StreamingAudioGraphEncoder needs at least one batch size and one cache bucket")
        self.max_batch, self.max_cache_bucket = max(self.batch_sizes), max(self.cache_buckets)
        # (batch, cache length) -> (graph, pooled output); every graph reads prefix views of one storage.
        self._graphs: dict[tuple[int, int], tuple[CUDAGraph, torch.Tensor]] = {}

    def _views(self, batch: int, cache_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = cache_len + self.unit_length
        return (
            self._mel[:batch],
            self._cache[:, :, :batch, :total, :],
            self._mask[:batch, :, :, :total],
            self._position_ids[:batch],
        )

    def capture(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot capture the streaming audio encoder graph mid-capture")
        from vllm.platforms import current_platform

        batch, total = self.max_batch, self.max_cache_bucket + self.unit_length
        zeros = functools.partial(torch.zeros, dtype=self.dtype, device=self.device)
        self._mel = zeros((batch, int(self.encoder.config.num_mel_bins), self.unit_frames))
        self._cache = zeros((len(self.encoder.layers), 2, batch, total, self.embed_dim))
        self._mask = zeros((batch, 1, self.unit_length, total))
        self._position_ids = torch.zeros((batch, self.unit_length), dtype=torch.long, device=self.device)
        pool = current_platform.get_global_graph_pool()
        current_stream = torch.cuda.current_stream(self.device)
        for key in itertools.product(self.batch_sizes, self.cache_buckets):
            views = self._views(*key)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.inference_mode():
                for _ in range(3):
                    self._forward(*views, key[1])
            current_stream.wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph, pool=pool):
                self._graphs[key] = (graph, self._forward(*views, key[1]))

    def _forward(
        self,
        mel: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        encoder = self.encoder
        batch = mel.shape[0]
        hidden = nn.functional.gelu(encoder.conv1(mel))
        hidden = nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
        prefix = _extra_context_trim(STEADY_PREFIX_EXTRA_FRAMES)
        hidden = hidden[:, prefix : prefix + self.unit_length, :]
        hidden = hidden + encoder.embed_positions(position_ids)
        head_dim = self.embed_dim // self.num_heads
        for layer_index, layer in enumerate(encoder.layers):
            attention = layer.self_attn
            residual = hidden
            normed = layer.self_attn_layer_norm(hidden)
            query = attention.q_proj(normed) * attention.scaling
            key, value = attention.k_proj(normed), attention.v_proj(normed)
            cache[layer_index, 0, :, cache_len : cache_len + self.unit_length, :].copy_(key)
            cache[layer_index, 1, :, cache_len : cache_len + self.unit_length, :].copy_(value)
            q = query.view(batch, self.unit_length, self.num_heads, head_dim).transpose(1, 2)
            k_all = cache[layer_index, 0].view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
            v_all = cache[layer_index, 1].view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
            attended = _attend(q, k_all, v_all, mask, self.implementation)
            attended = attended.transpose(1, 2).reshape(batch, self.unit_length, self.embed_dim)
            hidden = residual + attention.out_proj(attended)
            residual = hidden
            hidden = layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden)))
            hidden = residual + layer.fc2(hidden)
        hidden = encoder.layer_norm(hidden)
        embeds = self.projection(hidden)
        return self.pooler(embeds.transpose(1, 2)).transpose(1, 2)

    def encode(
        self,
        chunks: Sequence[StreamingAudioChunk],
    ) -> tuple[list[torch.Tensor | None], list[StreamingAudioKVCache | None]]:
        outputs: list[torch.Tensor | None] = [None] * len(chunks)
        caches: list[StreamingAudioKVCache | None] = [chunk.cache for chunk in chunks]
        steady: list[_Group] = []
        eager: list[int] = []
        for index, chunk in enumerate(chunks):
            cache_in = chunk.cache
            reset = cache_in is not None and cache_in.length + self.unit_length >= self.max_positions
            past = 0 if reset or cache_in is None else cache_in.length
            if not row_is_steady(chunk, expected_frames=self.unit_frames) or past > self.max_cache_bucket:
                eager.append(index)
                continue
            cache_out = StreamingAudioKVCache.for_encoder(self.encoder, self.page_positions) if reset else cache_in
            steady.append(_Group(index, cache_in, cache_out, past))
        for start in range(0, len(steady), self.max_batch):
            self._replay(steady[start : start + self.max_batch], chunks, outputs, caches)
        if eager:
            eager_outputs, eager_caches = encode_streaming_audio_batch(
                self.encoder,
                self.projection,
                self.pooler,
                [chunks[i] for i in eager],
                pool_step=self.pool_step,
                page_positions=self.page_positions,
            )
            for index, output, cache in zip(eager, eager_outputs, eager_caches, strict=True):
                outputs[index], caches[index] = output, cache
        return outputs, caches

    @torch.inference_mode()
    def _replay(
        self,
        group: list[_Group],
        chunks: Sequence[StreamingAudioChunk],
        outputs: list[torch.Tensor | None],
        caches: list[StreamingAudioKVCache | None],
    ) -> None:
        batch = select_bucket(len(group), self.batch_sizes)
        cache_len = select_bucket(max((row.past for row in group), default=0), self.cache_buckets)
        assert batch is not None and cache_len is not None
        mel, cache, mask, position_ids = self._views(batch, cache_len)
        mask.fill_(0.0)
        if len(group) < batch:
            mask[len(group) :, :, :, :cache_len].fill_(_NEG_INF)
            position_ids[len(group) :].copy_(torch.arange(self.unit_length, device=self.device))
        for slot, row in enumerate(group):
            feature = chunks[row.index].features
            feature = (feature[0] if feature.ndim == 3 else feature).to(dtype=self.dtype)
            if self.pinned_h2d and feature.device.type == "cpu":
                mel[slot].copy_(feature.pin_memory(), non_blocking=True)
            else:
                mel[slot].copy_(feature.to(device=self.device))
            pad = cache_len - row.past
            if pad:
                mask[slot, :, :, :pad].fill_(_NEG_INF)
            if row.past:
                cache[:, :, slot, pad:cache_len, :].copy_(row.cache_in.history(row.past))
            position_ids[slot].copy_(torch.arange(row.past, row.past + self.unit_length, device=self.device))
        graph, pooled = self._graphs[(batch, cache_len)]
        graph.replay()
        for slot, row in enumerate(group):
            outputs[row.index] = pooled[slot, : self.pooled_length].clone()
            row.cache_out.reserve(row.past + self.unit_length, dtype=self.dtype, device=self.device)
            row.cache_out.commit(row.past, cache[:, :, slot, cache_len : cache_len + self.unit_length, :])
            row.cache_out.length = row.past + self.unit_length
            caches[row.index] = row.cache_out
