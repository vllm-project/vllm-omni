# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched streaming Whisper encoder and CUDA graphs for MiniCPM-o duplex sessions."""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.cuda import CUDAGraph

#: Growth step of a session buffer, in encoder positions (a 1 s unit is 50):
#: at most five reallocations per 1500-position cycle, ~12 MiB average slack.
DEFAULT_KV_PAGE_POSITIONS = 250

_SUPPORTED_ATTENTION = ("sdpa", "eager")
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


@dataclass(eq=False, slots=True)
class StreamingAudioKVCache:
    """One session's streaming-encoder self-attention cache, written in place.

    Layout ``[layers, 2 (key/value), capacity, embed_dim]``: token-major, so appending a unit is one contiguous
    copy per layer and attention reads a strided ``[1, heads, length, head_dim]`` view without materializing it.
    Capacity grows in ``page_positions`` steps up to ``max_positions``.
    """

    num_layers: int
    embed_dim: int
    num_heads: int
    max_positions: int
    page_positions: int = DEFAULT_KV_PAGE_POSITIONS
    length: int = field(default=0, init=False)
    _buffer: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.num_layers, self.embed_dim, self.num_heads, self.max_positions, self.page_positions) <= 0:
            raise ValueError("streaming audio KV cache dimensions must be positive")

    @classmethod
    def for_encoder(cls, encoder: nn.Module, page_positions: int) -> StreamingAudioKVCache:
        config, positions = encoder.config, int(encoder.embed_positions.weight.shape[0])
        return cls(
            len(encoder.layers), int(config.d_model), int(config.encoder_attention_heads), positions, page_positions
        )

    @property
    def capacity(self) -> int:
        return 0 if self._buffer is None else int(self._buffer.shape[2])

    def reserve(self, positions: int, *, dtype: torch.dtype, device: torch.device) -> None:
        """Make room for ``positions`` entries, keeping the committed prefix."""
        if positions <= self.capacity:
            return
        pages = math.ceil(positions / self.page_positions) * self.page_positions
        capacity = max(positions, min(self.max_positions, pages))
        buffer = torch.empty((self.num_layers, 2, capacity, self.embed_dim), dtype=dtype, device=device)
        if self._buffer is not None and self.length:
            buffer[:, :, : self.length].copy_(self._buffer[:, :, : self.length])
        self._buffer = buffer

    def history(self, length: int) -> torch.Tensor:
        """``[layers, 2, length, embed_dim]`` rows ``[0, length)``, every layer at once."""
        assert self._buffer is not None
        return self._buffer[:, :, :length]

    def commit(self, offset: int, new_kv: torch.Tensor) -> None:
        """Write ``[layers, 2, positions, embed_dim]`` rows at ``offset``; the caller advances ``length``."""
        assert self._buffer is not None
        positions = int(new_kv.shape[2])
        self._buffer[:, :, offset : offset + positions].copy_(new_kv)

    def layer_views(self, start: int, end: int) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Per layer, ``(key_rows, value_rows, key_heads, value_heads)``: the ``[end - start, embed_dim]``
        write targets and the ``[1, heads, end, head_dim]`` attention inputs over ``[0, end)``."""
        assert self._buffer is not None
        head_dim = self.embed_dim // self.num_heads
        heads = (
            self._buffer[:, :, :end]
            .view(self.num_layers, 2, end, self.num_heads, head_dim)
            .transpose(2, 3)
            .unsqueeze(2)
        )
        rows = self._buffer[:, :, start:end]
        return rows[:, 0].unbind(0), rows[:, 1].unbind(0), heads[:, 0].unbind(0), heads[:, 1].unbind(0)

    def to_legacy_cache(self):
        """The committed history as the ``EncoderDecoderCache`` the per-session path grows."""
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_attention = DynamicCache()
        if self.length:
            _, _, keys, values = self.layer_views(0, self.length)
            for layer in range(self.num_layers):
                self_attention.update(keys[layer].contiguous(), values[layer].contiguous(), layer)
        return EncoderDecoderCache(self_attention, DynamicCache())


@dataclass(frozen=True, slots=True)
class StreamingAudioChunk:
    """One session's ready unit: log-mel frames plus the state it continues."""

    #: ``[n_mels, frames]`` or ``[1, n_mels, frames]`` log-mel features, host or device.
    features: torch.Tensor
    #: Session cache; ``None`` starts a new history.
    cache: StreamingAudioKVCache | None
    prefix_extra_frames: int = 0
    suffix_extra_frames: int = 0
    use_extra_context: bool = True
    #: ``audio_feature_lens`` of the unit when it differs from ``features.shape[-1]``.
    feature_length: int | None = None


@dataclass(slots=True)
class _Row:
    index: int
    frames: int
    start: int
    length: int
    pooled_length: int
    cache: StreamingAudioKVCache
    past: int = 0
    offset: int = 0

    @property
    def total(self) -> int:
        return self.past + self.length


def _extra_context_trim(frames: int) -> int:
    return (int(frames) + 1) // 2 if frames > 0 else 0


def streaming_batch_unsupported_reason(encoder: nn.Module) -> str | None:
    """Why ``encoder`` cannot take the batched path, or ``None`` when it can."""
    if encoder.training:
        return "training mode (dropout / layerdrop)"
    weight = getattr(getattr(encoder, "conv1", None), "weight", None)
    if weight is None or weight.dtype not in _SUPPORTED_DTYPES:
        # float16 needs the per-session inf/nan clamp, which is per tensor.
        return f"unsupported encoder dtype {getattr(weight, 'dtype', None)}"
    layers = getattr(encoder, "layers", None)
    if not layers:
        return "encoder has no layers"
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None or not all(
            hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj", "scaling")
        ):
            return "encoder layers are not Whisper self-attention layers"
    implementation = getattr(encoder.config, "_attn_implementation", None) or "eager"
    if implementation not in _SUPPORTED_ATTENTION:
        return f"attention implementation {implementation!r} is not batched"
    return None


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, implementation: str):
    """Whisper self-attention as the per-session path calls it; ``query`` already carries ``scaling``."""
    if implementation == "sdpa":
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, scale=1.0
        )
    weights = torch.matmul(query, key.transpose(2, 3)) + mask
    return torch.matmul(torch.nn.functional.softmax(weights, dim=-1), value)


def _stack_features(rows: list[_Row], features: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Zero-padded ``[B, mels, max_frames]`` on ``device`` with one host-to-device copy."""
    max_frames = max(row.frames for row in rows)
    on_host = all(feature.device.type == "cpu" for feature in features)
    pin = on_host and device.type == "cuda"
    batch = torch.zeros(
        (len(rows), int(features[0].shape[0]), max_frames),
        dtype=features[0].dtype,
        device="cpu" if on_host else device,
        pin_memory=pin,
    )
    for slot, (row, feature) in enumerate(zip(rows, features, strict=True)):
        batch[slot, :, : row.frames].copy_(feature)
    # Pinned staging keeps the host from waiting on queued GPU work; the
    # caching host allocator holds the buffer until the copy completes.
    return batch.to(device, non_blocking=pin)


def _position_rows(encoder: nn.Module, rows: list[_Row]) -> torch.Tensor:
    """Learned positions ``past .. past + n`` per row; past the table, repeat its last row."""
    table = encoder.embed_positions.weight
    capacity = int(table.shape[0])
    pieces: list[torch.Tensor] = []
    for row in rows:
        end = row.past + row.length
        if end <= capacity:
            pieces.append(table[row.past : end])
        else:
            pieces.append(table[row.past :])
            pieces.append(table[-1:].expand(end - max(row.past, capacity), -1))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


@torch.inference_mode()
def encode_streaming_audio_batch(
    encoder: nn.Module,
    projection: nn.Module,
    pooler: nn.Module,
    chunks: Sequence[StreamingAudioChunk],
    *,
    pool_step: int,
    page_positions: int = DEFAULT_KV_PAGE_POSITIONS,
) -> tuple[list[torch.Tensor | None], list[StreamingAudioKVCache | None]]:
    """Encode one streaming unit per session in one encoder pass."""
    implementation = getattr(encoder.config, "_attn_implementation", None) or "eager"
    weight = encoder.conv1.weight
    dtype, device = weight.dtype, weight.device
    max_positions = int(encoder.embed_positions.weight.shape[0])
    embed_dim, num_heads = int(encoder.config.d_model), int(encoder.config.encoder_attention_heads)

    outputs: list[torch.Tensor | None] = [None] * len(chunks)
    caches: list[StreamingAudioKVCache | None] = [chunk.cache for chunk in chunks]
    rows: list[_Row] = []
    features: list[torch.Tensor] = []
    for index, chunk in enumerate(chunks):
        feature = chunk.features
        if feature.ndim == 3:
            if feature.shape[0] != 1:
                raise ValueError("a streaming chunk carries exactly one audio")
            feature = feature[0]
        frames = int(feature.shape[-1])
        conv_length = (frames - 1) // 2 + 1
        prefix = _extra_context_trim(chunk.prefix_extra_frames) if chunk.use_extra_context else 0
        suffix = _extra_context_trim(chunk.suffix_extra_frames) if chunk.use_extra_context else 0
        length = conv_length - prefix - suffix
        if frames <= 0 or length <= 0:
            continue
        feature_length = frames if chunk.feature_length is None else int(chunk.feature_length)
        pooled_length = ((feature_length - 1) // 2 + 1 - pool_step) // pool_step + 1
        cache = chunk.cache
        if cache is None or cache.length + length >= max_positions:
            cache = StreamingAudioKVCache.for_encoder(encoder, page_positions)
        rows.append(_Row(index, frames, prefix, length, pooled_length, cache, past=cache.length))
        features.append(feature)
    if not rows:
        return outputs, caches

    offset = 0
    for row in rows:
        row.offset = offset
        offset += row.length
        row.cache.reserve(row.total, dtype=dtype, device=device)

    # CNN front end on the zero-padded batch, transformer layers on the packed rows.
    mel = _stack_features(rows, features, device).to(dtype=dtype)
    hidden = nn.functional.gelu(encoder.conv1(mel))
    max_frames = int(mel.shape[-1])
    for slot, row in enumerate(rows):
        if row.frames < max_frames:
            # conv2 must see zeros past each row's end, as its own padding would in the per-session call.
            hidden[slot, :, row.frames :].zero_()
    hidden = nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
    pieces = [hidden[slot, row.start : row.start + row.length] for slot, row in enumerate(rows)]
    hidden = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
    hidden = hidden + _position_rows(encoder, rows)

    # Every per-row view is built once here, not per layer: at batch sizes
    # that matter the host cost of view ops, not the GPU, bounds this loop.
    views = [row.cache.layer_views(row.past, row.total) for row in rows]
    lengths = [row.length for row in rows]
    total_rows = sum(lengths)
    head_dim = embed_dim // num_heads
    # The per-session path passes an all-zero additive mask; keep it for the same attention kernel.
    masks = [torch.zeros((1, 1, row.length, row.total), dtype=dtype, device=device) for row in rows]
    for layer_index, layer in enumerate(encoder.layers):
        attention = layer.self_attn
        residual = hidden
        normed = layer.self_attn_layer_norm(hidden)
        query = attention.q_proj(normed) * attention.scaling
        key, value = attention.k_proj(normed), attention.v_proj(normed)
        destinations = [view[0][layer_index] for view in views] + [view[1][layer_index] for view in views]
        torch._foreach_copy_(destinations, [*key.split(lengths), *value.split(lengths)])
        queries = query.view(1, total_rows, num_heads, head_dim).transpose(1, 2).split(lengths, dim=2)
        attended = [
            _attend(row_query, view[2][layer_index], view[3][layer_index], mask, implementation)
            for row_query, view, mask in zip(queries, views, masks, strict=True)
        ]
        # [1, heads, rows, head_dim] per row -> packed [rows, embed_dim].
        attended_rows = (attended[0] if len(attended) == 1 else torch.cat(attended, dim=2)).transpose(1, 2)
        hidden = residual + attention.out_proj(attended_rows.reshape(total_rows, embed_dim))
        residual = hidden
        hidden = layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden)))
        hidden = residual + layer.fc2(hidden)
    hidden = encoder.layer_norm(hidden)
    embeds = projection(hidden)
    if all(row.length % pool_step == 0 for row in rows):
        pooled = pooler(embeds.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        for row in rows:
            start = row.offset // pool_step
            count = min(row.length // pool_step, row.pooled_length)
            outputs[row.index] = pooled[start : start + count]
    else:
        for row in rows:
            segment = embeds[row.offset : row.offset + row.length].transpose(0, 1).unsqueeze(0)
            pooled = pooler(segment).squeeze(0).transpose(0, 1)
            outputs[row.index] = pooled[: row.pooled_length]

    for row in rows:
        row.cache.length = row.total
        caches[row.index] = row.cache
    return outputs, caches


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
