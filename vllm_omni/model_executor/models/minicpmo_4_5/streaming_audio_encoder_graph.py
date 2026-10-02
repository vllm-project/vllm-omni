# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for MiniCPM-o's streaming Whisper encoder, for the steady unit shape.

``encode_streaming_audio_batch`` still launches one attention kernel per row
per layer, so launches scale with the batch; a CUDA graph replays the whole
forward with one launch.

A Stage-0 duplex *steady* unit (every unit but a session's first) always has
``prefix_extra_frames=2``, ``suffix_extra_frames=2`` and one processor chunk of
frames, so its post-trim length ``unit_length`` is constant. Only the batch size
and each row's committed history vary, so graphs are captured per
``(batch bucket, cache-length bucket)``:

* padding rows replay stale buffer content, but their output is never read and
  rows never attend across the batch;
* a row's history is copied in right-aligned against the bucket
  (``bucket - past .. bucket``) and the unused prefix is masked to ``-inf``;
  the new unit is always written at ``bucket .. bucket + unit_length``.

Real positions are placed and masked exactly as the eager batch places them.
Rows this module cannot describe (a first unit, a short final unit, history
past the largest bucket) go to the eager batch.

Only one graph replays at a time and every replay refills its inputs, so all
graphs take prefix views of one shared storage per static input, sized for the
largest bucket (like ``WholeEulerExecutionArena``). Graphs are captured once, at
load (``build_streaming_audio_graph_encoder``), never while serving.

Resident mode (``resident_slots > 0``, default off): instead of a shared
``[layers, 2, max_batch, bucket + unit, d]`` history storage that every replay
refills from the session caches (and copies the new unit back out of), every
session's cache lives in one slot of a :class:`StreamingAudioKVSlotPool` and
the graph reads its history from, and writes its new unit into, that slot.
Inside the graph each layer gathers the history into a temporary with the
shared storage's exact layout (right-aligned, same strides), so attention sees
the same values at every unmasked position as the copy-in path. The pool
replaces both the shared storage and the per-session paged buffers.
"""

from __future__ import annotations

import logging
import weakref
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

logger = logging.getLogger(__name__)

#: ``stage0.py`` builds every unit but a session's first with these trims
#: (``("prefix_extra_frames", 0 if state.audio_chunk_idx == 0 else 2)``,
#: ``("suffix_extra_frames", 2)``); only that steady shape is captured.
STEADY_PREFIX_EXTRA_FRAMES = 2
STEADY_SUFFIX_EXTRA_FRAMES = 2

DEFAULT_GRAPH_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
#: Shared storage scales with the *largest* bucket only. 900 keeps it under
#: 3 GiB at batch 32 on MiniCPM-o 4.5's encoder (bf16, measured 2.8-2.9 GiB);
#: history past it (the reset bound is ~1450) falls back to eager.
DEFAULT_GRAPH_CACHE_BUCKETS: tuple[int, ...] = (250, 500, 750, 900)


def batch_sizes_for_sessions(max_sessions: int) -> tuple[int, ...]:
    """Powers of two below ``max_sessions`` plus ``max_sessions`` itself.

    Each live session contributes at most one steady row per step, so a larger
    bucket is never selected; 16 -> (1, 2, 4, 8, 16), 20 -> (1, 2, 4, 8, 16, 20).
    """
    top = max(1, int(max_sessions))
    sizes = [1 << i for i in range(top.bit_length()) if (1 << i) < top]
    return normalize_buckets([*sizes, top])


def select_bucket(value: int, buckets: Sequence[int]) -> int | None:
    """The smallest bucket ``>= value``, or ``None`` past the largest one."""
    return min((bucket for bucket in buckets if value <= bucket), default=None)


def normalize_buckets(values: Sequence[int]) -> tuple[int, ...]:
    """Sorted, deduplicated, positive bucket values."""
    return tuple(sorted({int(v) for v in values if int(v) > 0}))


def steady_unit_length(unit_frames: int) -> int:
    """Post-trim encoder positions of one steady unit of ``unit_frames`` mel frames."""
    conv_length = (int(unit_frames) - 1) // 2 + 1
    trim = _extra_context_trim(STEADY_PREFIX_EXTRA_FRAMES) + _extra_context_trim(STEADY_SUFFIX_EXTRA_FRAMES)
    return conv_length - trim


def steady_pooled_length(unit_frames: int, unit_length: int, pool_step: int) -> int:
    """The pooled length ``encode_streaming_audio_batch`` emits for a steady row.

    The smaller of its ``pooled_length`` cap (from the raw frame count) and the
    pooled post-trim length, like its ``pooled[: row.pooled_length]``.
    """
    nominal_cap = ((unit_frames - 1) // 2 + 1 - pool_step) // pool_step + 1
    return min(nominal_cap, unit_length // pool_step)


def row_is_steady(chunk: StreamingAudioChunk, *, expected_frames: int) -> bool:
    """Whether ``chunk`` is a steady unit of ``expected_frames`` frames: the captured shape."""
    if chunk.cache is None or not chunk.use_extra_context:
        return False
    if int(chunk.prefix_extra_frames) != STEADY_PREFIX_EXTRA_FRAMES:
        return False
    if int(chunk.suffix_extra_frames) != STEADY_SUFFIX_EXTRA_FRAMES:
        return False
    feature = chunk.features
    frames = int(feature.shape[-1])
    if frames != expected_frames:
        return False
    return chunk.feature_length is None or int(chunk.feature_length) == frames


class PooledStreamingAudioKVCache(StreamingAudioKVCache):
    """A :class:`StreamingAudioKVCache` whose buffer is one slot of a :class:`StreamingAudioKVSlotPool`.

    Capacity is the full ``max_positions`` from the start (``reserve`` never
    reallocates); the slot goes back to the pool when this object is collected,
    unless :meth:`StreamingAudioKVSlotPool.take_over` handed it to a successor.
    """

    __slots__ = ("__weakref__", "_finalizer", "pool", "slot")

    def __init__(self, pool: StreamingAudioKVSlotPool, slot: int) -> None:
        super().__init__(
            num_layers=pool.num_layers,
            embed_dim=pool.embed_dim,
            num_heads=pool.num_heads,
            max_positions=pool.max_positions,
            page_positions=pool.max_positions,
        )
        self.pool = pool
        self.slot = int(slot)
        self._buffer = pool.storage[self.slot, :, :, : pool.max_positions]
        self._finalizer = weakref.finalize(self, pool._release, self.slot)


class StreamingAudioKVSlotPool:
    """Fixed homes for the streaming caches of up to ``num_slots`` sessions.

    ``storage`` is ``[num_slots, layers, 2, max_positions + tail, d]``: positions
    ``[0, max_positions)`` of a slot are its session's history (a
    :class:`PooledStreamingAudioKVCache` view); the ``tail`` positions after them
    are never history and take the resident graph's padding-row writes.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        max_positions: int,
        tail_positions: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if min(num_slots, num_layers, embed_dim, num_heads, max_positions, tail_positions) <= 0:
            raise ValueError("streaming audio KV slot pool dimensions must be positive")
        self.num_slots = int(num_slots)
        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.max_positions = int(max_positions)
        self.tail_positions = int(tail_positions)
        self.storage = torch.zeros(
            (self.num_slots, self.num_layers, 2, self.max_positions + self.tail_positions, self.embed_dim),
            dtype=dtype,
            device=device,
        )
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        self._warned_exhausted = False

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def acquire(self) -> PooledStreamingAudioKVCache | None:
        """An empty cache in a free slot, or ``None`` when every slot is taken."""
        if not self._free:
            if not self._warned_exhausted:
                self._warned_exhausted = True
                logger.warning(
                    "MiniCPM-o resident audio KV pool exhausted (%d slots); further sessions use paged "
                    "caches and the eager batched encoder",
                    self.num_slots,
                )
            return None
        return PooledStreamingAudioKVCache(self, self._free.pop())

    def take_over(self, cache: PooledStreamingAudioKVCache) -> PooledStreamingAudioKVCache:
        """An empty cache in ``cache``'s slot (a reset); ``cache`` no longer frees the slot.

        The slot's memory is reused in place, so unlike a paged reset a forward
        that fails after this leaves ``cache``'s history partly overwritten.
        """
        assert cache.pool is self
        cache._finalizer.detach()
        return PooledStreamingAudioKVCache(self, cache.slot)

    def _release(self, slot: int) -> None:
        self._free.append(slot)


@dataclass(slots=True)
class _Group:
    """One steady row this call will replay through a graph."""

    index: int
    cache_in: StreamingAudioKVCache
    cache_out: StreamingAudioKVCache
    past: int


class StreamingAudioGraphEncoder:
    """Pre-captured CUDA graphs of the streaming encoder's steady unit shape.

    :meth:`capture` builds one graph per ``(batch size, cache bucket)``;
    :meth:`encode` replays them and hands every other row to
    :func:`encode_streaming_audio_batch`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        pooler: nn.Module,
        *,
        unit_frames: int,
        pool_step: int,
        batch_sizes: Sequence[int] = DEFAULT_GRAPH_BATCH_SIZES,
        cache_buckets: Sequence[int] = DEFAULT_GRAPH_CACHE_BUCKETS,
        page_positions: int = DEFAULT_KV_PAGE_POSITIONS,
        pinned_h2d: bool = False,
        resident_slots: int = 0,
    ) -> None:
        weight = encoder.conv1.weight
        if weight.device.type != "cuda":
            raise ValueError("StreamingAudioGraphEncoder requires a CUDA encoder")
        self.encoder = encoder
        self.projection = projection
        self.pooler = pooler
        self.device = weight.device
        self.dtype = weight.dtype
        self.pool_step = int(pool_step)
        self.page_positions = int(page_positions)
        # Stage host mel through pinned memory (see ``_stage_mel``).
        self.pinned_h2d = bool(pinned_h2d)
        self.num_layers = len(encoder.layers)
        self.embed_dim = int(encoder.config.d_model)
        self.num_heads = int(encoder.config.encoder_attention_heads)
        self.num_mels = int(encoder.config.num_mel_bins)
        self.max_positions = int(encoder.embed_positions.weight.shape[0])
        self.implementation = getattr(encoder.config, "_attn_implementation", None) or "eager"
        self.unit_frames = int(unit_frames)
        self.unit_length = steady_unit_length(self.unit_frames)
        if self.unit_length <= 0:
            raise ValueError(f"unit_frames={unit_frames} is too small for the steady prefix/suffix trim")
        self.pooled_length = steady_pooled_length(self.unit_frames, self.unit_length, self.pool_step)
        if self.pooled_length <= 0:
            raise ValueError(
                f"unit_frames={unit_frames} pool_step={pool_step} leaves no pooled output for the steady unit"
            )
        self.batch_sizes = normalize_buckets(batch_sizes)
        self.cache_buckets = normalize_buckets(cache_buckets)
        if not self.batch_sizes or not self.cache_buckets:
            raise ValueError("StreamingAudioGraphEncoder needs at least one batch size and one cache bucket")
        self.max_batch = max(self.batch_sizes)
        self.max_cache_bucket = max(self.cache_buckets)
        self.max_total = self.max_cache_bucket + self.unit_length
        # Resident mode (see the module docstring); 0 keeps the copy-in storage.
        self.resident_slots = max(0, int(resident_slots))

        self._graphs: dict[tuple[int, int], CUDAGraph] = {}
        self._pooled: dict[tuple[int, int], torch.Tensor] = {}
        # Shared static-input storage (see the module docstring), allocated by capture().
        self._mel_storage: torch.Tensor | None = None
        self._cache_storage: torch.Tensor | None = None
        self._mask_storage: torch.Tensor | None = None
        self._position_ids_storage: torch.Tensor | None = None
        # Resident mode: the slot pool and per-row ``[slot, history pad, write offset]``.
        self._pool: StreamingAudioKVSlotPool | None = None
        self._row_meta_storage: torch.Tensor | None = None
        self._iota: torch.Tensor | None = None

    @property
    def resident(self) -> bool:
        return self.resident_slots > 0

    def _allocate_storage(self) -> None:
        if self.resident:
            self._allocate_resident_storage()
            return
        self._mel_storage = torch.zeros(
            (self.max_batch, self.num_mels, self.unit_frames), dtype=self.dtype, device=self.device
        )
        self._cache_storage = torch.zeros(
            (self.num_layers, 2, self.max_batch, self.max_total, self.embed_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._mask_storage = torch.zeros(
            (self.max_batch, 1, self.unit_length, self.max_total), dtype=self.dtype, device=self.device
        )
        self._position_ids_storage = torch.zeros(
            (self.max_batch, self.unit_length), dtype=torch.long, device=self.device
        )

    def _allocate_resident_storage(self) -> None:
        self._mel_storage = torch.zeros(
            (self.max_batch, self.num_mels, self.unit_frames), dtype=self.dtype, device=self.device
        )
        self._pool = StreamingAudioKVSlotPool(
            num_slots=self.resident_slots,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            max_positions=self.max_positions,
            tail_positions=self.unit_length,
            dtype=self.dtype,
            device=self.device,
        )
        self._mask_storage = torch.zeros(
            (self.max_batch, 1, self.unit_length, self.max_total), dtype=self.dtype, device=self.device
        )
        self._position_ids_storage = torch.zeros(
            (self.max_batch, self.unit_length), dtype=torch.long, device=self.device
        )
        meta = torch.zeros((self.max_batch, 3), dtype=torch.long, device=self.device)
        # Until the first replay every row is a padding row: slot 0's tail.
        meta[:, 2] = self.max_positions
        self._row_meta_storage = meta
        self._iota = torch.arange(self.max_cache_bucket, dtype=torch.long, device=self.device)

    def new_cache(self, cache: StreamingAudioKVCache | None) -> StreamingAudioKVCache:
        """The cache a new session or a reset starts from: a pool slot when one is free."""
        pool = self._pool
        if pool is not None:
            if isinstance(cache, PooledStreamingAudioKVCache) and cache.pool is pool:
                return pool.take_over(cache)
            pooled = pool.acquire()
            if pooled is not None:
                return pooled
        return StreamingAudioKVCache(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            max_positions=self.max_positions,
            page_positions=self.page_positions,
        )

    def _views(self, batch: int, cache_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefix views of the shared storage for ``(batch, cache_len)``: the memory its graph captured."""
        assert self._cache_storage is not None
        total = cache_len + self.unit_length
        mel = self._mel_storage[:batch]
        cache = self._cache_storage[:, :, :batch, :total, :]
        mask = self._mask_storage[:batch, :, :, :total]
        position_ids = self._position_ids_storage[:batch]
        return mel, cache, mask, position_ids

    # --- capture -------------------------------------------------------------

    def capture(self) -> None:
        """Capture every ``(batch, cache-length)`` graph. Call once, at load."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot capture the streaming audio encoder graph mid-capture")
        from vllm.platforms import current_platform

        self._allocate_storage()
        pool = current_platform.get_global_graph_pool()
        for batch in self.batch_sizes:
            for cache_len in self.cache_buckets:
                self._capture_one(batch, cache_len, pool)

    def _capture_one(self, batch: int, cache_len: int, pool: object) -> None:
        key = (batch, cache_len)
        if self.resident:
            mel, mask, position_ids, meta = self._resident_views(batch, cache_len)
            self._capture_forward(key, lambda: self._forward_resident(mel, mask, position_ids, meta, cache_len), pool)
            return
        mel, cache, mask, position_ids = self._views(batch, cache_len)

        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                warmup = self._forward(mel, cache, mask, position_ids, cache_len)
        current_stream.wait_stream(warmup_stream)
        del warmup

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=pool):
            pooled = self._forward(mel, cache, mask, position_ids, cache_len)

        self._graphs[key] = graph
        self._pooled[key] = pooled

    def _capture_forward(self, key: tuple[int, int], forward, pool: object) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                warmup = forward()
        current_stream.wait_stream(warmup_stream)
        del warmup

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=pool):
            pooled = forward()

        self._graphs[key] = graph
        self._pooled[key] = pooled

    def _resident_views(
        self, batch: int, cache_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._mask_storage is not None and self._row_meta_storage is not None
        total = cache_len + self.unit_length
        return (
            self._mel_storage[:batch],
            self._mask_storage[:batch, :, :, :total],
            self._position_ids_storage[:batch],
            self._row_meta_storage[:batch],
        )

    def _forward_resident(
        self,
        mel: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
        meta: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        """:meth:`_forward` reading/writing each row's pool slot instead of the shared storage.

        ``meta[:, 0]`` is the row's slot, ``meta[:, 1]`` the history pad
        (``cache_len - past``), ``meta[:, 2]`` the write offset (0, or
        ``max_positions`` for a padding row, whose new unit lands in the tail).
        Per layer the history is gathered into a ``[2, batch, max_total, d]``
        temporary laid out like the shared storage, then the new unit is
        written into the slot.
        """
        assert self._pool is not None and self._iota is not None
        storage = self._pool.storage
        encoder = self.encoder
        batch = mel.shape[0]
        total = cache_len + self.unit_length
        slots = meta[:, 0:1]
        # History index j of a row reads slot position j - pad (pad positions
        # read position 0: finite, masked to -inf like the shared storage's stale prefix).
        history_positions = (self._iota[:cache_len].unsqueeze(0) - meta[:, 1:2]).clamp_(min=0) if cache_len else None
        write_positions = position_ids + meta[:, 2:3]
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
            key = attention.k_proj(normed)
            value = attention.v_proj(normed)
            kv = torch.empty((2, batch, self.max_total, self.embed_dim), dtype=self.dtype, device=self.device)
            if history_positions is not None:
                kv[0, :, :cache_len].copy_(storage[slots, layer_index, 0, history_positions])
                kv[1, :, :cache_len].copy_(storage[slots, layer_index, 1, history_positions])
            kv[0, :, cache_len:total].copy_(key)
            kv[1, :, cache_len:total].copy_(value)
            storage[slots, layer_index, 0, write_positions] = key
            storage[slots, layer_index, 1, write_positions] = value
            q = query.view(batch, self.unit_length, self.num_heads, head_dim).transpose(1, 2)
            k_all = kv[0, :, :total].view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
            v_all = kv[1, :, :total].view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
            attended = _attend(q, k_all, v_all, mask, self.implementation)
            attended = attended.transpose(1, 2).reshape(batch, self.unit_length, self.embed_dim)
            hidden = residual + attention.out_proj(attended)
            residual = hidden
            hidden = layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden)))
            hidden = residual + layer.fc2(hidden)
        hidden = encoder.layer_norm(hidden)
        embeds = self.projection(hidden)
        return self.pooler(embeds.transpose(1, 2)).transpose(1, 2)

    def _forward(
        self,
        mel: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        """``encode_streaming_audio_batch``'s per-row math with equal row shapes, as batched ops.

        Padding rows and history positions are inert only through the
        caller's mask (see the module docstring).
        """
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
            key = attention.k_proj(normed)
            value = attention.v_proj(normed)
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

    # --- replay ----------------------------------------------------------------

    def encode(
        self,
        chunks: Sequence[StreamingAudioChunk],
    ) -> tuple[list[torch.Tensor | None], list[StreamingAudioKVCache | None]]:
        """``encode_streaming_audio_batch``'s contract: covered steady rows replay a graph, the rest run eagerly."""
        outputs: list[torch.Tensor | None] = [None] * len(chunks)
        caches: list[StreamingAudioKVCache | None] = [chunk.cache for chunk in chunks]
        steady: list[_Group] = []
        eager_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            if not row_is_steady(chunk, expected_frames=self.unit_frames):
                eager_indices.append(index)
                continue
            cache_in = chunk.cache
            assert cache_in is not None
            if self.resident and not (
                isinstance(cache_in, PooledStreamingAudioKVCache) and cache_in.pool is self._pool
            ):
                # A paged cache (pool exhausted when it started): eager, in place.
                eager_indices.append(index)
                continue
            reset = cache_in.length + self.unit_length >= self.max_positions
            past = 0 if reset else cache_in.length
            if past > self.max_cache_bucket:
                eager_indices.append(index)
                continue
            # Resident: a reset reuses the row's slot in place (past is 0).
            cache_out = (
                StreamingAudioKVCache(
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    max_positions=self.max_positions,
                    page_positions=self.page_positions,
                )
                if reset and not self.resident
                else cache_in
            )
            steady.append(_Group(index=index, cache_in=cache_in, cache_out=cache_out, past=past))

        for start in range(0, len(steady), self.max_batch):
            group = steady[start : start + self.max_batch]
            if self.resident:
                self._replay_resident(group, chunks, outputs, caches)
            else:
                self._replay(group, chunks, outputs, caches)

        if eager_indices:
            sub_chunks = [chunks[i] for i in eager_indices]
            sub_outputs, sub_caches = encode_streaming_audio_batch(
                self.encoder,
                self.projection,
                self.pooler,
                sub_chunks,
                pool_step=self.pool_step,
                page_positions=self.page_positions,
                new_cache=self.new_cache if self.resident else None,
            )
            for local, index in enumerate(eager_indices):
                outputs[index] = sub_outputs[local]
                caches[index] = sub_caches[local]

        return outputs, caches

    def _stage_mel(self, dst: torch.Tensor, feature: torch.Tensor) -> None:
        """Write one unit's log-mel into its static slot.

        The processor builds the features on the host. A pageable
        ``.to(device)`` is a blocking copy that first waits for all queued work
        on the stream -- under async scheduling, the previous step's forward --
        so the next step's host preparation stalls in the middle. With
        ``pinned_h2d`` the dtype conversion stays on the host exactly as the
        blocking copy does it (an H2D copy converts on the source side), and
        only the transfer becomes a pinned, stream-ordered copy: same values,
        no host wait.
        """
        if self.pinned_h2d and feature.device.type == "cpu":
            dst.copy_(feature.to(dtype=self.dtype).pin_memory(), non_blocking=True)
        else:
            dst.copy_(feature.to(device=self.device, dtype=self.dtype))

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
        key = (batch, cache_len)

        mel, cache, mask, position_ids = self._views(batch, cache_len)

        mask.fill_(0.0)
        # Dummy/padding rows: mask their whole history away so only the
        # (unread) new unit is attendable, which can never be all -inf.
        if len(group) < batch:
            mask[len(group) :, :, :, :cache_len].fill_(_NEG_INF)
            position_ids[len(group) :].copy_(torch.arange(self.unit_length, device=self.device))

        for slot, row in enumerate(group):
            feature = chunks[row.index].features
            if feature.ndim == 3:
                feature = feature[0]
            self._stage_mel(mel[slot], feature)
            pad = cache_len - row.past
            if pad:
                mask[slot, :, :, :pad].fill_(_NEG_INF)
            if row.past:
                cache[:, :, slot, pad:cache_len, :].copy_(row.cache_in.history(row.past))
            position_ids[slot].copy_(torch.arange(row.past, row.past + self.unit_length, device=self.device))

        self._graphs[key].replay()

        pooled = self._pooled[key]
        for slot, row in enumerate(group):
            outputs[row.index] = pooled[slot, : self.pooled_length].clone()
            row.cache_out.reserve(row.past + self.unit_length, dtype=self.dtype, device=self.device)
            row.cache_out.commit(row.past, cache[:, :, slot, cache_len : cache_len + self.unit_length, :])
            row.cache_out.length = row.past + self.unit_length
            caches[row.index] = row.cache_out

    @torch.inference_mode()
    def _replay_resident(
        self,
        group: list[_Group],
        chunks: Sequence[StreamingAudioChunk],
        outputs: list[torch.Tensor | None],
        caches: list[StreamingAudioKVCache | None],
    ) -> None:
        """:meth:`_replay` for resident rows: no history copy-in, no new-unit copy-out."""
        batch = select_bucket(len(group), self.batch_sizes)
        cache_len = select_bucket(max((row.past for row in group), default=0), self.cache_buckets)
        assert batch is not None and cache_len is not None
        key = (batch, cache_len)

        mel, mask, position_ids, meta = self._resident_views(batch, cache_len)
        host_meta = torch.empty((batch, 3), dtype=torch.long, pin_memory=True)
        mask.fill_(0.0)
        if len(group) < batch:
            mask[len(group) :, :, :, :cache_len].fill_(_NEG_INF)
            position_ids[len(group) :].copy_(torch.arange(self.unit_length, device=self.device))
            host_meta[len(group) :, 0] = 0
            host_meta[len(group) :, 1] = cache_len
            host_meta[len(group) :, 2] = self.max_positions

        for slot, row in enumerate(group):
            feature = chunks[row.index].features
            if feature.ndim == 3:
                feature = feature[0]
            self._stage_mel(mel[slot], feature)
            pad = cache_len - row.past
            if pad:
                mask[slot, :, :, :pad].fill_(_NEG_INF)
            position_ids[slot].copy_(torch.arange(row.past, row.past + self.unit_length, device=self.device))
            cache = row.cache_in
            assert isinstance(cache, PooledStreamingAudioKVCache)
            host_meta[slot, 0] = cache.slot
            host_meta[slot, 1] = pad
            host_meta[slot, 2] = 0
        meta.copy_(host_meta, non_blocking=True)

        self._graphs[key].replay()

        pooled = self._pooled[key]
        for slot, row in enumerate(group):
            outputs[row.index] = pooled[slot, : self.pooled_length].clone()
            row.cache_out.length = row.past + self.unit_length
            caches[row.index] = row.cache_out


__all__ = [
    "DEFAULT_GRAPH_BATCH_SIZES",
    "DEFAULT_GRAPH_CACHE_BUCKETS",
    "PooledStreamingAudioKVCache",
    "STEADY_PREFIX_EXTRA_FRAMES",
    "STEADY_SUFFIX_EXTRA_FRAMES",
    "StreamingAudioGraphEncoder",
    "StreamingAudioKVSlotPool",
    "normalize_buckets",
    "row_is_steady",
    "select_bucket",
    "steady_pooled_length",
    "steady_unit_length",
]
