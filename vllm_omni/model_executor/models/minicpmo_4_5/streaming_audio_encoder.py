# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One forward of MiniCPM-o's streaming Whisper encoder for many duplex sessions.

The per-session path (``get_audio_embedding_streaming``) runs the encoder at
batch one against an ``EncoderDecoderCache`` that ``torch.cat``-grows every
layer on each one-second unit. This module encodes every session's ready unit
in one pass:

* convolutions run on a zero-padded ``[B, mels, frames]`` batch, everything
  after them on the packed ``[sum(n_b), d]`` rows, so padding is never attended;
* attention runs per row against that row's own cache, with the per-session
  shapes, mask and kernel;
* each session owns a :class:`StreamingAudioKVCache` written in place. Its
  ``length`` only advances after the whole forward, and a history at the
  position bound is replaced rather than cleared, so a failed forward leaves
  committed state untouched.

Results match the per-session path up to floating-point reassociation (not
bit-exact). :func:`streaming_batch_unsupported_reason` names configurations
this does not handle, which keep the per-session path.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

#: Growth step of a session buffer, in encoder positions (a 1 s unit is 50):
#: at most five reallocations per 1500-position cycle, ~12 MiB average slack.
DEFAULT_KV_PAGE_POSITIONS = 250

_SUPPORTED_ATTENTION = ("sdpa", "eager")
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


class StreamingAudioKVCache:
    """One session's streaming-encoder self-attention cache, written in place.

    Layout ``[layers, 2 (key/value), capacity, embed_dim]``: token-major, so
    appending a unit is one contiguous copy per layer and attention reads a
    strided ``[1, heads, length, head_dim]`` view without materializing it.
    Capacity grows in ``page_positions`` steps up to ``max_positions``.
    """

    __slots__ = ("_buffer", "embed_dim", "length", "max_positions", "num_heads", "num_layers", "page_positions")

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        max_positions: int,
        page_positions: int = DEFAULT_KV_PAGE_POSITIONS,
    ) -> None:
        if min(num_layers, embed_dim, num_heads, max_positions, page_positions) <= 0:
            raise ValueError("streaming audio KV cache dimensions must be positive")
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_positions = max_positions
        self.page_positions = page_positions
        self.length = 0
        self._buffer: torch.Tensor | None = None

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

    def keys(self, layer: int) -> torch.Tensor:
        """``[capacity, embed_dim]`` key rows of ``layer``."""
        assert self._buffer is not None
        return self._buffer[layer, 0]

    def values(self, layer: int) -> torch.Tensor:
        """``[capacity, embed_dim]`` value rows of ``layer``."""
        assert self._buffer is not None
        return self._buffer[layer, 1]

    def head_view(self, rows: torch.Tensor) -> torch.Tensor:
        """``[positions, embed_dim]`` rows as a ``[1, heads, positions, head_dim]`` view."""
        positions = int(rows.shape[0])
        return rows.view(positions, self.num_heads, self.embed_dim // self.num_heads).transpose(0, 1).unsqueeze(0)

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
        """Per-layer views of positions ``[start, end)``, built with a few ops for all layers.

        Returns ``(key_rows, value_rows, key_heads, value_heads)``: per layer the
        ``[end - start, embed_dim]`` rows (write targets) and the
        ``[1, heads, end, head_dim]`` attention inputs over ``[0, end)``.
        """
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
            for layer in range(self.num_layers):
                self_attention.update(
                    self.head_view(self.keys(layer)[: self.length]).contiguous(),
                    self.head_view(self.values(layer)[: self.length]).contiguous(),
                    layer,
                )
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


def streaming_batch_unsupported_reason(encoder: nn.Module, *, audio_encoder_layer: int | None) -> str | None:
    """Why ``encoder`` cannot take the batched path, or ``None`` when it can."""
    if audio_encoder_layer != -1:
        return "only the final encoder layer output is batched"
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


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    implementation: str,
) -> torch.Tensor:
    """Whisper self-attention of one row, exactly as the per-session path calls it.

    The query already carries ``scaling`` (Whisper scales after ``q_proj``), so
    the kernel runs with ``scale=1``. Returns ``[1, heads, positions, head_dim]``.
    """
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
    new_cache: Callable[[StreamingAudioKVCache | None], StreamingAudioKVCache] | None = None,
) -> tuple[list[torch.Tensor | None], list[StreamingAudioKVCache | None]]:
    """Encode one streaming unit per session in one encoder pass.

    Mirrors ``MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming``
    row by row: same CNN extra-context trim, same reset when the history would
    reach ``max_source_positions``, same learned positions, same pooled-length
    clip. Returns, per chunk, the ``[tokens, hidden]`` embeddings (``None`` when
    the unit is empty after trimming, which leaves its cache untouched) and the
    cache now holding that session's history.

    ``new_cache`` (default: a fresh paged :class:`StreamingAudioKVCache`) builds
    the cache of a new session or of a reset, from the row's current cache; the
    resident graph encoder passes one that hands out slot-pool caches.
    """
    implementation = getattr(encoder.config, "_attn_implementation", None) or "eager"
    weight = encoder.conv1.weight
    dtype, device = weight.dtype, weight.device
    max_positions = int(encoder.embed_positions.weight.shape[0])
    num_layers = len(encoder.layers)
    embed_dim = int(encoder.config.d_model)
    num_heads = int(encoder.config.encoder_attention_heads)

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
        # A new session, or the per-session path's reset: same bound, same
        # moment. The old cache is dropped only when the caller commits this one.
        if new_cache is not None and (cache is None or cache.length + length >= max_positions):
            cache = new_cache(cache)
        elif cache is None or cache.length + length >= max_positions:
            cache = StreamingAudioKVCache(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                max_positions=max_positions,
                page_positions=page_positions,
            )
        rows.append(
            _Row(
                index=index,
                frames=frames,
                start=prefix,
                length=length,
                pooled_length=pooled_length,
                cache=cache,
                past=cache.length,
            )
        )
        features.append(feature)
    if not rows:
        return outputs, caches

    offset = 0
    for row in rows:
        row.offset = offset
        offset += row.length
        row.cache.reserve(row.total, dtype=dtype, device=device)

    # --- CNN front end on the zero-padded batch -------------------------------
    mel = _stack_features(rows, features, device).to(dtype=dtype)
    hidden = nn.functional.gelu(encoder.conv1(mel))
    max_frames = int(mel.shape[-1])
    for slot, row in enumerate(rows):
        if row.frames < max_frames:
            # conv2 must see zeros past each row's end, as its own padding
            # would in the per-session call.
            hidden[slot, :, row.frames :].zero_()
    hidden = nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
    pieces = [hidden[slot, row.start : row.start + row.length] for slot, row in enumerate(rows)]
    hidden = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
    hidden = hidden + _position_rows(encoder, rows)

    # --- Transformer layers on the packed rows --------------------------------
    # Every per-row view is built once here, not per layer: at batch sizes
    # that matter the host cost of view ops, not the GPU, bounds this loop.
    views = [row.cache.layer_views(row.past, row.total) for row in rows]
    lengths = [row.length for row in rows]
    total_rows = sum(lengths)
    head_dim = embed_dim // num_heads
    # The per-session path passes an all-zero additive mask; keep it so each
    # row dispatches to the same attention kernel.
    masks = [torch.zeros((1, 1, row.length, row.total), dtype=dtype, device=device) for row in rows]
    for layer_index, layer in enumerate(encoder.layers):
        attention = layer.self_attn
        residual = hidden
        normed = layer.self_attn_layer_norm(hidden)
        query = attention.q_proj(normed) * attention.scaling
        key = attention.k_proj(normed)
        value = attention.v_proj(normed)
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

    # --- Projection and pooling ------------------------------------------------
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


__all__ = [
    "DEFAULT_KV_PAGE_POSITIONS",
    "StreamingAudioChunk",
    "StreamingAudioKVCache",
    "encode_streaming_audio_batch",
    "streaming_batch_unsupported_reason",
]
