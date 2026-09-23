# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn committed VAE chunks into MP4 bytes without materializing the video.

The producer side is the model's business: a Wan VAE walks a causal
frame-by-frame loop, a MiniMax-H3 VAE walks overlapping clips, and Wan S2V
decodes one clip per autoregressive iteration. What happens to a chunk once it
is final is not: every one of them quantizes to uint8, moves to the host once,
and lands in a bounded encoder. That consumer lives here so a model only has to
publish finished chunks -- :class:`SupportsChunkedVAEDecode` for producers that
can be driven, :class:`ChunkedVideoMP4Session` for producers that drive
themselves.

The move to the host runs as its own scheduled stage. A chunk is quantized on
the accelerator, copied without blocking into one of a fixed number of reusable
pinned host slots on a dedicated stream, and only read once its copy-completion
event has fired. The producer therefore returns to decoding while the copy is
still in flight, and the slot count bounds how many chunks can be in flight at
once.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import supports_chunked_vae_decode
from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder

logger = init_logger(__name__)

# Re-checked while waiting for a slot so an abort recorded without a notify
# still wakes the producer.
_SLOT_POLL_SECONDS = 0.05

_COPY_STREAMS: dict[tuple[str, int], torch.Stream] = {}
_COPY_STREAM_LOCK = threading.Lock()


def _copy_stream(device: torch.device) -> torch.Stream:
    """Return the process-wide chunk-copy stream for ``device``.

    One stream per device rather than one per request: copies are issued in
    order on it anyway, and a stream per request would churn accelerator
    resources for no ordering benefit.
    """
    with _COPY_STREAM_LOCK:
        key = (device.type, device.index or 0)
        stream = _COPY_STREAMS.get(key)
        if stream is None:
            stream = torch.Stream(device=device)
            _COPY_STREAMS[key] = stream
        return stream


def quantize_chunk(chunk: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
    """Quantize a ``BCTHW`` chunk to contiguous ``BTHWC`` uint8 on its own device.

    ``value_range`` is the interval the producer publishes, which differs per
    checkpoint (see ``SupportsChunkedVAEDecode.chunk_value_range``). Quantizing
    on the accelerator means the transfer that follows moves the final bytes
    rather than float frames, and making the result contiguous there keeps the
    copy a single flat move instead of a strided host-side gather.
    """
    low, high = float(value_range[0]), float(value_range[1])
    if high <= low:
        raise ValueError(f"value_range must be increasing, got {value_range!r}")
    scale = 255.0 / (high - low)
    frames = chunk.clamp(low, high).sub(low).mul(scale).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 4, 1).contiguous()


def chunk_to_uint8_frames(chunk: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    """Quantize a ``BCTHW`` chunk and block until it reaches the host.

    The path a session takes when no pinned ring is available, and the one a
    caller outside a session gets. A session on an accelerator uses
    :class:`PinnedChunkRing` instead so the copy does not stall the producer.
    """
    return quantize_chunk(chunk, value_range).cpu().numpy()


@dataclass
class ChunkTransferStats:
    """What the host transfer cost, for checking that it actually overlapped.

    ``copy_wait_seconds`` staying near zero while ``transfers`` climbs is the
    signal that copies completed behind the producer; it rising with them means
    the producer is outrunning the copy stream. ``slot_wait_seconds`` rising
    instead means the encoder is the bottleneck and backpressure has reached
    the producer.
    """

    transfers: int = 0
    slot_wait_seconds: float = 0.0
    copy_wait_seconds: float = 0.0
    peak_slots_in_use: int = 0
    peak_pending_bytes: int = 0


class _HostSlot:
    """One reusable pinned host buffer.

    The buffer is allocated on first use and never shrinks. ``batch_frames`` is
    a flush threshold rather than a cap, so a later batch can be larger than
    the first; the slot then keeps the larger buffer for the rest of the
    request instead of reallocating per chunk.
    """

    def __init__(self, *, pinned: bool) -> None:
        self._pinned = pinned
        self._buffer: torch.Tensor | None = None
        self.held_bytes = 0

    def reserve(self, frames: torch.Tensor) -> torch.Tensor:
        """Return a host view shaped like ``frames``, growing the buffer if needed."""
        needed = frames.numel()
        if self._buffer is None or self._buffer.numel() < needed:
            try:
                self._buffer = torch.empty(needed, dtype=frames.dtype, pin_memory=self._pinned)
            except Exception:
                # Pinning can be refused by the driver or unsupported on a
                # platform. A pageable buffer still copies correctly, only
                # synchronously, so degrade rather than fail the request.
                logger.warning("Pinned host buffer unavailable; chunk copies fall back to pageable memory")
                self._pinned = False
                self._buffer = torch.empty(needed, dtype=frames.dtype)
        self.held_bytes = needed * frames.element_size()
        return self._buffer[:needed].view(frames.shape)


class ChunkLease:
    """One chunk in flight: a host slot, its copy event, and its readers.

    The slot returns to the ring only after the last reader has finished with
    it. Being handed to an encoder is not enough -- the encoder's queue holds a
    view of this buffer until its muxing thread reads it -- so release is
    driven by the encoder rather than by dequeuing.
    """

    def __init__(
        self,
        *,
        ring: PinnedChunkRing,
        slot: _HostSlot,
        view: np.ndarray,
        source: torch.Tensor,
        landed: torch.Event | None,
        readers: int,
    ) -> None:
        self._ring = ring
        self._slot = slot
        self._view = view
        self._source: torch.Tensor | None = source
        self._landed = landed
        self._readers = readers
        self._lock = threading.Lock()

    def ready(self) -> bool:
        """Whether the copy has landed, without blocking on it."""
        return self._landed is None or self._landed.query()

    def wait(self) -> np.ndarray:
        """Block until the copy has landed, then hand over the host frames."""
        if self._landed is not None:
            started = time.perf_counter()
            self._landed.synchronize()
            self._ring.stats.copy_wait_seconds += time.perf_counter() - started
            self._landed = None
        # The source stays referenced until here so the allocator cannot hand
        # its device memory to the next chunk while the copy is still reading.
        self._source = None
        return self._view

    def release(self) -> None:
        """Report one reader done; the last one returns the slot to the ring."""
        with self._lock:
            if self._readers <= 0:
                return
            self._readers -= 1
            if self._readers > 0:
                return
        self._ring.reclaim(self._slot)

    def discard(self) -> None:
        """Drop an unread chunk during teardown, keeping the copy's memory alive.

        The copy may still be writing into the slot and reading from the source
        tensor, so this waits for it rather than letting either be reused
        underneath an in-flight transfer.
        """
        if self._landed is not None:
            self._landed.synchronize()
            self._landed = None
        self._source = None
        with self._lock:
            remaining, self._readers = self._readers, 0
        if remaining > 0:
            self._ring.reclaim(self._slot)


class PinnedChunkRing:
    """Fixed-depth pool of reusable host slots for chunk transfers.

    A slot walks ``free -> copying -> ready -> encoding -> free`` and holds
    exactly one chunk for that whole trip, so pending transfer bytes are
    bounded by the depth times the largest chunk of the request. Waiting for a
    free slot is where encoder backpressure reaches the producer.
    """

    def __init__(self, *, depth: int, device: torch.device) -> None:
        if depth <= 0:
            raise ValueError("transfer ring depth must be positive")
        self.depth = depth
        self.stats = ChunkTransferStats()
        self._on_device = device.type != "cpu"
        self._stream = _copy_stream(device) if self._on_device else None
        self._lock = threading.Lock()
        self._slot_freed = threading.Condition(self._lock)
        self._free = [_HostSlot(pinned=self._on_device) for _ in range(depth)]
        self._pending_bytes = 0
        self._failure: BaseException | None = None

    @property
    def slots_in_use(self) -> int:
        """Slots currently holding a chunk, whether copying, ready or encoding."""
        with self._lock:
            return self.depth - len(self._free)

    def transfer(self, frames: torch.Tensor, *, readers: int) -> ChunkLease:
        """Start one non-blocking copy of ``frames`` into a free slot."""
        if readers <= 0:
            raise ValueError("a transferred chunk needs at least one reader")
        slot = self._acquire()
        try:
            host = slot.reserve(frames)
            with self._lock:
                self._pending_bytes += slot.held_bytes
                self.stats.transfers += 1
                self.stats.peak_pending_bytes = max(self.stats.peak_pending_bytes, self._pending_bytes)
            landed = self._issue_copy(host, frames)
        except BaseException:
            self.reclaim(slot)
            raise
        return ChunkLease(ring=self, slot=slot, view=host.numpy(), source=frames, landed=landed, readers=readers)

    def abort(self, error: BaseException | None = None) -> None:
        """Fail every present and future wait for a slot."""
        with self._slot_freed:
            if self._failure is None:
                self._failure = error or RuntimeError("chunk transfer ring aborted")
            self._slot_freed.notify_all()

    def reclaim(self, slot: _HostSlot) -> None:
        """Return a slot whose last reader is done."""
        with self._slot_freed:
            self._pending_bytes -= slot.held_bytes
            slot.held_bytes = 0
            self._free.append(slot)
            self._slot_freed.notify()

    def _issue_copy(self, host: torch.Tensor, frames: torch.Tensor) -> torch.Event | None:
        if self._stream is None:
            host.copy_(frames)
            return None
        # The chunk was quantized on the compute stream, so the copy stream has
        # to wait for that work before reading it.
        source_ready = torch.Event()
        source_ready.record()
        self._stream.wait_event(source_ready)
        with self._stream:
            host.copy_(frames, non_blocking=True)
            landed = torch.Event()
            landed.record(self._stream)
        return landed

    def _acquire(self) -> _HostSlot:
        started = time.perf_counter()
        with self._slot_freed:
            while not self._free and self._failure is None:
                # ponytail: no request deadline yet, only an abort escape. A
                # unified deadline across producer, slot, queue and encoder
                # waits is RFC #6872 R4.
                self._slot_freed.wait(timeout=_SLOT_POLL_SECONDS)
            if self._failure is not None:
                raise self._failure
            slot = self._free.pop()
            in_use = self.depth - len(self._free)
            self.stats.peak_slots_in_use = max(self.stats.peak_slots_in_use, in_use)
        self.stats.slot_wait_seconds += time.perf_counter() - started
        return slot


class ChunkedVideoMP4Session:
    """Encode committed video chunks into one progressive MP4 per batch entry.

    Push finished ``BCTHW`` chunks as the producer commits them; each batch
    entry gets its own bounded encoder, so host transfer and H.264 encoding
    overlap whatever the producer is still decoding.

    ``audio_waveforms`` holds one waveform per batch entry (``None`` for a
    silent entry), so a caller with several outputs per prompt repeats a
    request's waveform across its entries. ``batch_frames`` coalesces transfers
    for producers that publish finer than a transfer is worth; ``crop`` trims
    the decoder's padding to the requested output size. ``transfer_slots`` sets
    how many chunks may be in flight to the host at once.
    """

    def __init__(
        self,
        *,
        value_range: tuple[float, float],
        fps: float,
        audio_waveforms: list[np.ndarray | None] | None = None,
        audio_sample_rate: int | None = None,
        batch_frames: int = 1,
        max_pending: int = 2,
        video_codec_options: dict[str, str] | None = None,
        crop: tuple[int, int] | None = None,
        transfer_slots: int = 2,
    ) -> None:
        if batch_frames <= 0:
            raise ValueError("batch_frames must be positive")
        self._value_range = value_range
        self._fps = fps
        self._audio_waveforms = audio_waveforms
        self._audio_sample_rate = audio_sample_rate
        self._batch_frames = batch_frames
        self._max_pending = max_pending
        self._video_codec_options = video_codec_options
        self._crop = crop
        self._transfer_slots = transfer_slots
        self._encoders: list[ChunkedMP4Encoder] = []
        self._pending: list[torch.Tensor] = []
        self._pending_frames = 0
        self._ring: PinnedChunkRing | None = None
        self._ring_resolved = False
        self._inflight: deque[ChunkLease] = deque()

    def push(self, chunk: torch.Tensor) -> None:
        """Queue one committed ``BCTHW`` chunk."""
        if self._crop is not None:
            height, width = self._crop
            chunk = chunk[..., :height, :width]
        self._pending.append(chunk)
        self._pending_frames += int(chunk.shape[2])
        if self._pending_frames >= self._batch_frames:
            self._flush()

    def finish(self) -> list[bytes]:
        """Flush what is pending and return one MP4 per batch entry."""
        self._flush()
        while self._inflight:
            self._feed_lease(self._inflight.popleft())
        return [encoder.finish() for encoder in self._encoders]

    def abort(self) -> None:
        if self._ring is not None:
            self._ring.abort()
        while self._inflight:
            self._inflight.popleft().discard()
        for encoder in self._encoders:
            encoder.abort()

    def transfer_stats(self) -> ChunkTransferStats | None:
        """Per-request transfer accounting, or ``None`` if no ring was used."""
        return None if self._ring is None else self._ring.stats

    @property
    def ring(self) -> PinnedChunkRing | None:
        """The transfer ring this session settled on, once a chunk has flushed."""
        return self._ring

    def _flush(self) -> None:
        if not self._pending:
            return
        frames = quantize_chunk(torch.cat(self._pending, dim=2), self._value_range)
        self._pending.clear()
        self._pending_frames = 0
        self._ensure_encoders(frames)
        ring = self._resolve_ring(frames.device)
        if ring is None:
            self._feed(frames.cpu().numpy(), release=None)
            return
        # Free a slot before asking for one, so the ring's own wait is only
        # ever for the encoder to catch up rather than for this loop.
        while len(self._inflight) >= ring.depth:
            self._feed_lease(self._inflight.popleft())
        self._inflight.append(ring.transfer(frames, readers=len(self._encoders)))
        while self._inflight and self._inflight[0].ready():
            self._feed_lease(self._inflight.popleft())

    def _feed_lease(self, lease: ChunkLease) -> None:
        try:
            self._feed(lease.wait(), release=lease.release)
        except BaseException:
            lease.discard()
            raise

    def _feed(self, frames: np.ndarray, *, release: Callable[[], None] | None) -> None:
        for index, encoder in enumerate(self._encoders):
            try:
                encoder.push(frames[index], on_consumed=release)
            except BaseException:
                # push() owns the callback for the entry it was given, so only
                # the encoders this chunk never reached still hold a reader.
                for _ in range(len(self._encoders) - index - 1):
                    if release is not None:
                        release()
                raise

    def _ensure_encoders(self, frames: torch.Tensor) -> None:
        if self._encoders:
            return
        self._encoders = [
            ChunkedMP4Encoder(
                width=int(frames.shape[3]),
                height=int(frames.shape[2]),
                fps=self._fps,
                audio_waveform=self._waveform_for(index, int(frames.shape[0])),
                audio_sample_rate=self._audio_sample_rate,
                max_pending=self._max_pending,
                video_codec_options=self._video_codec_options,
            )
            for index in range(int(frames.shape[0]))
        ]

    def _resolve_ring(self, device: torch.device) -> PinnedChunkRing | None:
        if self._ring_resolved:
            return self._ring
        self._ring_resolved = True
        if self._transfer_slots <= 0:
            return None
        try:
            self._ring = PinnedChunkRing(depth=self._transfer_slots, device=device)
        except Exception:
            logger.warning(
                "Pinned chunk transfers unavailable on %s; falling back to blocking copies",
                device,
                exc_info=True,
            )
        return self._ring

    def _waveform_for(self, index: int, batch_size: int) -> np.ndarray | None:
        if self._audio_waveforms is None:
            return None
        if len(self._audio_waveforms) != batch_size:
            raise ValueError(
                f"expected one audio waveform per batch entry, got "
                f"{len(self._audio_waveforms)} for {batch_size} entries"
            )
        return self._audio_waveforms[index]


def decode_to_mp4(vae: Any, z: torch.Tensor, **session_kwargs: Any) -> list[bytes]:
    """Decode ``z`` straight into one progressive MP4 per batch entry.

    Drives a VAE that declares :class:`SupportsChunkedVAEDecode`, so host
    transfer and encoding overlap the remaining decode and the full video is
    never materialized. Ranks that own no decode output receive no chunks and
    get an empty list, matching the empty tensor the full-decode path returns
    there.
    """
    if not supports_chunked_vae_decode(vae):
        raise TypeError(f"{type(vae).__name__} does not expose the chunked VAE decode capability")

    session = ChunkedVideoMP4Session(value_range=vae.chunk_value_range, **session_kwargs)
    try:
        vae.decode_with_chunks(z, on_chunk=session.push)
        videos = session.finish()
    except BaseException:
        session.abort()
        raise
    stats = session.transfer_stats()
    if stats is not None:
        logger.debug(
            "Chunked MP4 transfers=%d slot_wait=%.3fs copy_wait=%.3fs peak_slots=%d peak_bytes=%d",
            stats.transfers,
            stats.slot_wait_seconds,
            stats.copy_wait_seconds,
            stats.peak_slots_in_use,
            stats.peak_pending_bytes,
        )
    return videos
