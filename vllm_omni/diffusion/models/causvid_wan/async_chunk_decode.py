# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-blocking chunk→pixel side path for CausVid (reuses WanStreamingDecoder).

DiT ``on_finished`` only enqueues / isends latents. A background thread on the
VAE owner runs ``WanStreamingDecoder.decode_chunk`` so decode never sits on
the compute critical path.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

logger = init_logger(__name__)

# Bound for cancel() waiting on the latent sender after posting the stop sentinel.
_SENDER_CANCEL_JOIN_TIMEOUT_S = 5.0


def _rescale_latents(vae: Any, latent: torch.Tensor) -> torch.Tensor:
    """Match whole-clip decode: apply latents_mean/std before the decoder."""
    device = next(vae.parameters()).device
    dtype = vae.dtype
    latent = latent.to(device=device, dtype=dtype, non_blocking=True)
    if latent.is_cuda:
        torch.cuda.current_stream(device).synchronize()
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return latent / inv_std + mean


def _async_host_copy(latent: torch.Tensor, *, shape: tuple[int, ...] | None, dtype: torch.dtype | None):
    """Non-blocking D2H into pinned host; returns ``(host, cuda_event_or_None)``."""
    src = latent.detach().contiguous()
    if shape is None:
        host = torch.empty_like(src, device="cpu", pin_memory=src.is_cuda)
    else:
        assert dtype is not None
        host = torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
    event = None
    if src.is_cuda:
        host.copy_(src, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
    else:
        host.copy_(src)
    return host, event


@dataclass
class AsyncStreamDecodeJob:
    """One request's async decode side path (start before ``run_cell_pipeline``)."""

    on_finished: Callable[[int, torch.Tensor], None] | None
    join: Callable[[], torch.Tensor | None]
    _send_works: list = field(default_factory=list)
    _cancel: Callable[[], None] | None = None
    # Client streaming: chunks were handed to ``on_chunk`` and are not accumulated.
    streams_to_client: bool = False
    _emitted: list[int] = field(default_factory=lambda: [0])

    @property
    def emitted_chunks(self) -> int:
        return int(self._emitted[0]) if self._emitted else 0

    def cancel(self) -> None:
        """Stop side threads when the request is aborted before ``join()``."""
        cancel = self._cancel
        self._cancel = None
        if cancel is not None:
            cancel()


def start_async_chunk_decode(
    *,
    vae: Any | None,
    decoder: WanStreamingDecoder | None,
    session_id: str,
    n_chunks: int,
    latent_shape: tuple[int, ...],
    dtype: torch.dtype,
    pp_rank: int,
    source_rank: int | None,
    pp_group: Any,
    stream_group: Any,
    on_chunk: Callable[[int, torch.Tensor], None] | None = None,
) -> AsyncStreamDecodeJob:
    """Wire producer (final denoise) → consumer (VAE owner) without blocking DiT.

    * Producer ``on_finished``: async D2H + queue; a send thread waits the copy
      then ``isend`` (or local queue for S=1 single-rank layer PP, G=1).
    * Consumer thread: ``irecv``/queue → rescale → ``decode_chunk``.
    * ``on_chunk`` is client streaming: each decoded chunk is handed off and
      not accumulated. ``join()`` then returns ``None``.
    * Without ``on_chunk``, ``join()`` waits and returns concatenated pixels
      on the owner, else ``None``. ``async_chunk_decode`` overlap does not by
      itself stream to the client.
    """
    if source_rank is None:
        return _start_local(vae=vae, decoder=decoder, session_id=session_id, on_chunk=on_chunk)
    if pp_group is None or source_rank == 0:
        raise ValueError("remote async_chunk_decode needs vertical S (D or D+1) with source_rank != 0")
    if stream_group is None:
        raise ValueError("remote async_chunk_decode requires a gloo stream process group")
    if vae is not None and (vae.use_tiling or vae.is_distributed_enabled()):
        # Intentional hard fail under streaming too: mid-request CHUNK_READY
        # rides async_chunk_decode and has no tiled/distributed VAE fallback.
        raise ValueError(
            "async_chunk_decode requires non-tiled single-owner VAE "
            "(streaming_output has no sync mid-request decode fallback)"
        )

    source = pp_group.ranks[source_rank]
    owner = pp_group.ranks[0]
    send_works: list = []

    if pp_rank == source_rank:
        send_q: queue.Queue = queue.Queue()
        stop = object()
        sent_count = [0]
        send_lock = threading.Lock()
        sender_cancelled = threading.Event()

        def _sender() -> None:
            while True:
                item = send_q.get()
                if item is stop:
                    break
                _chunk, host, event = item
                if event is not None:
                    event.synchronize()
                # Cancel may have timed out while we were stuck on the copy
                # event; never isend a late real latent after cancel abandons
                # the stream group (filler path only runs once sender exits).
                if sender_cancelled.is_set():
                    continue
                work = dist.isend(host, dst=owner, group=stream_group)
                with send_lock:
                    send_works.append((host, work))
                    sent_count[0] += 1

        sender_thread = threading.Thread(target=_sender, name="causvid-latent-send", daemon=True)
        sender_thread.start()

        def on_finished(chunk: int, latent: torch.Tensor) -> None:
            host, event = _async_host_copy(latent, shape=latent_shape, dtype=dtype)
            send_q.put((chunk, host, event))

        def _fill_remaining_sends() -> None:
            """Complete the owner's posted irecvs after a mid-request cancel."""
            with send_lock:
                already = sent_count[0]
            for _ in range(max(0, n_chunks - already)):
                filler = torch.zeros(latent_shape, device="cpu", dtype=dtype)
                work = dist.isend(filler, dst=owner, group=stream_group)
                with send_lock:
                    send_works.append((filler, work))
                    sent_count[0] += 1
                work.wait()

        def join() -> torch.Tensor | None:
            send_q.put(stop)
            sender_thread.join()
            for _host, work in send_works:
                work.wait()
            return None

        def cancel() -> None:
            sender_cancelled.set()
            send_q.put(stop)
            sender_thread.join(timeout=_SENDER_CANCEL_JOIN_TIMEOUT_S)
            if sender_thread.is_alive():
                # Do not fill: a stuck sender that later wakes could isend a
                # real latent after fillers and re-pollute the stream group.
                # Caller must stop/rebuild workers and the stream process group.
                raise RuntimeError(
                    "async_chunk_decode cancel: latent sender did not stop within "
                    f"{_SENDER_CANCEL_JOIN_TIMEOUT_S:.1f}s; stream group may be "
                    "unusable until workers are rebuilt "
                    f"(session_id={session_id!r})"
                )
            # Owner posted ``n_chunks`` irecvs up front; fill any gap so the
            # next request cannot match a leftover receive on this stream group.
            _fill_remaining_sends()

        return AsyncStreamDecodeJob(on_finished=on_finished, join=join, _send_works=send_works, _cancel=cancel)

    if pp_rank == 0:
        if vae is None or decoder is None:
            raise ValueError("VAE owner requires VAE + WanStreamingDecoder for async_chunk_decode")
        buffers = [torch.empty(latent_shape, device="cpu", dtype=dtype, pin_memory=True) for _ in range(n_chunks)]
        recv_works = [dist.irecv(buf, src=source, group=stream_group) for buf in buffers]
        pixels_out: list[torch.Tensor] = []
        error: list[BaseException] = []
        emitted = [0]
        cancelled = threading.Event()

        def _consumer() -> None:
            assert decoder is not None and vae is not None
            state = decoder.new_decode_state(session_id)
            try:
                for chunk_index, (buf, work) in enumerate(zip(buffers, recv_works)):
                    # Always wait: posted irecvs must complete or the next
                    # request's isend can match a stale receive and hang.
                    work.wait()
                    if cancelled.is_set():
                        for leftover in recv_works[chunk_index + 1 :]:
                            leftover.wait()
                        break
                    scaled = _rescale_latents(vae, buf)
                    pixels = decoder.decode_chunk(scaled, state).clamp(-1, 1)
                    _record_decoded_chunk(
                        pixels=pixels,
                        chunk_index=chunk_index,
                        on_chunk=on_chunk,
                        pixels_out=pixels_out,
                        emitted=emitted,
                    )
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
                decoder.release(state)
            else:
                decoder.release(state)

        thread = threading.Thread(target=_consumer, name="causvid-stream-decode", daemon=True)
        thread.start()

        def join() -> torch.Tensor | None:
            thread.join()
            if error:
                raise error[0]
            if on_chunk is not None:
                return None
            if not pixels_out:
                return None
            return torch.cat(pixels_out, dim=2)

        def cancel() -> None:
            cancelled.set()
            # Producer cancel fills remaining isends; wait for the consumer to
            # drain every posted irecv so the stream group is reusable.
            thread.join(timeout=30.0)
            if thread.is_alive():
                logger.warning(
                    "async_chunk_decode cancel: consumer still alive after 30s drain "
                    "(posted irecv may be stuck); session_id=%s",
                    session_id,
                )

        return AsyncStreamDecodeJob(
            on_finished=None,
            join=join,
            _send_works=send_works,
            _cancel=cancel,
            streams_to_client=on_chunk is not None,
            _emitted=emitted,
        )

    def join() -> torch.Tensor | None:
        return None

    return AsyncStreamDecodeJob(on_finished=None, join=join, _send_works=send_works)


def _record_decoded_chunk(
    *,
    pixels: torch.Tensor,
    chunk_index: int,
    on_chunk: Callable[[int, torch.Tensor], None] | None,
    pixels_out: list[torch.Tensor],
    emitted: list[int],
) -> None:
    """Hand a chunk to the client emitter, or keep it for the final ``cat``."""
    if on_chunk is not None:
        on_chunk(chunk_index, pixels)
        emitted[0] += 1
        return
    pixels_out.append(pixels)


def _start_local(
    *,
    vae: Any | None,
    decoder: WanStreamingDecoder | None,
    session_id: str,
    on_chunk: Callable[[int, torch.Tensor], None] | None = None,
) -> AsyncStreamDecodeJob:
    """Same-rank path: queue latents to a decode thread (S=1)."""
    if vae is None or decoder is None:
        raise ValueError("local async_chunk_decode requires VAE + WanStreamingDecoder on this rank")
    if vae.use_tiling or vae.is_distributed_enabled():
        raise ValueError(
            "async_chunk_decode requires non-tiled single-owner VAE "
            "(streaming_output has no sync mid-request decode fallback)"
        )

    latent_q: queue.Queue = queue.Queue()
    stop = object()
    pixels_out: list[torch.Tensor] = []
    error: list[BaseException] = []
    emitted = [0]

    def _consumer() -> None:
        assert decoder is not None and vae is not None
        state = decoder.new_decode_state(session_id)
        try:
            while True:
                item = latent_q.get()
                if item is stop:
                    break
                # WanCell passes session-absolute chunk ids; client streaming
                # and the engine expect request-local 0..n-1 indices (same as
                # the remote decode path's enumerate).
                _session_chunk, host, event = item
                del _session_chunk
                if event is not None:
                    event.synchronize()
                scaled = _rescale_latents(vae, host)
                pixels = decoder.decode_chunk(scaled, state).clamp(-1, 1)
                _record_decoded_chunk(
                    pixels=pixels,
                    chunk_index=emitted[0],
                    on_chunk=on_chunk,
                    pixels_out=pixels_out,
                    emitted=emitted,
                )
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
            decoder.release(state)
        else:
            decoder.release(state)

    thread = threading.Thread(target=_consumer, name="causvid-local-stream-decode", daemon=True)
    thread.start()

    def on_finished(chunk: int, latent: torch.Tensor) -> None:
        host, event = _async_host_copy(latent, shape=None, dtype=None)
        latent_q.put((chunk, host, event))

    def join() -> torch.Tensor | None:
        latent_q.put(stop)
        thread.join()
        if error:
            raise error[0]
        if on_chunk is not None:
            return None
        if not pixels_out:
            return None
        return torch.cat(pixels_out, dim=2)

    def cancel() -> None:
        latent_q.put(stop)
        thread.join(timeout=5.0)

    return AsyncStreamDecodeJob(
        on_finished=on_finished,
        join=join,
        _cancel=cancel,
        streams_to_client=on_chunk is not None,
        _emitted=emitted,
    )
