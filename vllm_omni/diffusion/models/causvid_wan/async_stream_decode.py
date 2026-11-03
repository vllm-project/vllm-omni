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

from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder


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


def start_async_stream_decode(
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
) -> AsyncStreamDecodeJob:
    """Wire producer (final denoise) → consumer (VAE owner) without blocking DiT.

    * Producer ``on_finished``: async D2H + queue; a send thread waits the copy
      then ``isend`` (or local queue for S=1 single-rank layer PP, G=1).
    * Consumer thread: ``irecv``/queue → rescale → ``decode_chunk``.
    * ``join()`` waits side threads and returns concatenated pixels on owner,
      else ``None``.
    """
    if source_rank is None:
        return _start_local(vae=vae, decoder=decoder, session_id=session_id)
    if pp_group is None or source_rank == 0:
        raise ValueError("remote stream_decode needs vertical S (D or D+1) with source_rank != 0")
    if stream_group is None:
        raise ValueError("remote stream_decode requires a gloo stream process group")
    if vae is not None and (vae.use_tiling or vae.is_distributed_enabled()):
        raise ValueError("stream_decode requires non-tiled single-owner VAE")

    source = pp_group.ranks[source_rank]
    owner = pp_group.ranks[0]
    send_works: list = []

    if pp_rank == source_rank:
        send_q: queue.Queue = queue.Queue()
        stop = object()

        def _sender() -> None:
            while True:
                item = send_q.get()
                if item is stop:
                    break
                _chunk, host, event = item
                if event is not None:
                    event.synchronize()
                work = dist.isend(host, dst=owner, group=stream_group)
                send_works.append((host, work))

        sender_thread = threading.Thread(target=_sender, name="causvid-latent-send", daemon=True)
        sender_thread.start()

        def on_finished(chunk: int, latent: torch.Tensor) -> None:
            host, event = _async_host_copy(latent, shape=latent_shape, dtype=dtype)
            send_q.put((chunk, host, event))

        def join() -> torch.Tensor | None:
            send_q.put(stop)
            sender_thread.join()
            for _host, work in send_works:
                work.wait()
            return None

        return AsyncStreamDecodeJob(on_finished=on_finished, join=join, _send_works=send_works)

    if pp_rank == 0:
        if vae is None or decoder is None:
            raise ValueError("VAE owner requires VAE + WanStreamingDecoder for stream_decode")
        buffers = [torch.empty(latent_shape, device="cpu", dtype=dtype, pin_memory=True) for _ in range(n_chunks)]
        recv_works = [dist.irecv(buf, src=source, group=stream_group) for buf in buffers]
        pixels_out: list[torch.Tensor] = []
        error: list[BaseException] = []

        def _consumer() -> None:
            assert decoder is not None and vae is not None
            state = decoder.new_decode_state(session_id)
            try:
                for buf, work in zip(buffers, recv_works):
                    work.wait()
                    scaled = _rescale_latents(vae, buf)
                    pixels_out.append(decoder.decode_chunk(scaled, state).clamp(-1, 1))
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
            if not pixels_out:
                return None
            return torch.cat(pixels_out, dim=2)

        return AsyncStreamDecodeJob(on_finished=None, join=join, _send_works=send_works)

    def join() -> torch.Tensor | None:
        return None

    return AsyncStreamDecodeJob(on_finished=None, join=join, _send_works=send_works)


def _start_local(
    *,
    vae: Any | None,
    decoder: WanStreamingDecoder | None,
    session_id: str,
) -> AsyncStreamDecodeJob:
    """Same-rank path: queue latents to a decode thread (S=1)."""
    if vae is None or decoder is None:
        raise ValueError("local stream_decode requires VAE + WanStreamingDecoder on this rank")
    if vae.use_tiling or vae.is_distributed_enabled():
        raise ValueError("stream_decode requires non-tiled single-owner VAE")

    latent_q: queue.Queue = queue.Queue()
    stop = object()
    pixels_out: list[torch.Tensor] = []
    error: list[BaseException] = []

    def _consumer() -> None:
        assert decoder is not None and vae is not None
        state = decoder.new_decode_state(session_id)
        try:
            while True:
                item = latent_q.get()
                if item is stop:
                    break
                _chunk, host, event = item
                if event is not None:
                    event.synchronize()
                scaled = _rescale_latents(vae, host)
                pixels_out.append(decoder.decode_chunk(scaled, state).clamp(-1, 1))
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
        if not pixels_out:
            return None
        return torch.cat(pixels_out, dim=2)

    return AsyncStreamDecodeJob(on_finished=on_finished, join=join)
