# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local producer/consumer transport for native Ming audio decoding."""

from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .talker_batch import BatchedVAEState


class _CancelledError(Exception):
    pass


class MingStreamingJob:
    """Generate on a separate CUDA stream and wake the decoder for ready data.

    The producer marks the final latent, so even a one-chunk utterance can
    flush the VAE without submitting an empty latent or duplicating a frame.
    By default, wait for the polling interval and then drain ready latents.
    With low_latency_streaming, decode the first audio as soon as latents are
    ready, then coalesce within the polling budget and flush final latents
    immediately. This changes VAE chunk boundaries and is an explicit opt-in.
    The bounded queue applies backpressure to slow consumers.
    """

    def __init__(self, produce, vae, device, poll_seconds=0.1, *, low_latency_streaming=False):
        self.vae = vae
        self.device = device
        self.poll_seconds = float(poll_seconds)
        self.low_latency_streaming = low_latency_streaming
        self.queue = Queue(maxsize=32)
        self.cancelled = Event()
        self.done = Event()
        self.error = None
        self.past_key_values = None
        self.stream_state = (None, None, None)
        self.finished = False
        self._has_audio_output = False
        # The caller may have cloned input tensors on its current stream.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))

        def run():
            try:
                with (
                    torch.inference_mode(),
                    torch.accelerator.device_index(device.index),
                    torch.cuda.stream(torch.cuda.Stream(device)),
                ):
                    torch.cuda.current_stream().wait_event(ready)
                    produce(self._publish)
            except _CancelledError:
                pass
            except BaseException as exc:
                self.error = exc
            finally:
                self.done.set()

        self.thread = Thread(target=run, name="ming-latent-producer", daemon=True)
        self.thread.start()

    def _publish(self, latent, last, ready=None):
        # As in ming_flash.llm_job, publish only completed latent tensors.
        # A producer that already queued later work passes the latent's event.
        if ready is None:
            torch.cuda.current_stream().synchronize()
        else:
            ready.synchronize()
        while not self.cancelled.is_set():
            try:
                self.queue.put((latent, last), timeout=0.01)
                return
            except Full:
                continue
        raise _CancelledError()

    def next_audio(self):
        # Initial waiting and later coalescing share one budget per call.
        deadline = monotonic() + self.poll_seconds
        if not self.low_latency_streaming:
            # Preserve legacy Ming collection: wait once, then drain without
            # waiting for subsequent arrivals. VAE numerics can depend on the
            # resulting chunk boundaries, even when the latents are identical.
            self.cancelled.wait(self.poll_seconds)
        parts = []
        last = False
        while not last:
            if self.cancelled.is_set():
                raise _CancelledError()
            if self.error is not None:
                raise self.error
            try:
                latent, last = self.queue.get_nowait()
            except Empty:
                if self.done.is_set():
                    # Completion follows the last put. Recheck after observing
                    # done: the producer may have finished after our empty read.
                    try:
                        latent, last = self.queue.get_nowait()
                    except Empty as exc:
                        if self.cancelled.is_set():
                            raise _CancelledError()
                        if self.error is not None:
                            raise self.error
                        if not parts:
                            raise RuntimeError("Ming latent producer ended without a final latent") from exc
                        break
                else:
                    if not self.low_latency_streaming or (parts and not self._has_audio_output):
                        break
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    try:
                        # A put wakes this immediately. Once audio has started,
                        # collect subsequent latents only within this deadline.
                        latent, last = self.queue.get(timeout=remaining)
                    except Empty:
                        continue
            parts.append(latent)
        if self.cancelled.is_set():
            raise _CancelledError()
        if self.error is not None:
            raise self.error
        if not parts:
            return torch.empty(1, 1, 0, device=self.device), False
        audio, self.stream_state, self.past_key_values = self.vae.decode(
            torch.cat(parts, dim=1),
            past_key_values=self.past_key_values,
            stream_state=self.stream_state,
            use_cache=True,
            last_chunk=last,
        )
        self.finished = last
        if audio.numel():
            self._has_audio_output = True
        return audio, last

    def close(self):
        self.cancelled.set()
        # Wait for the current AR step to complete before another request
        # can reuse the CFM graph pool or the process-wide seeded RNG.
        self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.past_key_values = None
        self.stream_state = (None, None, None)


class MingBatchedStreamJob:
    """Consumer half of a request decoded by the shared talker batch engine.

    The engine thread publishes completed latents; on the model runner thread
    ``take`` drains everything that has arrived for a shared batched VAE
    decode (``mark_decoded`` records its result), or ``next_audio`` decodes
    it with this job's own VAE state. Waiting happens once per runner step for
    all requests (see ``urgent``), not per request. A request that is not
    streamed collects every latent and is decoded by ``decode_full`` when its
    final latent arrives.
    """

    def __init__(self, vae, device, *, decode_full=None):
        self.vae = vae
        self.device = device
        self.decode_full = decode_full
        self.request = None
        self.latents = deque()
        self.collected = []
        self.error = None
        self.final_published = False
        self.past_key_values = None
        self.stream_state = (None, None, None)
        # Stream position and buffers owned by a BatchedAudioVAEDecoder.
        self.vae_state: BatchedVAEState = {}
        self.finished = False
        self._has_audio_output = False

    def publish(self, latent, last):
        # Called from the engine thread after the producing step completed.
        self.latents.append((latent, last))
        if last:
            self.final_published = True

    def fail(self, exc):
        self.error = exc
        self.final_published = True

    def urgent(self):
        """Whether a waiting runner step should stop coalescing for this job."""
        if self.error is not None:
            return True
        if not self.latents:
            return False
        if self.decode_full is not None:
            return self.final_published
        # First audio as early as possible, the final flush immediately.
        return not self._has_audio_output or self.final_published

    def _drain(self):
        if self.error is not None:
            raise self.error
        parts = []
        last = False
        while self.latents and not last:
            latent, last = self.latents.popleft()
            if latent.is_cuda:
                # Produced on the engine stream; keep it alive for this stream's VAE.
                latent.record_stream(torch.cuda.current_stream(latent.device))
            parts.append(latent)
        # A request with no speakable segment is final without any latent.
        last = last or (self.final_published and not parts and not self.latents and not self.finished)
        return parts, last

    def take(self):
        """Return ``(latents or None, last)`` for a streamed job's next decode."""
        parts, last = self._drain()
        if not parts:
            self.finished = last
            return None, last
        return torch.cat(parts, dim=1), last

    def mark_decoded(self, audio, last):
        self.finished = last
        if audio.numel():
            self._has_audio_output = True

    def next_audio(self):
        if self.decode_full is not None:
            parts, last = self._drain()
            self.collected.extend(parts)
            if not last:
                return torch.empty(1, 1, 0, device=self.device), False
            self.finished = True
            return self.decode_full(self.collected), True
        latents, last = self.take()
        if latents is None:
            return torch.empty(1, 1, 0, device=self.device), last
        audio, self.stream_state, self.past_key_values = self.vae.decode(
            latents,
            past_key_values=self.past_key_values,
            stream_state=self.stream_state,
            use_cache=True,
            last_chunk=last,
        )
        self.mark_decoded(audio, last)
        return audio, last

    def close(self, engine=None, vae_decoder=None):
        if engine is not None and self.request is not None and not self.finished:
            engine.cancel(self.request)
        if vae_decoder is not None:
            vae_decoder.release(self.vae_state)
        self.latents.clear()
        self.collected = []
        self.past_key_values = None
        self.stream_state = (None, None, None)
