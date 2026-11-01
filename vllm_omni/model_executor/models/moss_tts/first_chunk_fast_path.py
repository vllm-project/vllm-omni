# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode a stream's first codec chunk as soon as Stage 1 receives it.

The Stage-1 engine loop only sees a new chunk between steps, and a step waits
for its codec graph to finish. A first chunk that arrives mid-step therefore
waits for the rest of that step, then for its own replay. This path takes the
first chunk from the receive thread, replays a small dedicated graph on its
own stream while the main step runs, and emits the audio straight to the
engine output queue. The main path later receives the same chunk, finds it
already decoded, and continues the stream from the state written here.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker_v2.first_audio_sender import FirstAudioSink, _PreparedDelivery

logger = init_logger(__name__)


class _SlotHandoff:
    """Orders the main path's first use of a slot after the fast decode."""

    __slots__ = ("done", "event", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.event: torch.cuda.Event | None = None
        self.error: Exception | None = None


@dataclass
class _Job:
    request_id: str
    request_key: str
    delivery: _PreparedDelivery
    slot: int
    codes: torch.Tensor  # (n_vq, T) int64 on host
    handoff: _SlotHandoff


class MossFirstChunkFastPath:
    def __init__(
        self,
        session: Any,
        wrapper: Any,
        *,
        n_vq: int,
        frames: int,
        codebook_size: int,
        samples_per_frame: int,
        n_channels: int,
        sample_rate: torch.Tensor,
        device: torch.device,
        gate_main: bool = False,
        max_active_streams: int = 0,
        handoff_timeout_s: float = 30.0,
    ) -> None:
        if not math.isfinite(handoff_timeout_s) or handoff_timeout_s <= 0:
            raise ValueError("handoff_timeout_s must be positive and finite")
        self._handoff_timeout_s = float(handoff_timeout_s)
        self._session = session
        # When set, the main codec stream waits for an in-flight fast decode
        # before its next replay, so the small first-chunk graph runs mostly
        # uncontended instead of interleaving with a large regular replay.
        self._gate_main = bool(gate_main)
        self._max_active_streams = max(0, int(max_active_streams))
        self._inflight: torch.cuda.Event | None = None
        self._wrapper = wrapper
        self._n_vq = int(n_vq)
        self.frames = int(frames)
        self._codebook_size = int(codebook_size)
        self._samples = int(frames) * int(samples_per_frame)
        self._n_channels = int(n_channels)
        self._sr = sample_rate
        if device.index is None:
            device = torch.device(device.type, torch.accelerator.current_device_index())
        self._device = device
        self._max_batch = max(wrapper.batch_sizes)
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._host_codes = torch.zeros((self._n_vq, self._max_batch, self.frames), dtype=torch.long).pin_memory()
        self._host_slots = torch.zeros(self._max_batch, dtype=torch.long).pin_memory()
        audio_shape = (self._max_batch, self._n_channels, self._samples)
        self._host_audio = torch.empty(audio_shape, dtype=torch.float32).pin_memory()
        self._copied = torch.cuda.Event()
        self._sink: FirstAudioSink | None = None
        # Request key -> slot for chunks decoded here and not yet seen by the main path.
        self._decoded: dict[str, int] = {}
        # A failed output-queue handoff must be replayed from the already
        # decoded samples. Re-decoding would advance the streaming state twice.
        self._unsent_audio: dict[str, torch.Tensor] = {}
        self._handoffs: dict[int, _SlotHandoff] = {}
        self._lock = threading.Lock()
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._worker_error: Exception | None = None
        self._emitted = 0
        self._batches = 0
        self._decode_s = 0.0

    def bind(self, sink: FirstAudioSink) -> None:
        if not callable(getattr(sink, "prepare", None)):
            raise TypeError("The codec first-chunk sink requires prepare(request_ids)")
        with self._lock:
            if self._closed:
                raise RuntimeError("The codec first-chunk decoder is closed")
            self._sink = sink
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="moss-codec-first-chunk", daemon=True)
                self._thread.start()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            # A stalled decode may still own GPU state: keep its thread and
            # handoff, rather than permitting a restart or slot reuse.
            if thread.is_alive():
                logger.warning("Codec first-chunk decoder is still stopping; slot waits remain bounded")

    # ------------------------------------------------------------------
    # Receive thread
    # ------------------------------------------------------------------
    def submit(
        self,
        request_id: str,
        request_key: str,
        codes_flat: Any,
        req_slots: dict[str, int],
    ) -> bool:
        """Claim a first chunk for fast decode; False leaves it to the main path."""
        # Own the codes: the job outlives this call and the payload it came from.
        codes = torch.as_tensor(codes_flat, dtype=torch.long).reshape(-1).clone()
        if codes.numel() != self._n_vq * self.frames:
            return False
        with self._lock:
            if (
                self._closed
                or self._sink is None
                or self._thread is None
                or not self._thread.is_alive()
                or request_key in req_slots
                or (self._max_active_streams and len(req_slots) >= self._max_active_streams)
            ):
                return False
            delivery = self._sink.prepare([request_id])
            if request_id not in delivery.routes:
                return False
            slot = self._session.acquire(oldest=True)
            if slot is None:
                return False
            handoff = _SlotHandoff()
            self._handoffs[slot] = handoff
            self._decoded[request_key] = slot
            req_slots[request_key] = slot
            codes = codes.reshape(self._n_vq, self.frames)
            # Enqueue under the admission lock: close's sentinel cannot
            # overtake an accepted job.
            self._jobs.put(_Job(request_id, request_key, delivery, slot, codes, handoff))
        return True

    # ------------------------------------------------------------------
    # Main (engine) thread
    # ------------------------------------------------------------------
    def get_request_slot(self, request_key: str, req_slots: dict[str, int]) -> int | None:
        """Observe a completed admission before deciding whether cleanup is needed.

        The scheduler removes cancelled routes before the runner's finish hook.
        A submit that already froze its route must finish leasing its slot
        before that hook checks it. Waiting for decode happens outside this
        lock, so the worker can still complete the slot handoff.
        """
        with self._lock:
            return req_slots.get(request_key)

    def take_decoded(self, request_key: str) -> bool:
        """True once per request whose first chunk this path already decoded."""
        with self._lock:
            return self._decoded.pop(request_key, None) is not None

    def take_unsent_audio(self, request_key: str) -> torch.Tensor | None:
        """Return first-chunk samples when the out-of-band sink rejected them."""
        with self._lock:
            return self._unsent_audio.pop(request_key, None)

    def forget(self, request_key: str) -> None:
        with self._lock:
            self._decoded.pop(request_key, None)
            self._unsent_audio.pop(request_key, None)

    def gate(self) -> None:
        """Queue the current stream behind the latest fast decode, if still running."""
        event = self._inflight
        if self._gate_main and event is not None and not event.query():
            torch.cuda.current_stream(self._device).wait_event(event)

    def order_after(self, slot: int) -> None:
        """Make the current stream's next use of ``slot`` follow the fast decode."""
        with self._lock:
            handoff = self._handoffs.get(slot)
        if handoff is None:
            return
        if not handoff.done.wait(timeout=self._handoff_timeout_s):
            # Keep the handoff. The worker may still be writing the slot, so
            # neither cancellation nor a retry may release/re-decode it.
            raise TimeoutError(f"Codec first-chunk slot {slot} did not complete within {self._handoff_timeout_s:g}s")
        if handoff.error is not None:
            raise RuntimeError(f"Codec first-chunk decode failed for slot {slot}") from handoff.error
        if handoff.event is not None:
            torch.cuda.current_stream(self._device).wait_event(handoff.event)
        with self._lock:
            self._handoffs.pop(slot, None)

    # ------------------------------------------------------------------
    # Decode thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            current_omni_platform.set_device(self._device)
            self._run_jobs()
        except Exception as error:
            logger.exception("MOSS codec first-chunk worker failed")
            self._worker_error = error
        finally:
            with self._lock:
                self._closed = True
                for handoff in self._handoffs.values():
                    if not handoff.done.is_set():
                        handoff.error = self._worker_error or RuntimeError("Codec first-chunk worker stopped")
                        handoff.done.set()
            # Accepted, unprocessed routes must not be left waiting for PCM.
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    self._fail_delivery(job)

    @staticmethod
    def _fail_delivery(job: _Job) -> None:
        try:
            job.delivery.fail([job.request_id])
        except Exception:
            logger.exception("Codec first-chunk error delivery failed for %s", job.request_id)

    def _run_jobs(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            jobs = [job]
            while len(jobs) < self._max_batch:
                try:
                    extra = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if extra is None:
                    self._jobs.put(None)
                    break
                jobs.append(extra)
            try:
                start = time.perf_counter()
                self._decode(jobs)
                self._decode_s += time.perf_counter() - start
                self._batches += 1
                before, self._emitted = self._emitted, self._emitted + len(jobs)
                if self._emitted // 1000 != before // 1000:
                    logger.info(
                        "MOSS codec first-chunk fast path: %d chunks in %d batches, %.2f ms/batch",
                        self._emitted,
                        self._batches,
                        1e3 * self._decode_s / self._batches,
                    )
            except Exception as error:
                logger.exception("MOSS codec first-chunk fast decode failed for %s", [j.request_id for j in jobs])
                # A partial graph replay may have advanced the slot. Surface
                # the failure instead of re-decoding and advancing it twice.
                for item in jobs:
                    item.handoff.error = error
                    item.handoff.done.set()
                    self._fail_delivery(item)
                raise

    @torch.no_grad()
    def _decode(self, jobs: list[_Job]) -> None:
        n = len(jobs)
        for row, item in enumerate(jobs):
            self._host_codes[:, row].copy_(item.codes)
            self._host_slots[row] = item.slot
        stream = self._stream
        with torch.cuda.stream(stream):
            for item in jobs:
                self._session.order_after_reset(item.slot, stream)
            codes = self._host_codes[:, :n].to(self._device, non_blocking=True).clamp_(0, self._codebook_size - 1)
            slots = self._host_slots[:n].to(self._device, non_blocking=True)
            result = self._wrapper.decode(codes, slots)
            if result is None:
                raise RuntimeError(f"no first-chunk codec graph for B={n}, T={self.frames}")
            audio = result[0][:n, ..., : self._samples].float()
            state_written = torch.cuda.Event()
            state_written.record(stream)
            self._inflight = state_written
            self._host_audio[:n].view_as(audio).copy_(audio, non_blocking=True)
            self._copied.record(stream)
        for item in jobs:
            item.handoff.event = state_written
        self._copied.synchronize()
        for row, item in enumerate(jobs):
            wav = self._host_audio[row].clone()
            if self._n_channels == 1:
                wav = wav.reshape(-1)
            try:
                item.delivery([item.request_id], [wav], self._sr)
            except Exception:
                logger.exception("MOSS codec first-chunk output failed for %s", item.request_id)
                with self._lock:
                    self._unsent_audio[item.request_key] = wav
            finally:
                item.handoff.done.set()
