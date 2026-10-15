# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Codec chunk decoding inside the MOSS-TTS-Local talker stage.

Each decode step appends one frame per stream to a device-resident ring. The
final frame of a stream is the frame whose code row is all padding: the talker
emits its stop token on that very step (as in the two-stage pipeline), so the
scheduler releases the request's KV blocks and batch slot at once. The step's
continue mask tells the host which frame that was once its copy completes; the
host never waits for it. Frames become *known* when their step's mask has
arrived, and only known audio frames are cut into chunks (the first chunk,
optional ramp chunks, then steady chunks).

Cut chunks wait in a ready queue and are decoded together on a side CUDA
stream beside the talker's next steps: a decode costs nearly the same for one
stream as for dozens (each of its ~1000 kernels has a fixed cost), so like the
two-stage codec stage it decodes once enough streams are ready, or once a chunk
has waited a few steps (sooner for a stream's first chunk). Audio is
delivered with the first step of its stream that runs after the decode
completed, so no step output waits for the codec.

A stream whose request stopped is *detached*: its pending frames are copied out
of the ring (the slot serves the next request), its remaining audio is cut and
decoded on later steps, also on steps without a batch, and handed to the tail
sender, which hands it to the scheduler to emit, in order, with the request's
terminal output that the scheduler held back. Aborted requests release their codec stream at once.

Every non-padding frame before the stop is decoded in order. Chunk boundaries
and BF16 rounding can change the waveform relative to the two-stage decoder.

With early first audio (``fused_early_first_audio``), a stream's first frame is
queued as its first chunk as soon as it is sampled, right after the final
prefill, instead of after its continue mask arrives: a padding frame is decoded
as code 0 and dropped by its validity on the device. That chunk rides the next
packed decode, which also starts the stream's codec state, and the first-audio
sender hands its PCM to the engine output queue once the decode completes,
ahead of the step output path. The stream's later outputs are marked so the
orchestrator holds them until that first frame has arrived.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu import buffer_utils

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder
    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender, _PreparedDelivery

    from .tail_audio_sender import TailAudioSender

logger = init_logger(__name__)

# Codec decodes allowed to overlap the talker before its stream waits for them.
_CODEC_LAG = 1


@dataclass
class FusedAudioStep:
    """Audio delivered with one step; the runner's streaming-audio interface."""

    num_reqs: int
    rows: list[int]
    pcm: list[torch.Tensor]
    sample_rate: torch.Tensor
    # Rows whose first frame was sent ahead by the first-audio sender.
    after_first: set[int] = field(default_factory=set)

    def to_cpu(
        self, copy_stream: torch.cuda.Stream, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> FusedAudioStep:
        return FusedAudioStep(
            self.num_reqs,
            self.rows,
            [copy_tensor(pcm) for pcm in self.pcm],
            self.sample_rate,
            after_first=self.after_first,
        )

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        parts: dict[int, list[torch.Tensor]] = {}
        for row, pcm in zip(self.rows, self.pcm, strict=True):
            if pcm.shape[-1]:
                parts.setdefault(row, []).append(pcm)
        output: list[dict[str, torch.Tensor] | None] = [None] * self.num_reqs
        for row, pieces in parts.items():
            pcm = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
            output[row] = {"model_outputs": pcm, "sr": self.sample_rate}
            if row in self.after_first:
                output[row][FIRST_AUDIO_REQUIRED_KEY] = torch.tensor(True)
        return output


@dataclass
class _KeepRecord:
    event: torch.cuda.Event
    host: torch.Tensor
    slots: list[int]
    req_ids: list[str | None]


@dataclass
class _Ready:
    req_id: str
    codes: torch.Tensor | None  # [n_vq, frames], owned; None for an empty last chunk
    last: bool
    urgent: bool
    step: int
    # Producer event of ``codes`` when it was not made on the codec stream.
    dep: torch.cuda.Event | None = None
    # Early first frame: routes frozen when the frame was sampled, and the
    # frame's validity on the device. Its PCM goes to the first-audio sender as
    # soon as this decode completes.
    first_delivery: _PreparedDelivery | None = None
    first_valid: torch.Tensor | None = None


@dataclass
class _Decoded:
    req_id: str
    pcm: torch.Tensor
    event: torch.cuda.Event


@dataclass
class _Stream:
    req_id: str
    # Frames [head, head + count) are pending, in the slot's ring row or, once
    # detached, in ``owned``; the first ``known`` of them are confirmed audio.
    head: int = 0
    count: int = 0
    known: int = 0
    chunks_emitted: int = 0
    # The padding frame was seen; frames after it are discarded.
    stopped: bool = False
    # The last chunk was cut.
    closed: bool = False
    # The first frame was cut when it was sampled, ahead of its continue mask.
    precut: bool = False
    owned: torch.Tensor | None = None
    owned_event: torch.cuda.Event | None = None


class MossFusedAudio:
    """Request-slot chunk state for the talker's in-process codec."""

    def __init__(self, codec: MossTTSCodecDecoder, capacity: int, n_vq: int, pad_id: int, device: torch.device) -> None:
        self.codec = codec
        self.pad_id = int(pad_id)
        self.chunk_frames = int(codec._stream_chunk_frames)
        self.initial_frames = int(codec._initial_stream_chunk_frames) or self.chunk_frames
        if self.chunk_frames <= 0:
            raise ValueError("The single-stage MOSS pipeline requires codec_chunk_frames > 0")
        ramp = codec._connector_extra().get("codec_chunk_ramp_frames", [])
        if not isinstance(ramp, (list, tuple)) or any(type(size) is not int or size <= 0 for size in ramp):
            raise ValueError("codec_chunk_ramp_frames must be a list of positive integers")
        self.chunk_schedule = (self.initial_frames, *ramp, self.chunk_frames)
        if ramp and any(later <= earlier for earlier, later in zip(self.chunk_schedule, self.chunk_schedule[1:])):
            raise ValueError("codec_chunk_ramp_frames must increase between the initial and steady chunk sizes")
        self.sample_rate = codec._sr_tensor
        self.device = device
        # Codec decodes, terminal resets and cancellations share this stream.
        self.stream = torch.cuda.Stream(device=device)
        codec._codec_stream = self.stream
        # fused_codec_precise_deps: a decode waits only for the data it reads
        # (chunk gathers on an otherwise idle stream, detach copies) instead of
        # everything queued on the talker stream, so an idle codec starts at once.
        self.precise_deps = bool(codec._connector_int("fused_codec_precise_deps", 0)) and device.type == "cuda"
        self.gather_stream = torch.cuda.Stream(device=device) if self.precise_deps else None
        # A chunk plus the frames whose masks are still in flight.
        self.ring = 4 * max(self.chunk_frames, self.initial_frames)
        self.pending = torch.zeros((capacity, self.ring, n_vq), dtype=torch.long, device=device)
        self.streams: list[_Stream | None] = [None] * capacity
        self.slot_of: dict[str, int] = {}
        self.detached: dict[str, _Stream] = {}
        self.tail_sender: TailAudioSender | None = None
        # Early first audio: requests whose first frame went to the sender,
        # until that frame is known; then those whose first frame was audio,
        # whose later outputs must follow it.
        self.early: set[str] = set()
        self.after_first: set[str] = set()
        self.first_sender: FirstAudioSender | None = None
        self._keep_records: list[_KeepRecord] = []
        self._keep_free: list[torch.Tensor] = []
        self._capacity = capacity
        self._decoded: list[_Decoded] = []
        # Codec work of recent launches. The talker's stream waits for decodes
        # older than ``_CODEC_LAG`` launches: a saturated GPU otherwise runs
        # the talker ahead and the codec, and every stream's audio, falls behind.
        self._decode_events: deque[torch.cuda.Event] = deque()
        self._ready: list[_Ready] = []
        self._steps = 0
        self._drain_steps = -1
        self.min_batch = max(1, codec._connector_int("fused_decode_min_batch", 32))
        self.max_wait = max(0, codec._connector_int("fused_decode_max_wait_steps", 3))
        self.first_wait = max(0, codec._connector_int("fused_decode_first_wait_steps", 1))
        # Packed decode (codec_packed_decode): one codec call takes every ready
        # chunk whatever its length, at most once per ``packed_interval`` steps.
        self.packed = bool(getattr(codec, "packed_decode_ready", False))
        self.packed_interval = max(1, codec._connector_int("fused_decode_packed_interval", 1))
        self.last_wait = max(0, codec._connector_int("fused_decode_last_wait_steps", self.max_wait))
        self._last_launch = -(1 << 30)
        self._step: FusedAudioStep | None = None
        # SMs the talker's cuBLASLt GEMMs leave free, process-wide from here on
        # (its CUDA graphs are captured later), so codec kernels on the side
        # stream run beside them. The decode-sized GEMMs are bandwidth-bound and
        # keep their speed with a few dozen SMs fewer. The codec's packed graphs
        # set their own carveout while they are captured.
        talker_carveout = codec._connector_int("talker_gemm_sm_carveout", 0)
        if talker_carveout and device.type == "cuda":
            torch._C._set_sm_carveout_experimental(talker_carveout)
            logger.info("MOSS single-stage talker GEMM SM carveout: %d", talker_carveout)
        # Per-step ring indices go through pinned device-mapped (UVA) pools,
        # one per upload site, each used once per step (see ``_UvaIndexPools``
        # in ``local_model_state``), instead of a pinned allocation + H2D copy.
        self._append_pool = self._gather_pool = None
        if device.type == "cuda" and is_uva_available():
            depth = buffer_utils._DEFAULT_MAX_CONCURRENCY + 2
            self._append_pool = buffer_utils.UvaBufferPool(3 * capacity, torch.int64, depth)
            self._gather_pool = buffer_utils.UvaBufferPool(2 * capacity * self.chunk_frames, torch.int64, depth)

    def add_first_frames(self, req_ids: Sequence[str], codes: torch.Tensor, valid: torch.Tensor) -> list[str]:
        """Queue just-sampled first frames ([n, n_vq] codes) as urgent first chunks.

        They are cut before their continue mask arrives: a padding frame is
        decoded as code 0 and the sender drops it by ``valid``. The ring still
        receives the frame with its step; ``_cut`` skips it once known.
        Returns the requests whose first frame the sender will deliver; the
        others keep the regular first chunk.
        """
        sender = self.first_sender
        if sender is None:
            return []
        ids = [str(req_id) for req_id in req_ids]
        delivery = sender.prepare([req_id for req_id in ids if req_id in self.slot_of])
        if not delivery.routes:
            return []
        frames = torch.where(valid[:, None], codes, 0)
        dep = torch.cuda.Event()
        dep.record()
        accepted = []
        for row, req_id in enumerate(ids):
            if req_id not in delivery.routes:
                continue
            self.streams[self.slot_of[req_id]].precut = True
            self.early.add(req_id)
            self._ready.append(
                _Ready(
                    req_id,
                    frames[row][:, None],
                    False,
                    True,
                    self._steps,
                    dep,
                    first_delivery=delivery,
                    first_valid=valid[row],
                )
            )
            accepted.append(req_id)
        return accepted

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, slot: int, req_id: str) -> None:
        previous = self.streams[slot]
        if previous is not None and self.slot_of.get(previous.req_id) == slot:
            self.slot_of.pop(previous.req_id)
        self.streams[slot] = _Stream(req_id)
        self.slot_of[req_id] = slot

    def finish(self, req_ids) -> None:
        """Detach stopped requests whose terminal output waits for their audio; release the rest."""
        released = []
        for req_id in map(str, req_ids):
            slot = self.slot_of.pop(req_id, None)
            if slot is None:
                if req_id in self.detached and not self.tail_sender.is_deferred(req_id):
                    self.detached.pop(req_id)
                    released.append(req_id)
                continue
            stream = self.streams[slot]
            self.streams[slot] = None
            if stream is not None and self.tail_sender is not None and self.tail_sender.is_deferred(req_id):
                self._detach(slot, stream)
            else:
                released.append(req_id)
        self._release(released)

    def preempt(self, req_id: str) -> None:
        # The scheduler reports preemption as a request error, not a resumable stop.
        slot = self.slot_of.pop(req_id, None)
        if slot is not None:
            self.streams[slot] = None
        self.detached.pop(req_id, None)
        self._release([req_id])

    def _release(self, req_ids: list[str]) -> None:
        if not req_ids:
            return
        gone = set(req_ids)
        self.early.difference_update(gone)
        self.after_first.difference_update(gone)
        self._decoded = [audio for audio in self._decoded if audio.req_id not in gone]
        self._ready = [chunk for chunk in self._ready if chunk.req_id not in gone]
        self.codec.on_requests_finished(req_ids)

    def _detach(self, slot: int, stream: _Stream) -> None:
        # The slot's ring row serves the next request: own the pending frames.
        if stream.count:
            # The pending frames are one ring range, wrapped at most once:
            # copy it with device slices, without a host index upload.
            row = self.pending[slot]
            end = stream.head + stream.count
            if end <= self.ring:
                stream.owned = row[stream.head : end].clone()
            else:
                stream.owned = torch.cat((row[stream.head :], row[: end - self.ring]))
            if self.precise_deps:
                stream.owned_event = torch.cuda.Event()
                stream.owned_event.record()
        stream.head = 0
        self.detached[stream.req_id] = stream
        # Audio decoded before the stop goes ahead of the tail, in order.
        kept = []
        for audio in self._decoded:
            if audio.req_id == stream.req_id:
                self.tail_sender.submit(
                    audio.req_id,
                    [(audio.pcm, audio.event)],
                    self.sample_rate,
                    False,
                    after_first=audio.req_id in self.after_first,
                )
            else:
                kept.append(audio)
        self._decoded = kept

    def _stream_of(self, slot: int, req_id: str | None) -> _Stream | None:
        if req_id is None:
            return None
        stream = self.streams[slot]
        if stream is not None and stream.req_id == req_id:
            return stream
        return self.detached.get(req_id)

    # ------------------------------------------------------------------
    # Per step
    # ------------------------------------------------------------------

    def _apply_keep_records(self) -> None:
        """Confirm frames of steps whose masks arrived, in step order."""
        done = 0
        for record in self._keep_records:
            if not record.event.query():
                break
            keep = record.host[: len(record.slots)].numpy()
            for index, (slot, req_id) in enumerate(zip(record.slots, record.req_ids, strict=True)):
                stream = self._stream_of(slot, req_id)
                if stream is None or stream.stopped:
                    continue
                if keep[index]:
                    stream.known += 1
                else:
                    # The padding frame and every frame after it are discarded.
                    stream.stopped = True
                    stream.count = stream.known
            self._keep_free.append(record.host)
            done += 1
        del self._keep_records[:done]

    def _record_keep(self, keep: torch.Tensor, entries: list[tuple[int, str | None]]) -> None:
        host = (
            self._keep_free.pop() if self._keep_free else torch.empty(self._capacity, dtype=torch.bool, pin_memory=True)
        )
        host[: keep.shape[0]].copy_(keep, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._keep_records.append(_KeepRecord(event, host, [s for s, _ in entries], [r for _, r in entries]))

    def _cut(self, stream: _Stream, final: bool) -> list[tuple[int, int, bool, bool]]:
        """(start, frames, last, urgent) chunks of known frames; advances the stream."""
        chunks = []
        while not stream.closed:
            if stream.precut and not stream.chunks_emitted:
                if stream.known < 1:
                    if not final:
                        break
                else:
                    # The in-band first frame was already decoded; it was audio.
                    stream.head = (stream.head + 1) % self.ring if stream.owned is None else stream.head + 1
                    stream.count -= 1
                    stream.known -= 1
                    stream.chunks_emitted += 1
                    stream.precut = False
                    if stream.req_id in self.early:
                        self.early.discard(stream.req_id)
                        self.after_first.add(stream.req_id)
                    continue
            threshold = self.chunk_schedule[min(stream.chunks_emitted, len(self.chunk_schedule) - 1)]
            if stream.known >= threshold and not (final and stream.known == stream.count == threshold):
                frames = threshold
            elif final:
                frames = stream.known
            else:
                break
            last = final and frames == stream.count
            # Startup chunks bridge a short playback buffer: use the first
            # chunk's bounded wait, rather than waiting for a steady batch.
            urgent = stream.chunks_emitted < len(self.chunk_schedule) - 1
            chunks.append((stream.head, frames, last, urgent))
            stream.head = (stream.head + frames) % self.ring if stream.owned is None else stream.head + frames
            stream.count -= frames
            stream.known -= frames
            stream.chunks_emitted += 1
            if last:
                stream.closed = True
        return chunks

    def _service_detached(self) -> None:
        """Cut the remaining audio of stopped requests once all their frames are known."""
        for stream in self.detached.values():
            if stream.closed or not (stream.stopped or stream.known == stream.count):
                continue
            for start, frames, last, _urgent in self._cut(stream, final=True):
                codes = stream.owned[start : start + frames].t() if frames else None
                # Batched like steady chunks: the stream holds no batch slot.
                self._ready.append(_Ready(stream.req_id, codes, last, False, self._steps, stream.owned_event))

    def _launch(self, main: torch.cuda.Stream, force: bool) -> None:
        ready = self._ready
        if not ready:
            return
        if force:
            batch = ready
        elif self.packed:
            if self._steps - self._last_launch < self.packed_interval:
                return
            steady = [chunk for chunk in ready if not chunk.urgent]
            due = (
                len(steady) >= self.min_batch
                or (bool(steady) and self._steps - steady[0].step >= self.max_wait)
                or any(chunk.urgent and self._steps - chunk.step >= self.first_wait for chunk in ready)
                # A stream's last chunk gates its client's next request.
                or any(chunk.last and self._steps - chunk.step >= self.last_wait for chunk in ready)
            )
            if not due:
                return
            batch = ready
        else:
            # First chunks (one frame) and steady chunks decode in separate
            # codec calls, each mostly fixed cost. A due first chunk must not
            # drag along a steady batch that is not due yet.
            steady = [chunk for chunk in ready if not chunk.urgent]
            due_steady = bool(steady) and (
                len(steady) >= self.min_batch or self._steps - steady[0].step >= self.max_wait
            )
            firsts = [chunk for chunk in ready if chunk.urgent]
            due_first = len(firsts) >= self.min_batch or any(
                self._steps - chunk.step >= self.first_wait for chunk in firsts
            )
            if due_steady and due_first:
                batch = ready
            elif due_first:
                batch = firsts
            elif due_steady:
                # A stream's chunks decode in order: take its queued first chunk along.
                with_first = {chunk.req_id for chunk in steady} & {chunk.req_id for chunk in firsts}
                batch = [chunk for chunk in ready if not chunk.urgent or chunk.req_id in with_first]
            else:
                return
        taken = {id(chunk) for chunk in batch}
        self._ready = [chunk for chunk in ready if id(chunk) not in taken]
        self._last_launch = self._steps
        if self.precise_deps:
            for chunk in batch:
                if chunk.dep is not None:
                    self.stream.wait_event(chunk.dep)
        else:
            self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            for chunk in batch:
                if chunk.codes is not None:
                    chunk.codes.record_stream(self.stream)
            # A stream may have several queued chunks: decode them in order, at
            # most one per stream in each codec call.
            rank: dict[str, int] = {}
            waves: list[list[tuple[int, _Ready]]] = []
            for position, chunk in enumerate(batch):
                if chunk.codes is None:
                    continue
                index = rank.get(chunk.req_id, 0)
                rank[chunk.req_id] = index + 1
                if index == len(waves):
                    waves.append([])
                waves[index].append((position, chunk))
            decoded: dict[int, torch.Tensor] = {}
            decode = self.codec._decode_streaming_packed if self.packed else self.codec._decode_streaming_batch
            for wave in waves:
                decoded.update(decode([(position, chunk.req_id, chunk.codes, chunk.last) for position, chunk in wave]))
            firsts = [(position, chunk) for position, chunk in enumerate(batch) if chunk.first_delivery is not None]
            if firsts:
                self._send_first_frames(firsts, decoded)
            empty_last = [chunk.req_id for chunk in batch if chunk.codes is None and chunk.last]
            if empty_last:
                self.codec.on_requests_finished(empty_last)
            event = torch.cuda.Event()
            event.record(self.stream)
        self._decode_events.append(event)
        for position, chunk in enumerate(batch):
            if chunk.first_delivery is not None:
                continue
            wav = decoded.get(position)
            if chunk.req_id in self.detached:
                pieces = [(wav, event)] if wav is not None else []
                if pieces or chunk.last:
                    self.tail_sender.submit(
                        chunk.req_id,
                        pieces,
                        self.sample_rate,
                        chunk.last,
                        after_first=chunk.req_id in self.after_first,
                    )
                if chunk.last:
                    self.detached.pop(chunk.req_id)
                    self.after_first.discard(chunk.req_id)
                    self.early.discard(chunk.req_id)
            elif wav is not None and chunk.req_id in self.slot_of:
                self._decoded.append(_Decoded(chunk.req_id, wav, event))

    def _send_first_frames(self, firsts, decoded) -> None:
        """Hand in-band first frames to the first-audio sender; runs on the codec stream."""
        from vllm_omni.worker_v2.first_audio_sender import _PreparedDelivery

        routes: dict[str, int] = {}
        for _, chunk in firsts:
            routes[chunk.req_id] = chunk.first_delivery.routes[chunk.req_id]
            chunk.first_valid.record_stream(self.stream)
        ids = list(routes)
        pcm = torch.stack([decoded[position] for position, _ in firsts])
        valid = torch.stack([chunk.first_valid for _, chunk in firsts])
        sink = firsts[0][1].first_delivery.sink
        self.first_sender.submit_prepared(_PreparedDelivery(sink, routes), ids, pcm, self.sample_rate, valid)

    def drain_detached_audio(self) -> None:
        """Advance stopped requests' audio; called on every step, with or without a batch."""
        if not self.detached:
            return
        self._apply_keep_records()
        self._service_detached()
        # Running batches launch decodes in step(). Without one since the last
        # drain the engine only steps for these streams: decode what is ready.
        idle = self._steps == self._drain_steps
        self._drain_steps = self._steps
        if idle:
            self._launch(torch.cuda.current_stream(self.device), force=True)

    def step(
        self,
        input_batch: InputBatch,
        decode_rows: list[int],
        codes: torch.Tensor,
        keep: torch.Tensor,
        upload: Callable[[np.ndarray], torch.Tensor],
    ) -> None:
        """Append this step's frames, queue chunk decodes and deliver finished audio."""
        if self.tail_sender is None:
            raise RuntimeError("The single-stage MOSS pipeline requires an in-process engine with a tail-audio sender")
        num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping_np[decode_rows].tolist()
        self._steps += 1
        self._apply_keep_records()
        main = torch.cuda.current_stream(self.device)
        while len(self._decode_events) > _CODEC_LAG:
            main.wait_event(self._decode_events.popleft())

        # Append this step's frames.
        append_rows: list[int] = []
        append_slots: list[int] = []
        append_pos: list[int] = []
        recorded: list[tuple[int, str | None]] = []
        for index, slot in enumerate(slots):
            stream = self.streams[slot]
            if stream is None or stream.stopped:
                recorded.append((slot, None))
                continue
            if stream.count >= self.ring:
                raise RuntimeError("MOSS single-stage codec ring overflow; continue masks are not arriving")
            append_rows.append(index)
            append_slots.append(slot)
            append_pos.append((stream.head + stream.count) % self.ring)
            stream.count += 1
            recorded.append((slot, stream.req_id))
        if append_rows:
            index = np.asarray([append_rows, append_slots, append_pos], dtype=np.int64).reshape(-1)
            pool = self._append_pool
            index = (pool.copy_to_uva(index) if pool is not None else upload(index)).view(3, -1)
            frames = codes.index_select(0, index[0])
            # Padding is outside the codebook; it is never cut, but keep the ring valid.
            self.pending.index_put_((index[1], index[2]), frames.masked_fill(frames.eq(self.pad_id), 0))
        self._record_keep(keep, recorded)

        # Cut first and steady chunks of known frames of running streams.
        gather_slots: list[int] = []
        gather_pos: list[int] = []
        cut: list[tuple[str, int, bool]] = []  # (request, frames, urgent)
        for slot in slots:
            stream = self.streams[slot]
            if stream is None or stream.closed:
                continue
            for start, frames, _last, urgent in self._cut(stream, final=False):
                gather_slots.extend([slot] * frames)
                gather_pos.extend((start + offset) % self.ring for offset in range(frames))
                cut.append((stream.req_id, frames, urgent))
        if cut:
            # One owned gather for every chunk: later steps rewrite the ring.
            index = np.asarray([gather_slots, gather_pos], dtype=np.int64).reshape(-1)
            pool = self._gather_pool
            dep = None
            if self.precise_deps and pool is not None:
                # Known frames were written by steps whose masks already arrived:
                # gather them beside the talker. Later steps rewrite these ring
                # positions on the talker stream, so it waits for the gather.
                index = pool.copy_to_uva(index).view(2, -1)
                with torch.cuda.stream(self.gather_stream):
                    rows_of_chunks = self.pending[index[0], index[1]]
                    dep = torch.cuda.Event()
                    dep.record(self.gather_stream)
                main.wait_event(dep)
            else:
                index = (pool.copy_to_uva(index) if pool is not None else upload(index)).view(2, -1)
                rows_of_chunks = self.pending[index[0], index[1]]
            offset = 0
            for req_id, frames, urgent in cut:
                chunk = rows_of_chunks[offset : offset + frames].t()
                self._ready.append(_Ready(req_id, chunk, False, urgent, self._steps, dep))
                offset += frames
        self._service_detached()
        self._launch(main, force=False)

        # Deliver audio whose decode completed to this step's rows. Audio of
        # a request that stopped on a step still in flight is forwarded by the
        # scheduler, which holds that request's terminal output.
        row_of = {req_id: row for row, req_id in enumerate(input_batch.req_ids[:num_reqs])}
        rows: list[int] = []
        pcm: list[torch.Tensor] = []
        after_first: set[int] = set()
        kept: list[_Decoded] = []
        ready: dict[int, bool] = {}
        for audio in self._decoded:
            row = row_of.get(audio.req_id)
            slot = self.slot_of.get(audio.req_id)
            if row is not None and slot is not None and self.streams[slot].stopped:
                row = None
            key = id(audio.event)
            if key not in ready:
                ready[key] = audio.event.query()
            if row is None or not ready[key]:
                kept.append(audio)
                continue
            rows.append(row)
            pcm.append(audio.pcm)
            if audio.req_id in self.after_first:
                after_first.add(row)
        self._decoded = kept
        self._step = FusedAudioStep(num_reqs, rows, pcm, self.sample_rate, after_first=after_first)

    def take_step(self, num_reqs: int) -> FusedAudioStep:
        step = self._step
        self._step = None
        if step is None:
            return FusedAudioStep(num_reqs, [], [], self.sample_rate)
        return step
