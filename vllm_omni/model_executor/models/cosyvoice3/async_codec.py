# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Append-only GPU token buffers and coalesced per-request codec progress.

The producer never waits for the codec to consume a step notification. A slow
codec catches up from immutable token/validity prefixes instead of a step FIFO.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_omni.platforms import current_omni_platform

from .cosyvoice3_code2wav import iter_decode_streaming_batch

if TYPE_CHECKING:
    from .cosyvoice3_code2wav import CosyVoice3Code2Wav, StreamingFlowItem, StreamingHiFTState


@dataclass
class StreamingDecodeState:
    """Pipeline-owned accumulation; the decoder cache is opaque to this layer."""

    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    emitted: int = 0
    hop: int = 25
    cache: StreamingHiFTState | None = None
    finished: bool = False  # Final window planned; GPU and client completion are tracked separately.


@dataclass
class SampleEntry:
    """CPU sampling metadata; token values remain in the supplied GPU tensor."""

    request_id: str
    position: int
    length_finished: bool
    conditioning: dict


@dataclass
class TokenBuffer:
    """Append-only storage: AR writes suffixes, codec confirms copied validity bits."""

    tokens: torch.Tensor
    mask: torch.Tensor
    host_mask: torch.Tensor
    conditioning: dict
    last_position: int = -1  # Producer-owned.
    limit_position: int | None = None
    confirmed_end: int = 0  # Consumer-owned, exclusive end of confirmed validity bits.


@dataclass
class TokenProgress:
    """Latest AR prefix and the GPU event that makes that prefix readable."""

    buffer: TokenBuffer
    end: int  # Exclusive end of the published token prefix.
    limit_position: int | None
    producer_done: torch.cuda.Event


@dataclass
class ControlRead:
    """Retain the source and pinned destination until the validity D2H completes."""

    progress: TokenProgress
    start: int
    copy_done: torch.cuda.Event


@dataclass
class DecodeProgress:
    """Confirmed token prefix and optional EOS/length-limit token position."""

    buffer: TokenBuffer
    length: int
    stop_position: int | None


@dataclass
class ReadyAudio:
    request_id: str
    pcm: torch.Tensor | None
    final: bool


@dataclass
class PendingAudio:
    """Retain GPU PCM until its pinned-host copy completes."""

    output: ReadyAudio
    copy_done: torch.cuda.Event
    source: torch.Tensor | None


class AsyncCodec:
    """AR submits prefixes; one codec thread consumes them and publishes CPU PCM.

    register/submit/cancel/release run on the engine thread. Token updates and
    retirements cross a short mailbox lock; reads, decoding and audio copies
    stay on the codec thread. Only completed D2H results enter outputs.
    """

    def __init__(
        self,
        decoder: CosyVoice3Code2Wav,
        device,
        max_model_len: int,
        pending_audio_threshold: int,
        chunk_size: int = 25,
        lookahead: int = 3,
    ):
        self.decoder = decoder
        if chunk_size <= 0 or lookahead < 0:
            raise ValueError("CosyVoice3 chunk size must be positive and lookahead nonnegative")
        self.chunk_size = chunk_size
        self.lookahead = lookahead
        self.states: dict[str, StreamingDecodeState] = {}
        self.device = device
        self.max_model_len = max_model_len
        self.outputs: queue.SimpleQueue[ReadyAudio] = queue.SimpleQueue()
        # Checked at batch admission; not a strict per-row bound.
        self.pending_audio_threshold = pending_audio_threshold
        self.buffers: dict[str, TokenBuffer] = {}
        # Cross-thread mailboxes coalesce updates instead of retaining steps.
        self.mailbox: dict[str, TokenProgress] = {}
        self.retirements: dict[str, tuple[bool, bool]] = {}
        self.lock = threading.Lock()
        # Codec-owned state; tombstones survive pending control/audio copies.
        self.cancelled: set[str] = set()
        self.ended: set[str] = set()
        self.released: set[str] = set()
        self.reads: dict[str, ControlRead] = {}
        self.progress: dict[str, DecodeProgress] = {}
        self.pending: deque[PendingAudio] = deque()
        self.error: BaseException | None = None
        self.started = threading.Event()
        self.tick = threading.Event()
        self.closed = False
        self.thread = threading.Thread(target=self.run, name="cosyvoice3-codec", daemon=True)
        self.thread.start()
        if not self.started.wait(10):
            raise TimeoutError("CosyVoice3 codec thread did not initialize")
        self.check_error()

    def check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError("CosyVoice3 asynchronous codec failed") from self.error

    def submit(
        self,
        entries: Sequence[SampleEntry],
        tokens: torch.Tensor,
        mask: torch.Tensor,
        rows: Sequence[int] | None = None,
    ) -> None:
        self.check_error()
        if self.closed:
            raise RuntimeError("CosyVoice3 codec is closed")
        updates = []
        source_rows = range(len(entries)) if rows is None else rows
        for row, entry in zip(source_rows, entries, strict=True):
            key = entry.request_id
            buffer = self.buffers.get(key)
            if buffer is None:
                if entry.position != 0:
                    raise ValueError("CosyVoice3 token buffer must start at position zero")
                buffer = TokenBuffer(
                    torch.empty((1, self.max_model_len), dtype=torch.long, device=tokens.device),
                    torch.empty(self.max_model_len, dtype=torch.bool, device=tokens.device),
                    torch.empty(self.max_model_len, dtype=torch.bool, pin_memory=True),
                    dict(entry.conditioning),
                )
                self.buffers[key] = buffer
            position = entry.position
            if buffer.limit_position is not None and position > buffer.limit_position:
                # The runner may already have scheduled a step beyond its CPU
                # length limit. It cannot extend the accepted speech prefix.
                continue
            if not buffer.last_position < position < self.max_model_len or position != buffer.last_position + 1:
                raise ValueError("CosyVoice3 token positions must be contiguous and within max_model_len")
            # Only new suffix positions are written. Codec reads frozen prefixes
            # directly; neither graph outputs nor an unbounded step queue are retained.
            buffer.tokens[:, position : position + 1].copy_(tokens[row : row + 1].reshape(1, 1))
            buffer.mask[position : position + 1].copy_(mask[row : row + 1])
            buffer.last_position = position
            if entry.length_finished and buffer.limit_position is None:
                buffer.limit_position = position
            updates.append((key, buffer, position + 1, buffer.limit_position))
        if not updates:
            return
        ready = torch.cuda.Event()
        ready.record()
        with self.lock:
            for key, buffer, end, limit in updates:
                self.mailbox[key] = TokenProgress(buffer, end, limit, ready)
        self.tick.set()

    def retire(self, request_id: str, *, cancel: bool, release: bool) -> None:
        self.check_error()
        key = request_id
        with self.lock:
            cancelled, released = self.retirements.get(key, (False, False))
            self.retirements[key] = (cancelled or cancel, released or release)
        if release:
            self.buffers.pop(key, None)
        self.tick.set()

    def cancel(self, request_id: str) -> None:
        self.retire(request_id, cancel=True, release=False)

    def release(self, request_id: str) -> None:
        self.retire(request_id, cancel=False, release=True)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.tick.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("CosyVoice3 codec did not drain during shutdown")
        self.check_error()

    def consume_retirements(self) -> None:
        with self.lock:
            controls, self.retirements = self.retirements, {}
            for key in controls:
                self.mailbox.pop(key, None)
        for key, (cancelled, released) in controls.items():
            if cancelled:
                self.cancelled.add(key)
            if released:
                self.released.add(key)
            self.progress.pop(key, None)
            self.finish_requests({key})
        # ControlRead keeps source flags and pinned destination alive until its
        # event completes, even if the request was cancelled in the meantime.

    def publish(self) -> None:
        while self.pending and self.pending[0].copy_done.query():
            audio = self.pending.popleft().output
            if audio.request_id not in self.cancelled:
                self.outputs.put(audio)
        pending = {x.output.request_id for x in self.pending}
        for key in self.released - pending - self.reads.keys():
            self.cancelled.discard(key)
            self.ended.discard(key)
            self.released.discard(key)

    def start_reads(self) -> None:
        """Copy only unread validity suffixes, ordered after AR prefix writes."""
        with self.lock:
            snapshots = [(key, value) for key, value in self.mailbox.items() if key not in self.reads]
            for key, _ in snapshots:
                self.mailbox.pop(key)
        with torch.cuda.stream(self.control_stream):
            for key, progress in snapshots:
                if key in self.ended or key in self.cancelled:
                    continue
                buffer = progress.buffer
                start = buffer.confirmed_end
                self.control_stream.wait_event(progress.producer_done)
                buffer.mask.record_stream(self.control_stream)
                buffer.host_mask[start : progress.end].copy_(buffer.mask[start : progress.end], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(self.control_stream)
                self.reads[key] = ControlRead(progress, start, copied)

    def poll_reads(self) -> None:
        """Confirm EOS on the CPU only after its validity copy is complete."""
        for key, read in list(self.reads.items()):
            if not read.copy_done.query():
                continue
            del self.reads[key]
            if key in self.cancelled or key in self.released or key in self.ended:
                continue
            progress = read.progress
            buffer = progress.buffer
            values = buffer.host_mask[read.start : progress.end].tolist()
            eos = next((read.start + i for i, value in enumerate(values) if not value), None)
            limit = progress.limit_position
            terminal = limit
            if eos is not None:
                terminal = eos if limit is None else min(eos, limit)
            length = progress.end
            if terminal is not None:
                length = terminal + int(values[terminal - read.start])
                self.ended.add(key)
            buffer.confirmed_end = progress.end
            self.progress[key] = DecodeProgress(buffer, length, terminal)

    def decode_ready(self) -> bool:
        """Submit ready chunks and their D2H without waiting for GPU completion."""
        items, bindings, empty_finals = [], [], []
        for key, progress in self.progress.items():
            if key in self.cancelled or key in self.released:
                continue
            buffer = progress.buffer
            buffer.tokens.record_stream(self.stream)
            self.initialize(key, buffer.conditioning, buffer.tokens)
            state = self.states[key]
            if state.finished:
                continue
            for tensor in (state.prompt_token, state.prompt_feat, state.embedding):
                tensor.record_stream(self.stream)
            item = self.prepare(key, progress.length, progress.stop_position)
            if item is not None:
                items.append(item)
                bindings.append((key, item["finalize"]))
            elif state.finished and progress.length == 0:
                empty_finals.append(key)
        if items:
            for index, wav, cache in iter_decode_streaming_batch(self.decoder, items):
                key, final = bindings[index]
                self.apply_result(key, cache)
                # Snapshot before the next request can reuse graph output storage.
                wav = wav.reshape(-1).float().clone()
                decoded = torch.cuda.Event()
                decoded.record(self.stream)
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_event(decoded)
                    host = torch.empty(wav.shape, dtype=wav.dtype, pin_memory=True)
                    host.copy_(wav, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record(self.copy_stream)
                    self.pending.append(PendingAudio(ReadyAudio(key, host, final), copied, wav))
                # Publish any completed copies between submissions, without
                # waiting for either HiFT or D2H before queuing the next row.
                self.publish()
        for key in empty_finals:
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
            self.pending.append(PendingAudio(ReadyAudio(key, None, True), copied, None))
        return bool(items or empty_finals)

    @torch.inference_mode()
    def run(self) -> None:
        try:
            current_omni_platform.set_device(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.control_stream = torch.cuda.Stream(device=self.device)
            self.started.set()
            with torch.cuda.stream(self.stream):
                while True:
                    self.tick.clear()
                    self.consume_retirements()
                    self.publish()
                    self.start_reads()
                    self.poll_reads()
                    decoded = self.decode_ready() if len(self.pending) < self.pending_audio_threshold else False
                    with self.lock:
                        incoming = bool(self.mailbox or self.retirements)
                    if self.closed and not incoming and not self.reads and not self.pending and not decoded:
                        break
                    if not decoded:
                        self.tick.wait(0.0005)
                self.stream.synchronize()
                self.copy_stream.synchronize()
                self.control_stream.synchronize()
                self.finish_requests(set(self.states))
                self.buffers.clear()
                self.progress.clear()
        except BaseException as error:
            self.error = error
        finally:
            self.started.set()

    def initialize(self, req_id: str, conditioning: Mapping, token_buffer: torch.Tensor) -> None:
        """Bind an append-only producer buffer; chunk/cache state stays consumer-owned."""
        from vllm_omni.data_entry_keys import to_struct

        from .cosyvoice3 import _normalize_request_conditioning
        from .utils import unpad_prompt_conditioning

        if req_id in self.states:
            return
        payload = to_struct(_normalize_request_conditioning({"embed": dict(conditioning)}))
        embed = payload.embed
        if embed is None or any(getattr(embed, key) is None for key in ("speech_token", "speech_feat", "embedding")):
            raise ValueError(f"CosyVoice3 single-stage request {req_id} has no prompt conditioning")
        prompt_token, prompt_feat = unpad_prompt_conditioning(
            embed.speech_token, embed.speech_feat, embed.speech_token_len
        )
        state = StreamingDecodeState(
            token=token_buffer,
            prompt_token=prompt_token[:1],
            prompt_feat=prompt_feat[:1],
            embedding=embed.embedding[:1],
            hop=self.chunk_size,
        )
        self.states[req_id] = state

    def prepare(self, req_id: str, length: int, stop_position: int | None = None) -> StreamingFlowItem | None:
        """Plan one chunk from a confirmed prefix, draining full chunks before its tail.

        stop_position is the terminal token's position, not the prefix length:
        length excludes EOS but includes the last speech token at a length stop.

        None means no decoder work: insufficient tokens, an already finished
        plan, or an empty terminal prefix. Only the last case newly sets
        state.finished; the executor sends its terminal notification without
        invoking the decoder. A returned item's finalize flag owns tail policy.
        """
        state = self.states[req_id]
        if state.finished:
            return None
        pad = (-state.prompt_token.shape[1]) % self.chunk_size if state.emitted == 0 else 0
        hop = state.hop + pad
        threshold = state.emitted + hop + self.lookahead
        final = stop_position is not None and stop_position < threshold
        if not final and length < threshold:
            return None
        if length == 0:
            state.finished = final
            return None
        prefix = length if final else threshold
        item: StreamingFlowItem = {
            "token": state.token[:, :prefix],
            "prompt_token": state.prompt_token,
            "prompt_feat": state.prompt_feat,
            "embedding": state.embedding,
            "cache_state": state.cache,
            "token_offset_tokens": state.emitted,
            "finalize": final,
        }
        if not final:
            state.emitted += hop
            state.hop = min(4 * self.chunk_size, 2 * state.hop)
        state.finished = final
        return item

    def apply_result(self, req_id: str, cache: StreamingHiFTState | None) -> None:
        state = self.states[req_id]
        state.cache = None if state.finished else cache

    def finish_requests(self, request_ids: Iterable[str]) -> None:
        for req_id in request_ids:
            self.states.pop(req_id, None)
