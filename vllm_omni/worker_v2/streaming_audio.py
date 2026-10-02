# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned PCM accumulation for a stage that decodes audio in place."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY


@dataclass
class AudioRequest:
    samples_per_frame: int
    chunk_frames: int
    active: bool = True
    emitted: bool = False
    direct_first_pending: bool = False
    first_audio_required: bool = False
    pending: list[torch.Tensor | np.ndarray] = field(default_factory=list)
    pending_samples: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def accept_first_audio(self) -> None:
        with self.lock:
            if self.active:
                self.direct_first_pending = True

    def push(self, pcm: torch.Tensor | np.ndarray, ended: bool) -> torch.Tensor | None:
        with self.lock:
            if not self.active:
                return None
            size = pcm.numel() if isinstance(pcm, torch.Tensor) else pcm.size
            if self.direct_first_pending:
                self.direct_first_pending = False
                if size:
                    pcm = pcm[self.samples_per_frame :]
                    size = max(0, size - self.samples_per_frame)
                    self.emitted = True
                    self.first_audio_required = True
            if size:
                self.pending.append(pcm)
                self.pending_samples += size
            frames = self.pending_samples // self.samples_per_frame
            out = None
            if frames and (ended or not self.emitted or frames >= self.chunk_frames):
                if all(isinstance(part, np.ndarray) for part in self.pending):
                    array = np.concatenate(self.pending) if len(self.pending) > 1 else self.pending[0]
                    out = torch.from_numpy(array)
                else:
                    parts = [torch.from_numpy(part) if isinstance(part, np.ndarray) else part for part in self.pending]
                    out = torch.cat(parts) if len(parts) > 1 else parts[0]
                self.pending.clear()
                self.pending_samples = 0
                self.emitted = True
            if ended:
                self.active = False
                self.pending.clear()
                self.pending_samples = 0
            return out

    def finish(self) -> None:
        with self.lock:
            self.active = False
            self.pending.clear()
            self.pending_samples = 0


class StreamingAudioBuffer:
    def __init__(self, samples_per_frame: int, chunk_frames: int) -> None:
        self.samples_per_frame = samples_per_frame
        self.chunk_frames = chunk_frames
        self.requests: dict[str, AudioRequest] = {}

    def add(self, request_id: str) -> AudioRequest:
        if request_id not in self.requests:
            self.requests[request_id] = AudioRequest(self.samples_per_frame, self.chunk_frames)
        return self.requests[request_id]

    def finish(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            state = self.requests.pop(request_id, None)
            if state is not None:
                # Already queued outputs retain this object, never a fresh lookup
                # by ID. A late copy cannot resurrect a cancelled request.
                state.finish()


@dataclass
class StreamingAudioOutput:
    wav: torch.Tensor
    valid: torch.Tensor
    sample_rate: torch.Tensor
    event: torch.cuda.Event | None
    query_start_loc: np.ndarray
    requests: list[AudioRequest | None]
    length_end: np.ndarray

    def to_cpu(
        self,
        copy_stream: torch.cuda.Stream,
        copy_tensor: Callable[[torch.Tensor], torch.Tensor],
    ) -> StreamingAudioOutput:
        if self.event is not None:
            copy_stream.wait_event(self.event)
        return StreamingAudioOutput(
            copy_tensor(self.wav),
            copy_tensor(self.valid),
            self.sample_rate,
            None,
            self.query_start_loc,
            self.requests,
            self.length_end,
        )

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        valid = self.valid.numpy().astype(bool)
        # Each D2H batch owns its host allocation. NumPy slices retain that
        # backing tensor while a request accumulates PCM across steps; no
        # per-request Torch views are needed until an audio chunk is emitted.
        # Keep the tensor path for dtypes NumPy cannot represent (e.g. BF16).
        wav = self.wav.numpy() if self.wav.dtype in (torch.float16, torch.float32, torch.float64) else None
        output: list[dict[str, torch.Tensor] | None] = []
        required_marker = None
        for i, state in enumerate(self.requests):
            payload = None
            start, end = self.query_start_loc[i : i + 2]
            if state is not None and end > start:
                if wav is not None:
                    if end - start == 1:
                        pcm = wav[start:end].reshape(-1) if valid[start] else wav[0:0].reshape(-1)
                    else:
                        pcm = wav[start:end][valid[start:end]].reshape(-1)
                elif end - start == 1:
                    pcm = self.wav[start:end].reshape(-1) if valid[start] else self.wav.new_empty(0)
                else:
                    pcm = self.wav[start:end][torch.from_numpy(valid[start:end])].reshape(-1)
                ended = bool(self.length_end[i]) or not valid[end - 1]
                chunk = state.push(pcm, ended)
                if chunk is not None:
                    payload = {"model_outputs": chunk, "sr": self.sample_rate}
                if state.first_audio_required and (chunk is not None or ended):
                    if required_marker is None:
                        required_marker = torch.tensor(True)
                    if payload is None:
                        payload = {}
                    payload[FIRST_AUDIO_REQUIRED_KEY] = required_marker
            output.append(payload)
        return output
