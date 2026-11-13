# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete ZAC decode graphs: RVQ → neural decoder → iSTFT/FIR → resampling.

Buckets pad both batch and RVQ time axes. Device valid lengths govern all state
commits and audio coverage; final is a per-row device input, not a graph key.
Only a bounded spectral overlap is recomputed, never the utterance prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

from .codec import ZACBatchedSynthesis, ZACDecoderNetwork, ZACDecoderState

logger = init_logger(__name__)


@dataclass
class _Bucket:
    graph: _ReplayGraph
    codes: torch.Tensor
    slots: torch.Tensor
    lengths: torch.Tensor
    final: torch.Tensor
    audio: torch.Tensor
    valid: torch.Tensor


class _ReplayGraph(Protocol):
    def replay(self) -> None: ...


class ZACCodecGraphWrapper:
    """Capture complete decode by (batch bucket, RVQ frame bucket).

    Neural and synthesis state have stable addresses. Capture and batch padding
    use dedicated non-leaseable slots. Replay outputs alias bucket storage; the
    manager copies valid PCM before another replay. Unsupported shapes execute
    the same padded-length implementation eagerly, preserving its state layout.
    """

    def __init__(
        self,
        network: ZACDecoderNetwork,
        state: ZACDecoderState,
        capacity: int,
        batch_sizes: tuple[int, ...],
        frame_sizes: tuple[int, ...],
    ) -> None:
        if not batch_sizes or not frame_sizes or any(type(n) is not int or n < 1 for n in (*batch_sizes, *frame_sizes)):
            raise ValueError("Zonos3 Codec graph buckets must contain positive integers")
        if max(batch_sizes) > capacity or state.frames.shape[0] < capacity + max(batch_sizes):
            raise ValueError("Zonos3 Codec graph state needs separate live and scratch slots")
        self.network, self.state, self.capacity = network, state, capacity
        self.batch_sizes, self.frame_sizes = sorted(set(batch_sizes)), sorted(set(frame_sizes))
        self.synthesis = ZACBatchedSynthesis(network, state.frames.shape[0])
        self.buckets: dict[tuple[int, int], _Bucket] = {}
        self.replay_count = 0
        self.eager_count = 0
        self._failed = False

    def reset(self, slots: torch.Tensor) -> None:
        self.synthesis.reset(slots)

    def frame_bucket(self, frames: int) -> int:
        """Round new RVQ storage up, without changing any row's valid length."""
        return next((n for n in self.frame_sizes if n >= max(1, frames)), max(1, frames))

    def _decode(
        self, codes: torch.Tensor, slots: torch.Tensor, lengths: torch.Tensor, final: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spectra = self.network(codes, slots, self.state, lengths)
        return self.synthesis(spectra, slots, lengths, final)

    @torch.inference_mode()
    def capture(self) -> None:
        """Warm and capture at model loading, before any serving requests exist."""
        if self.synthesis.device.type != "cuda":
            raise ValueError("Zonos3 Codec CUDA graph capture requires a CUDA device")
        if self.buckets or self._failed:
            raise RuntimeError("Zonos3 Codec graph capture may only run once")
        device = self.synthesis.device
        for batch in reversed(self.batch_sizes):
            for frames in reversed(self.frame_sizes):
                slots = torch.arange(self.capacity, self.capacity + batch, device=device)
                codes = torch.zeros(batch, frames, 10, dtype=torch.long, device=device)
                lengths = torch.full((batch,), frames, dtype=torch.long, device=device)
                final = torch.zeros(batch, dtype=torch.bool, device=device)
                result: tuple[torch.Tensor, torch.Tensor] | None = None

                def execute() -> None:
                    nonlocal result
                    result = self._decode(codes, slots, lengths, final)

                try:
                    for _ in range(3):
                        self.state.reset(slots)
                        self.reset(slots)
                        execute()
                    current_omni_platform.synchronize()
                    self.state.reset(slots)
                    self.reset(slots)
                    current_omni_platform.synchronize()
                    graph = current_omni_platform.capture_graph(execute)
                    if graph is None:
                        raise RuntimeError("CUDA platform did not capture Zonos3 Codec graph")
                    assert result is not None
                    self.buckets[batch, frames] = _Bucket(graph, codes, slots, lengths, final, *result)
                except Exception:
                    self._failed = True
                    raise  # Do not continue with a partially captured/poisoned allocator.
                self.state.reset(slots)
                self.reset(slots)
                logger.info("Captured complete Zonos3 Codec graph B=%d RVQ=%d (including synthesis)", batch, frames)

    @torch.inference_mode()
    def __call__(
        self, codes: torch.Tensor, slots: torch.Tensor, lengths: torch.Tensor, final: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._failed:
            raise RuntimeError("Zonos3 Codec graph capture failed; restart the worker")
        batch, frames = codes.shape[:2]
        b = next((n for n in self.batch_sizes if n >= batch), None)
        t = self.frame_bucket(frames)
        bucket = self.buckets.get((b, t))
        if bucket is None:
            self.eager_count += 1
            return self._decode(codes, slots, lengths, final)
        bucket.codes.zero_()
        bucket.codes[:batch, :frames].copy_(codes)
        bucket.lengths.zero_()
        bucket.lengths[:batch].copy_(lengths)
        bucket.final.zero_()
        bucket.final[:batch].copy_(final)
        bucket.slots.copy_(torch.arange(self.capacity, self.capacity + b, device=slots.device))
        bucket.slots[:batch].copy_(slots)
        bucket.graph.replay()
        self.replay_count += 1
        return bucket.audio[:batch], bucket.valid[:batch]
